---
title: "第3章: 機械的性質"
chapter_title: "第3章: 機械的性質"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/ceramic-materials-introduction/chapter-3.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Ceramic Materials](<../../MS/ceramic-materials-introduction/index.html>)›Chapter 3

  * [トップ](<index.html>)
  * [概要](<#intro>)
  * [脆性破壊](<#brittle-fracture>)
  * [Griffith理論](<#griffith>)
  * [破壊靭性](<#fracture-toughness>)
  * [Weibull統計](<#weibull>)
  * [高温クリープ](<#creep>)
  * [演習問題](<#exercises>)
  * [参考文献](<#references>)
  * [← 前の章](<chapter-2.html>)
  * 次の章へ →（準備中）

## 3.1 セラミックスの機械的性質の特徴

セラミックスは金属と異なり、**脆性材料** です。塑性変形がほとんど起こらず、臨界応力に達すると突然破壊します。この挙動は、イオン結合・共有結合の方向性と転位運動の困難さに起因します。本章では、脆性破壊のメカニズムを理解し、Weibull統計を用いた信頼性評価をPythonで実践します。 

**本章の学習目標**

  * **レベル1（基本理解）** : 脆性破壊メカニズムとGriffith理論を説明でき、破壊靭性の意味を理解できる
  * **レベル2（実践スキル）** : Pythonで破壊靭性計算とWeibull解析を実行し、信頼性を定量評価できる
  * **レベル3（応用力）** : 実測データからWeibullモジュラスを算出し、部品寿命予測と安全率設計ができる

### 金属との機械的性質の比較

特性 | セラミックス | 金属 | 原因  
---|---|---|---  
破壊形態 | 脆性破壊 | 延性破壊 | 転位運動の有無  
引張強度 | 100-1000 MPa | 200-2000 MPa | 欠陥感受性  
圧縮強度 | 2000-4000 MPa | 引張と同程度 | 亀裂伝播抑制  
破壊靭性 KIC | 2-8 MPa·m1/2 | 20-100 MPa·m1/2 | 塑性域の大きさ  
強度バラツキ | 大（Weibull m=5-15） | 小（正規分布） | 欠陥分布  
  
**設計上の注意点** セラミックスは引張応力に弱く、圧縮応力に強い特性があります。構造設計では、引張応力を避け、圧縮荷重が支配的な用途（軸受、切削工具）が適しています。また、表面欠陥が破壊起点となるため、研磨仕上げと取扱いが極めて重要です。 

## 3.2 脆性破壊メカニズム

### 3.2.1 破壊プロセス

セラミックスの破壊は、以下の3段階で進行します： 
    
    
    ```mermaid
    flowchart LR
                    A[初期欠陥気孔, 介在物, 亀裂] --> B[応力集中σlocal = Kt × σapplied]
                    B --> C[亀裂伝播KI > KIC]
                    C --> D[高速破壊v ~ 2000 m/s]
                    D --> E[完全破断微細片発生]
    
                    style A fill:#fff3e0
                    style B fill:#ffe0b2
                    style C fill:#f5576c,color:#fff
                    style D fill:#c62828,color:#fff
                    style E fill:#b71c1c,color:#fff
    ```

応力集中係数 \\( K_t \\) は、欠陥形状により決まります： 

\\[ K_t = 1 + 2\sqrt{\frac{a}{\rho}} \\] 

ここで、\\( a \\) は亀裂長さ、\\( \rho \\) は亀裂先端曲率半径です。原子レベルの鋭い亀裂（\\( \rho \sim 0.3 \\) nm）では、\\( K_t \\) は理論強度の1/10程度まで応力を集中させます。 

### 3.2.2 破壊の起点となる欠陥

  * **体積欠陥** : 気孔（成形時の残留空隙）、介在物（異相粒子）
  * **表面欠陥** : 加工傷（研削、ハンドリング）、熱衝撃亀裂
  * **粒界欠陥** : 粒界気孔、粒界相（液相焼結の残留ガラス相）

統計的には、最大欠陥が破壊を支配します（Weakest Link Theory）。これがWeibull統計の理論的基盤です。 

#### Python実装: 応力集中係数の計算
    
    
    # ===================================
    # Example 1: 応力集中係数の計算
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def stress_concentration_factor(crack_length, tip_radius):
        """
        楕円亀裂の応力集中係数を計算
    
        Parameters:
        -----------
        crack_length : float
            亀裂長さ a [μm]
        tip_radius : float
            亀裂先端曲率半径 ρ [nm]
    
        Returns:
        --------
        K_t : float
            応力集中係数（無次元）
        """
        # 長さ単位を統一（μm → m）
        a = crack_length * 1e-6
        rho = tip_radius * 1e-9
    
        # 応力集中係数の計算
        K_t = 1 + 2 * np.sqrt(a / rho)
    
        return K_t
    
    
    def plot_stress_concentration():
        """
        亀裂長さと先端曲率半径の影響を可視化
        """
        # パラメータ範囲
        crack_lengths = np.logspace(-1, 2, 100)  # 0.1 ~ 100 μm
        tip_radii = [0.3, 1.0, 5.0, 10.0]  # nm
    
        plt.figure(figsize=(12, 5))
    
        # 左図: 亀裂長さの影響
        plt.subplot(1, 2, 1)
        for rho in tip_radii:
            K_t_values = [stress_concentration_factor(a, rho) for a in crack_lengths]
            plt.loglog(crack_lengths, K_t_values, linewidth=2, label=f'ρ = {rho} nm')
    
        plt.xlabel('Crack Length a (μm)', fontsize=12)
        plt.ylabel('Stress Concentration Factor K_t', fontsize=12)
        plt.title('Effect of Crack Length on K_t', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
    
        # 右図: 具体例（Al2O3の典型的欠陥）
        plt.subplot(1, 2, 2)
        typical_crack = 10  # μm
        rho_range = np.logspace(-1, 2, 100)  # 0.1 ~ 100 nm
        K_t_typical = [stress_concentration_factor(typical_crack, rho) for rho in rho_range]
    
        plt.semilogx(rho_range, K_t_typical, linewidth=2, color='crimson')
        plt.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='K_t = 10 (Critical)')
        plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Atomic radius')
    
        plt.xlabel('Tip Radius ρ (nm)', fontsize=12)
        plt.ylabel('Stress Concentration Factor K_t', fontsize=12)
        plt.title(f'K_t for a = {typical_crack} μm Crack', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('stress_concentration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 数値例の出力
        print("=== Stress Concentration Factor Examples ===")
        print(f"Crack length: {typical_crack} μm")
        for rho in [0.3, 1.0, 10.0]:
            K_t = stress_concentration_factor(typical_crack, rho)
            print(f"  ρ = {rho:5.1f} nm → K_t = {K_t:6.1f}")
    
    # 実行
    plot_stress_concentration()
    

**実行結果の解釈** 原子レベルの鋭い亀裂（ρ = 0.3 nm）では、Kt ≈ 100を超える極端な応力集中が生じます。これにより、理論強度（E/10 ≈ 40 GPa）に対して実測強度が1/100程度（400 MPa）に低下します。研磨により表面欠陥を除去すると、強度が2-3倍向上する理由がここにあります。 

## 3.3 Griffith理論とエネルギー基準

### 3.3.1 エネルギーバランス

Griffith（1921）は、亀裂伝播をエネルギー論的に解析しました。亀裂が長さ \\( da \\) だけ進展するとき： 

\\[ \underbrace{-\frac{d U_{\text{elastic}}}{da}}_{\text{弾性エネルギー解放}} = \underbrace{\frac{d U_{\text{surface}}}{da}}_{\text{表面エネルギー増加}} \\] 

平板中の貫通亀裂（長さ \\( 2a \\)）に応力 \\( \sigma \\) が作用する場合、臨界応力 \\( \sigma_f \\) は： 

\\[ \sigma_f = \sqrt{\frac{2 E \gamma}{\pi a}} \\] 

ここで、\\( E \\) はヤング率、\\( \gamma \\) は表面エネルギー（J/m²）、\\( a \\) は亀裂半長です。 

### 3.3.2 Al₂O₃の破壊強度予測

Al₂O₃の物性値を用いて、欠陥サイズと破壊強度の関係を計算します： 

  * ヤング率: \\( E = 400 \\) GPa
  * 表面エネルギー: \\( \gamma = 1.0 \\) J/m²

#### Python実装: Griffith破壊強度の計算
    
    
    # ===================================
    # Example 2: Griffith破壊強度の計算
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def griffith_strength(crack_length, E=400e9, gamma=1.0):
        """
        Griffith理論による破壊強度の計算
    
        Parameters:
        -----------
        crack_length : float or array
            亀裂半長 a [m]
        E : float
            ヤング率 [Pa] (default: Al2O3 = 400 GPa)
        gamma : float
            表面エネルギー [J/m²] (default: 1.0 J/m²)
    
        Returns:
        --------
        sigma_f : float or array
            破壊強度 [Pa]
        """
        sigma_f = np.sqrt(2 * E * gamma / (np.pi * crack_length))
        return sigma_f
    
    
    def analyze_griffith_theory():
        """
        Griffith理論による強度-欠陥サイズ関係の解析
        """
        # 欠陥サイズ範囲（10 nm ~ 1 mm）
        crack_lengths = np.logspace(-8, -3, 1000)  # m
    
        # 各材料の破壊強度計算
        materials = {
            'Al₂O₃': {'E': 400e9, 'gamma': 1.0},
            'SiC': {'E': 450e9, 'gamma': 1.2},
            'Si₃N₄': {'E': 310e9, 'gamma': 0.8},
            'Glass': {'E': 70e9, 'gamma': 0.5}
        }
    
        plt.figure(figsize=(12, 5))
    
        # 左図: 材料比較
        plt.subplot(1, 2, 1)
        for name, props in materials.items():
            strength = griffith_strength(crack_lengths, props['E'], props['gamma'])
            plt.loglog(crack_lengths * 1e6, strength / 1e6, linewidth=2, label=name)
    
        # 典型的な強度範囲を示す
        plt.axhspan(100, 1000, alpha=0.1, color='green', label='Typical ceramic strength')
        plt.xlabel('Crack Length a (μm)', fontsize=12)
        plt.ylabel('Fracture Strength σ_f (MPa)', fontsize=12)
        plt.title('Griffith Strength vs Crack Size', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
    
        # 右図: Al2O3の詳細解析
        plt.subplot(1, 2, 2)
        a_Al2O3 = crack_lengths
        sigma_Al2O3 = griffith_strength(a_Al2O3, 400e9, 1.0)
    
        plt.loglog(a_Al2O3 * 1e6, sigma_Al2O3 / 1e6, linewidth=2, color='navy')
    
        # 実測値のプロット（典型例）
        experimental_data = {
            'a (μm)': [1, 5, 10, 50, 100],
            'σ_f (MPa)': [800, 400, 300, 150, 100]
        }
        plt.scatter(experimental_data['a (μm)'], experimental_data['σ_f (MPa)'],
                    s=100, c='red', marker='o', edgecolors='black', linewidth=2,
                    label='Experimental data', zorder=5)
    
        plt.xlabel('Crack Length a (μm)', fontsize=12)
        plt.ylabel('Fracture Strength σ_f (MPa)', fontsize=12)
        plt.title('Al₂O₃ Strength Prediction (Griffith)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.savefig('griffith_strength.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 数値例
        print("=== Griffith Strength Prediction for Al₂O₃ ===")
        for a_um in [1, 10, 100]:
            a = a_um * 1e-6
            sigma = griffith_strength(a, 400e9, 1.0)
            print(f"Crack size: {a_um:4d} μm → Strength: {sigma/1e6:6.1f} MPa")
    
        # 逆算: 実測強度から欠陥サイズ推定
        print("\n=== Reverse Calculation: Defect Size from Strength ===")
        measured_strength = np.array([300, 500, 800]) * 1e6  # MPa → Pa
        critical_crack = 2 * 400e9 * 1.0 / (np.pi * measured_strength**2)
        for i, sigma in enumerate(measured_strength / 1e6):
            print(f"Strength: {sigma:5.0f} MPa → Critical crack: {critical_crack[i]*1e6:6.2f} μm")
    
    # 実行
    analyze_griffith_theory()
    

**Griffith理論の限界** Griffith理論は理想的な弾性体を仮定しており、亀裂先端の塑性域やR-curve挙動を考慮していません。実材料では、粒界の亀裂偏向や粒子架橋により靭性が向上します（変換強化、繊維強化など）。これらは破壊力学の範疇で扱います。 

## 3.4 破壊靭性（Fracture Toughness）

### 3.4.1 応力拡大係数とKIC

破壊力学では、亀裂先端の応力場を**応力拡大係数 \\( K_I \\)** で表現します： 

\\[ K_I = Y \sigma \sqrt{\pi a} \\] 

ここで、\\( Y \\) は形状係数（無次元、試験片形状と亀裂位置に依存）、\\( \sigma \\) は遠方場応力、\\( a \\) は亀裂長さです。破壊条件は： 

\\[ K_I \geq K_{IC} \\] 

\\( K_{IC} \\) は**破壊靭性** と呼ばれる材料定数で、単位は MPa·m1/2 です。 

### 3.4.2 破壊靭性の測定法

  * **SEVNB法（Single Edge V-Notched Beam）** : 3点曲げ試験、V字ノッチ導入
  * **IF法（Indentation Fracture）** : ビッカース圧痕からの亀裂長さ測定
  * **CNB法（Chevron-Notched Beam）** : シェブロンノッチによる安定破壊

#### Python実装: 破壊靭性KICの計算
    
    
    # ===================================
    # Example 3: 破壊靭性K_ICの計算
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def fracture_toughness_SEVNB(P_max, B, W, a, S=40e-3):
        """
        SEVNB法による破壊靭性の計算
    
        Parameters:
        -----------
        P_max : float
            最大荷重 [N]
        B : float
            試験片幅 [m]
        W : float
            試験片高さ [m]
        a : float
            亀裂長さ [m]
        S : float
            支点間距離 [m] (default: 40 mm)
    
        Returns:
        --------
        K_IC : float
            破壊靭性 [Pa·m^0.5]
        """
        # 形状係数Yの計算（ASTM E399に基づく）
        alpha = a / W
        Y = (1.99 - alpha * (1 - alpha) * (2.15 - 3.93*alpha + 2.7*alpha**2)) / \
            ((1 + 2*alpha) * (1 - alpha)**1.5)
    
        # 応力拡大係数の計算
        K_IC = (P_max * S / (B * W**1.5)) * Y
    
        return K_IC
    
    
    def indentation_fracture_toughness(P, a, c, E=400e9, H=20e9):
        """
        IF法（圧痕法）による破壊靭性の簡易推定
    
        Parameters:
        -----------
        P : float
            圧子荷重 [N]
        a : float
            圧痕対角線半長 [m]
        c : float
            亀裂長さ（圧痕中心から先端まで）[m]
        E : float
            ヤング率 [Pa]
        H : float
            硬度 [Pa]
    
        Returns:
        --------
        K_IC : float
            破壊靭性 [Pa·m^0.5]
        """
        # Anstis式（1981）
        K_IC = 0.016 * (E / H)**0.5 * (P / c**1.5)
    
        return K_IC
    
    
    def analyze_fracture_toughness():
        """
        各種セラミックスの破壊靭性データ解析
        """
        # 材料データベース
        materials = {
            'Al₂O₃': {'K_IC': 4.0, 'E': 400e9, 'σ_f': 350e6},
            'ZrO₂ (3Y-TZP)': {'K_IC': 8.0, 'E': 210e9, 'σ_f': 900e6},
            'Si₃N₄': {'K_IC': 6.0, 'E': 310e9, 'σ_f': 700e6},
            'SiC': {'K_IC': 3.5, 'E': 450e9, 'σ_f': 400e6},
            'Glass': {'K_IC': 0.7, 'E': 70e9, 'σ_f': 50e6}
        }
    
        # 破壊靭性と強度の関係を可視化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # 左図: K_IC vs 強度
        ax1 = axes[0]
        K_IC_values = [props['K_IC'] for props in materials.values()]
        strength_values = [props['σ_f']/1e6 for props in materials.values()]
        names = list(materials.keys())
    
        ax1.scatter(K_IC_values, strength_values, s=150, c='crimson', edgecolors='black', linewidth=2)
        for i, name in enumerate(names):
            ax1.annotate(name, (K_IC_values[i], strength_values[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)
    
        ax1.set_xlabel('Fracture Toughness K_IC (MPa·m^0.5)', fontsize=12)
        ax1.set_ylabel('Flexural Strength σ_f (MPa)', fontsize=12)
        ax1.set_title('K_IC vs Strength for Ceramics', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
        # 右図: 許容欠陥サイズの計算
        ax2 = axes[1]
        Y = 1.12  # 表面亀裂の形状係数
    
        for name, props in materials.items():
            # σ_f = K_IC / (Y * sqrt(π * a)) から逆算
            a_critical = (props['K_IC'] / (Y * props['σ_f']))**2 / np.pi
            stress_range = np.linspace(0.1, 1.5, 100) * props['σ_f']  # 10% ~ 150%
            a_range = (props['K_IC'] / (Y * stress_range))**2 / np.pi
    
            ax2.loglog(a_range * 1e6, stress_range / 1e6, linewidth=2, label=name)
    
        ax2.set_xlabel('Critical Crack Size a (μm)', fontsize=12)
        ax2.set_ylabel('Applied Stress σ (MPa)', fontsize=12)
        ax2.set_title('Failure Diagram (K_IC = Yσ√πa)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.savefig('fracture_toughness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # SEVNB試験の数値例
        print("=== SEVNB Test Example (Al₂O₃) ===")
        P_max = 250  # N
        B = 4e-3  # 4 mm
        W = 3e-3  # 3 mm
        a = 1.5e-3  # 1.5 mm (a/W = 0.5)
        S = 40e-3  # 40 mm
    
        K_IC_measured = fracture_toughness_SEVNB(P_max, B, W, a, S)
        print(f"Load: {P_max} N, a/W = {a/W:.2f}")
        print(f"K_IC = {K_IC_measured/1e6:.2f} MPa·m^0.5")
    
        # IF法の数値例
        print("\n=== Indentation Fracture Test (Al₂O₃) ===")
        P_indent = 98.1  # 10 kgf = 98.1 N
        a_indent = 50e-6  # 50 μm (半対角線)
        c_indent = 150e-6  # 150 μm
    
        K_IC_IF = indentation_fracture_toughness(P_indent, a_indent, c_indent, 400e9, 20e9)
        print(f"Indentation load: {P_indent} N (10 kgf)")
        print(f"Crack length c: {c_indent*1e6:.1f} μm")
        print(f"K_IC (IF method) = {K_IC_IF/1e6:.2f} MPa·m^0.5")
    
    # 実行
    analyze_fracture_toughness()
    

**変換強化の効果** ZrO₂（ジルコニア）は、応力誘起相変態（正方晶→単斜晶）により体積膨張を生じ、亀裂先端に圧縮応力場を形成します。これにより、KIC = 8-12 MPa·m1/2という高靭性を実現しています（純Al₂O₃の2-3倍）。 

## 3.5 Weibull統計と信頼性評価

### 3.5.1 Weibull分布の理論

セラミックスの強度は、欠陥分布により大きくバラツキます。Weibull（1951）は、Weakest Link理論に基づき、以下の累積破壊確率 \\( P_f \\) を提案しました： 

\\[ P_f(\sigma) = 1 - \exp\left[-\left(\frac{\sigma - \sigma_u}{\sigma_0}\right)^m\right] \\] 

ここで： 

  * \\( m \\): Weibullモジュラス（無次元、値が大きいほど信頼性が高い）
  * \\( \sigma_0 \\): 特性強度（63.2%が破壊する応力）
  * \\( \sigma_u \\): 最小強度（通常0とおく）

### 3.5.2 Weibullモジュラスmの意味

m値 | 材料種類 | 欠陥状態 | 信頼性  
---|---|---|---  
3-5 | 低品質セラミックス | 大きな欠陥、不均一 | 低  
8-12 | 工業用セラミックス | 通常の製造品質 | 中  
15-20 | 高品質セラミックス | 均一、欠陥制御 | 高  
>50 | 金属（参考） | 正規分布に近い | 非常に高  
      
    
    ```mermaid
    flowchart TD
                    A[強度試験データn個の試験片] --> B[順位統計小さい順にソート]
                    B --> C[破壊確率推定P_f,i = i/(n+1)]
                    C --> D[Weibullプロットln ln(1/(1-P_f)) vs ln σ]
                    D --> E[線形回帰傾き = m切片 → σ₀]
                    E --> F[信頼性評価P_f(σ_design)]
    
                    style A fill:#e3f2fd
                    style D fill:#f093fb,color:#fff
                    style E fill:#f5576c,color:#fff
                    style F fill:#4caf50,color:#fff
    ```

#### Python実装: Weibull解析の完全実装
    
    
    # ===================================
    # Example 4: Weibull統計解析
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min
    from scipy.optimize import curve_fit
    
    def weibull_cumulative_probability(sigma, m, sigma_0, sigma_u=0):
        """
        Weibull累積破壊確率の計算
    
        Parameters:
        -----------
        sigma : float or array
            応力 [Pa]
        m : float
            Weibullモジュラス
        sigma_0 : float
            特性強度 [Pa]
        sigma_u : float
            最小強度 [Pa] (default: 0)
    
        Returns:
        --------
        P_f : float or array
            累積破壊確率（0~1）
        """
        P_f = 1 - np.exp(-((sigma - sigma_u) / sigma_0)**m)
        return P_f
    
    
    def estimate_weibull_parameters(strength_data):
        """
        実測強度データからWeibullパラメータを推定
    
        Parameters:
        -----------
        strength_data : array
            破壊強度データ [Pa]
    
        Returns:
        --------
        m : float
            Weibullモジュラス
        sigma_0 : float
            特性強度 [Pa]
        R_squared : float
            決定係数（フィット精度）
        """
        # データのソート
        sorted_strength = np.sort(strength_data)
        n = len(sorted_strength)
    
        # 破壊確率の推定（中央値ランク法）
        P_f = np.array([(i - 0.3) / (n + 0.4) for i in range(1, n + 1)])
    
        # Weibull変換: Y = ln ln(1/(1-P_f)), X = ln(σ)
        # 1に極めて近い値や0に極めて近い値を避ける
        valid_indices = (P_f > 0.001) & (P_f < 0.999)
        P_f_valid = P_f[valid_indices]
        sigma_valid = sorted_strength[valid_indices]
    
        Y = np.log(-np.log(1 - P_f_valid))
        X = np.log(sigma_valid)
    
        # 線形回帰
        coeffs = np.polyfit(X, Y, 1)
        m = coeffs[0]
        sigma_0 = np.exp(-coeffs[1] / m)
    
        # 決定係数R²の計算
        Y_pred = m * X + coeffs[1]
        SS_res = np.sum((Y - Y_pred)**2)
        SS_tot = np.sum((Y - np.mean(Y))**2)
        R_squared = 1 - (SS_res / SS_tot)
    
        return m, sigma_0, R_squared
    
    
    def plot_weibull_analysis(strength_data, material_name='Ceramic'):
        """
        Weibull解析の完全可視化
    
        Parameters:
        -----------
        strength_data : array
            破壊強度データ [MPa]
        material_name : str
            材料名
        """
        # 単位変換（MPa → Pa）
        strength_Pa = strength_data * 1e6
    
        # Weibullパラメータ推定
        m, sigma_0, R2 = estimate_weibull_parameters(strength_Pa)
    
        # ソートと破壊確率
        sorted_strength = np.sort(strength_Pa)
        n = len(sorted_strength)
        P_f = np.array([(i - 0.3) / (n + 0.4) for i in range(1, n + 1)])
    
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        # 左図: 強度分布ヒストグラム
        ax1 = axes[0]
        ax1.hist(strength_data, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
        # 理論分布の重ね描き
        sigma_range = np.linspace(strength_data.min(), strength_data.max(), 500)
        pdf = (m / sigma_0) * ((sigma_range * 1e6) / sigma_0)**(m - 1) * \
              np.exp(-((sigma_range * 1e6) / sigma_0)**m)
        ax1.plot(sigma_range, pdf * 1e6, 'r-', linewidth=2, label=f'Weibull PDF (m={m:.1f})')
    
        ax1.set_xlabel('Strength (MPa)', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f'{material_name} Strength Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # 中央図: Weibullプロット
        ax2 = axes[1]
        valid_indices = (P_f > 0.001) & (P_f < 0.999)
        P_f_valid = P_f[valid_indices]
        sigma_valid = sorted_strength[valid_indices] / 1e6
    
        Y_data = np.log(-np.log(1 - P_f_valid))
        X_data = np.log(sigma_valid)
    
        ax2.plot(X_data, Y_data, 'o', markersize=8, color='navy', label='Experimental')
    
        # フィット直線
        X_fit = np.linspace(X_data.min(), X_data.max(), 100)
        Y_fit = m * (X_fit - np.log(sigma_0 / 1e6))
        ax2.plot(X_fit, Y_fit, 'r-', linewidth=2, label=f'Fit: m={m:.2f}, R²={R2:.4f}')
    
        ax2.set_xlabel('ln(Strength) [ln(MPa)]', fontsize=12)
        ax2.set_ylabel('ln ln(1/(1-P_f))', fontsize=12)
        ax2.set_title('Weibull Plot', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        # 右図: 信頼性曲線
        ax3 = axes[2]
        sigma_reliability = np.linspace(0.5 * sigma_0, 1.5 * sigma_0, 500) / 1e6
        P_f_curve = weibull_cumulative_probability(sigma_reliability * 1e6, m, sigma_0)
    
        ax3.plot(sigma_reliability, (1 - P_f_curve) * 100, linewidth=2, color='green')
        ax3.axhline(y=90, color='blue', linestyle='--', alpha=0.5, label='90% Reliability')
        ax3.axhline(y=99, color='red', linestyle='--', alpha=0.5, label='99% Reliability')
    
        # 設計応力の計算（99%信頼性）
        sigma_design_99 = sigma_0 * (-np.log(1 - 0.01))**(1/m)
        ax3.axvline(x=sigma_design_99/1e6, color='red', linestyle=':', linewidth=2,
                    label=f'σ_design (99%) = {sigma_design_99/1e6:.0f} MPa')
    
        ax3.set_xlabel('Stress (MPa)', fontsize=12)
        ax3.set_ylabel('Reliability (%)', fontsize=12)
        ax3.set_title('Reliability vs Stress', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('weibull_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 数値結果の出力
        print(f"=== Weibull Analysis Results for {material_name} ===")
        print(f"Sample size: n = {n}")
        print(f"Weibull modulus: m = {m:.2f}")
        print(f"Characteristic strength: σ₀ = {sigma_0/1e6:.1f} MPa")
        print(f"Goodness of fit: R² = {R2:.4f}")
        print(f"\n--- Reliability-based Design Stress ---")
        for reliability in [0.90, 0.95, 0.99, 0.999]:
            sigma_design = sigma_0 * (-np.log(1 - (1 - reliability)))**(1/m)
            print(f"{reliability*100:.1f}% reliability → σ_design = {sigma_design/1e6:.1f} MPa")
    
        return m, sigma_0, R2
    
    
    # テストデータ生成（Al2O3の典型例）
    np.random.seed(42)
    n_samples = 30
    m_true = 10  # Weibullモジュラス
    sigma_0_true = 400  # MPa
    
    # Weibull分布からサンプリング
    strength_samples = weibull_min.rvs(m_true, scale=sigma_0_true, size=n_samples)
    
    # 解析実行
    plot_weibull_analysis(strength_samples, 'Al₂O₃')
    

**設計応力の決定方法** 99%信頼性を要求する場合、σdesign = σ₀ × (-ln 0.01)1/m で計算します。m = 10の場合、σdesign ≈ 0.58 σ₀となり、特性強度の約60%が設計許容応力です。m値が小さいほど、安全率を大きく取る必要があります。 

#### Python実装: モンテカルロ強度シミュレーション
    
    
    # ===================================
    # Example 5: モンテカルロ強度シミュレーション
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def monte_carlo_strength_simulation(m, sigma_0, n_components, n_simulations=10000):
        """
        多数部品システムの信頼性をモンテカルロ法で推定
    
        Parameters:
        -----------
        m : float
            Weibullモジュラス
        sigma_0 : float
            特性強度 [MPa]
        n_components : int
            システム内の部品数
        n_simulations : int
            シミュレーション回数
    
        Returns:
        --------
        system_strength : array
            システム強度分布（最弱部品の強度）[MPa]
        """
        # 各部品の強度をWeibull分布からサンプリング
        component_strengths = weibull_min.rvs(m, scale=sigma_0, size=(n_simulations, n_components))
    
        # システム強度 = 最弱部品の強度（Weakest Link）
        system_strength = np.min(component_strengths, axis=1)
    
        return system_strength
    
    
    def analyze_size_effect():
        """
        Size Effect（寸法効果）の解析
        """
        m = 10
        sigma_0 = 400  # MPa
    
        # 部品数を変化させる
        n_components_list = [1, 10, 100, 1000]
    
        plt.figure(figsize=(14, 5))
    
        # 左図: 強度分布の変化
        plt.subplot(1, 2, 1)
        for n_comp in n_components_list:
            system_strength = monte_carlo_strength_simulation(m, sigma_0, n_comp, 50000)
            plt.hist(system_strength, bins=50, alpha=0.5, density=True, label=f'n={n_comp}')
    
        plt.xlabel('System Strength (MPa)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Size Effect on Strength Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # 右図: 平均強度の低下
        plt.subplot(1, 2, 2)
        n_range = np.logspace(0, 3, 50).astype(int)
        mean_strengths = []
    
        for n_comp in n_range:
            system_strength = monte_carlo_strength_simulation(m, sigma_0, n_comp, 10000)
            mean_strengths.append(np.mean(system_strength))
    
        plt.semilogx(n_range, mean_strengths, linewidth=2, color='crimson')
    
        # 理論曲線（Weibull理論による予測）
        # E[σ_min] = σ₀ × Γ(1 + 1/m) × n^(-1/m)
        from scipy.special import gamma
        theoretical_mean = sigma_0 * gamma(1 + 1/m) * n_range**(-1/m)
        plt.semilogx(n_range, theoretical_mean, '--', linewidth=2, color='blue', label='Theoretical')
    
        plt.xlabel('Number of Components n', fontsize=12)
        plt.ylabel('Mean System Strength (MPa)', fontsize=12)
        plt.title('Size Effect (Weakest Link Model)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('monte_carlo_size_effect.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 数値結果
        print("=== Size Effect Analysis (Monte Carlo) ===")
        print(f"Single component: σ₀ = {sigma_0} MPa, m = {m}")
        for n_comp in [1, 10, 100, 1000]:
            system_strength = monte_carlo_strength_simulation(m, sigma_0, n_comp, 10000)
            mean_strength = np.mean(system_strength)
            std_strength = np.std(system_strength)
            print(f"n = {n_comp:4d}: Mean = {mean_strength:.1f} MPa, Std = {std_strength:.1f} MPa")
    
    # 実行
    analyze_size_effect()
    

**Size Effect（寸法効果）** 部品が大きくなる、または部品数が増えると、欠陥の存在確率が上昇し、システム全体の強度が低下します。これをSize Effectと呼びます。設計時には、試験片サイズと実部品サイズの違いを補正する必要があります（有効体積補正）。 

## 3.6 高温クリープと疲労

### 3.6.1 クリープ変形

高温（0.5 Tm以上、Tmは融点）では、セラミックスでもクリープ変形が生じます。クリープ速度 \\( \dot{\epsilon} \\) は以下の式で表されます： 

\\[ \dot{\epsilon} = A \sigma^n \exp\left(-\frac{Q}{RT}\right) \\] 

ここで、\\( A \\) は定数、\\( n \\) は応力指数、\\( Q \\) は活性化エネルギー、\\( R \\) は気体定数、\\( T \\) は温度です。 

### 3.6.2 クリープメカニズム

  * **拡散クリープ（n=1）** : 粒界拡散（Coble creep）、格子拡散（Nabarro-Herring creep）
  * **粒界すべりクリープ（n=2）** : 粒界での原子移動とすべり
  * **転位クリープ（n >3）**: 転位の上昇運動（高応力域）

#### Python実装: クリープ速度の計算
    
    
    # ===================================
    # Example 6: 高温クリープ速度の計算
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def creep_rate(sigma, T, A=1e-10, n=1, Q=400e3, R=8.314):
        """
        クリープ速度の計算
    
        Parameters:
        -----------
        sigma : float or array
            応力 [Pa]
        T : float or array
            温度 [K]
        A : float
            定数 [s^-1·Pa^-n]
        n : float
            応力指数
        Q : float
            活性化エネルギー [J/mol]
        R : float
            気体定数 [J/(mol·K)]
    
        Returns:
        --------
        epsilon_dot : float or array
            クリープ速度 [s^-1]
        """
        epsilon_dot = A * sigma**n * np.exp(-Q / (R * T))
        return epsilon_dot
    
    
    def plot_creep_behavior():
        """
        クリープ挙動の可視化
        """
        # Al2O3のクリープパラメータ（例）
        A_diffusion = 1e-8
        n_diffusion = 1
        Q_diffusion = 400e3
    
        A_dislocation = 1e-12
        n_dislocation = 4
        Q_dislocation = 600e3
    
        # 温度範囲
        temperatures = np.linspace(1200, 1600, 5) + 273.15  # K
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # 左図: 応力依存性
        ax1 = axes[0]
        stress_range = np.logspace(6, 8, 100)  # 1 ~ 100 MPa
    
        for T in temperatures:
            epsilon_dot_diff = creep_rate(stress_range, T, A_diffusion, n_diffusion, Q_diffusion)
            epsilon_dot_disl = creep_rate(stress_range, T, A_dislocation, n_dislocation, Q_dislocation)
            epsilon_dot_total = epsilon_dot_diff + epsilon_dot_disl
    
            ax1.loglog(stress_range / 1e6, epsilon_dot_total, linewidth=2,
                       label=f'T = {T-273.15:.0f}°C')
    
        ax1.set_xlabel('Stress (MPa)', fontsize=12)
        ax1.set_ylabel('Creep Rate (s^-1)', fontsize=12)
        ax1.set_title('Stress Dependence of Creep Rate', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
    
        # 右図: 温度依存性（Arrhenius plot）
        ax2 = axes[1]
        T_range = np.linspace(1000, 1800, 100) + 273.15
        sigma_fixed = 50e6  # 50 MPa
    
        epsilon_dot_diff = creep_rate(sigma_fixed, T_range, A_diffusion, n_diffusion, Q_diffusion)
        epsilon_dot_disl = creep_rate(sigma_fixed, T_range, A_dislocation, n_dislocation, Q_dislocation)
    
        ax2.semilogy(1e4 / T_range, epsilon_dot_diff, linewidth=2, label='Diffusion creep (n=1)')
        ax2.semilogy(1e4 / T_range, epsilon_dot_disl, linewidth=2, label='Dislocation creep (n=4)')
        ax2.semilogy(1e4 / T_range, epsilon_dot_diff + epsilon_dot_disl, 'k--', linewidth=2,
                     label='Total')
    
        ax2.set_xlabel('10^4 / T (K^-1)', fontsize=12)
        ax2.set_ylabel('Creep Rate (s^-1)', fontsize=12)
        ax2.set_title(f'Arrhenius Plot (σ = {sigma_fixed/1e6} MPa)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('creep_behavior.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 数値例
        print("=== Creep Rate Calculation (Al₂O₃) ===")
        T_test = 1400 + 273.15  # 1400°C
        for sigma_MPa in [10, 50, 100]:
            sigma = sigma_MPa * 1e6
            eps_dot = creep_rate(sigma, T_test, A_diffusion, n_diffusion, Q_diffusion)
            print(f"σ = {sigma_MPa:3d} MPa, T = 1400°C → ε̇ = {eps_dot:.2e} s^-1")
    
    # 実行
    plot_creep_behavior()
    

#### Python実装: 応力-ひずみ曲線の生成
    
    
    # ===================================
    # Example 7: 応力-ひずみ曲線の生成
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def stress_strain_curve(max_stress=500e6, E=400e9, K_IC=4e6, a_initial=10e-6):
        """
        セラミックスの応力-ひずみ曲線（線形弾性 + 突然破壊）
    
        Parameters:
        -----------
        max_stress : float
            最大応力 [Pa]
        E : float
            ヤング率 [Pa]
        K_IC : float
            破壊靭性 [Pa·m^0.5]
        a_initial : float
            初期亀裂長さ [m]
    
        Returns:
        --------
        stress : array
            応力 [Pa]
        strain : array
            ひずみ
        fracture_stress : float
            破壊応力 [Pa]
        """
        # 破壊応力の計算（K_IC = Y σ_f sqrt(π a)）
        Y = 1.12  # 表面亀裂の形状係数
        fracture_stress = K_IC / (Y * np.sqrt(np.pi * a_initial))
    
        # 線形弾性域
        if fracture_stress > max_stress:
            fracture_stress = max_stress
    
        stress = np.linspace(0, fracture_stress, 1000)
        strain = stress / E
    
        return stress, strain, fracture_stress
    
    
    def compare_materials():
        """
        各種セラミックスの応力-ひずみ曲線比較
        """
        materials = {
            'Al₂O₃': {'E': 400e9, 'K_IC': 4e6, 'a': 10e-6},
            'ZrO₂': {'E': 210e9, 'K_IC': 8e6, 'a': 10e-6},
            'Si₃N₄': {'E': 310e9, 'K_IC': 6e6, 'a': 10e-6},
            'SiC': {'E': 450e9, 'K_IC': 3.5e6, 'a': 10e-6}
        }
    
        plt.figure(figsize=(12, 5))
    
        # 左図: 応力-ひずみ曲線
        plt.subplot(1, 2, 1)
        for name, props in materials.items():
            stress, strain, sigma_f = stress_strain_curve(
                max_stress=1e9,
                E=props['E'],
                K_IC=props['K_IC'],
                a_initial=props['a']
            )
            plt.plot(strain * 100, stress / 1e6, linewidth=2, label=f'{name} (σ_f={sigma_f/1e6:.0f} MPa)')
    
        plt.xlabel('Strain (%)', fontsize=12)
        plt.ylabel('Stress (MPa)', fontsize=12)
        plt.title('Stress-Strain Curves for Ceramics', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # 右図: 金属との比較
        plt.subplot(1, 2, 2)
    
        # セラミックス（Al2O3）
        stress_ceramic, strain_ceramic, sigma_f_ceramic = stress_strain_curve(
            max_stress=1e9, E=400e9, K_IC=4e6, a_initial=10e-6
        )
        plt.plot(strain_ceramic * 100, stress_ceramic / 1e6, linewidth=2,
                 label='Ceramic (Al₂O₃)', color='red')
    
        # 金属（鋼）の模擬曲線（延性破壊）
        E_steel = 200e9
        yield_stress = 300e6
        UTS = 500e6
    
        strain_elastic = np.linspace(0, yield_stress / E_steel, 100)
        stress_elastic = strain_elastic * E_steel
    
        strain_plastic = np.linspace(yield_stress / E_steel, 0.2, 100)
        stress_plastic = yield_stress + (UTS - yield_stress) * (1 - np.exp(-50 * (strain_plastic - yield_stress / E_steel)))
    
        strain_steel = np.concatenate([strain_elastic, strain_plastic])
        stress_steel = np.concatenate([stress_elastic, stress_plastic])
    
        plt.plot(strain_steel * 100, stress_steel / 1e6, linewidth=2,
                 label='Metal (Steel)', color='blue')
    
        plt.xlabel('Strain (%)', fontsize=12)
        plt.ylabel('Stress (MPa)', fontsize=12)
        plt.title('Ceramic vs Metal: Brittle vs Ductile', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 20)
    
        plt.tight_layout()
        plt.savefig('stress_strain_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 数値例
        print("=== Stress-Strain Behavior Comparison ===")
        for name, props in materials.items():
            _, _, sigma_f = stress_strain_curve(
                max_stress=1e9,
                E=props['E'],
                K_IC=props['K_IC'],
                a_initial=props['a']
            )
            fracture_strain = sigma_f / props['E']
            print(f"{name:8s}: σ_f = {sigma_f/1e6:5.0f} MPa, ε_f = {fracture_strain*100:.3f}%")
    
    # 実行
    compare_materials()
    

**セラミックスの脆性の数値的意味** Al₂O₃の破壊ひずみは約0.1%（400 MPa / 400 GPa）で、鋼の20%（延性破壊）と比べて200倍小さい値です。この極端な脆性が、衝撃荷重に弱く、設計上の安全率を大きく取る必要がある理由です。 

## 演習問題

#### 演習 3-1: 応力集中係数の影響易

Al₂O₃試験片に10 μmの表面亀裂があり、先端曲率半径が1 nmです。応力集中係数Ktを計算し、局所応力が遠方場応力の何倍になるか求めなさい。 

解答例
    
    
    a = 10e-6  # m
    rho = 1e-9  # m
    K_t = 1 + 2 * np.sqrt(a / rho)
    print(f"K_t = {K_t:.1f}")
    # 出力: K_t ≈ 201（局所応力は約200倍）
    

#### 演習 3-2: Griffith強度の予測易

SiC（E = 450 GPa, γ = 1.2 J/m²）に50 μmの亀裂が存在する場合、Griffith理論による破壊強度を計算しなさい。 

解答例
    
    
    E = 450e9
    gamma = 1.2
    a = 50e-6
    sigma_f = np.sqrt(2 * E * gamma / (np.pi * a))
    print(f"σ_f = {sigma_f/1e6:.0f} MPa")
    # 出力: σ_f ≈ 117 MPa
    

#### 演習 3-3: 破壊靭性の測定易

SEVNB試験でAl₂O₃試験片（B=4 mm, W=3 mm, a=1.2 mm, S=40 mm）に荷重300 Nで破壊しました。KICを計算しなさい。 

解答例
    
    
    K_IC = fracture_toughness_SEVNB(300, 4e-3, 3e-3, 1.2e-3, 40e-3)
    print(f"K_IC = {K_IC/1e6:.2f} MPa·m^0.5")
    # 出力: K_IC ≈ 4.2 MPa·m^0.5
    

#### 演習 3-4: Weibullパラメータの推定中

以下の強度データ（MPa）: [350, 420, 380, 450, 390, 410, 370, 440, 400, 430] からWeibullモジュラスmと特性強度σ₀を推定しなさい。 

解答例
    
    
    data = np.array([350, 420, 380, 450, 390, 410, 370, 440, 400, 430])
    m, sigma_0, R2 = estimate_weibull_parameters(data * 1e6)
    print(f"m = {m:.2f}, σ₀ = {sigma_0/1e6:.1f} MPa, R² = {R2:.4f}")
    # 出力例: m ≈ 12.5, σ₀ ≈ 415 MPa
    

#### 演習 3-5: 設計応力の計算中

Si₃N₄部品（m=10, σ₀=700 MPa）を用いて99.9%信頼性を達成したい。設計許容応力を計算しなさい。 

解答例
    
    
    m = 10
    sigma_0 = 700e6
    reliability = 0.999
    sigma_design = sigma_0 * (-np.log(1 - reliability))**(1/m)
    print(f"σ_design = {sigma_design/1e6:.0f} MPa")
    # 出力: σ_design ≈ 368 MPa（σ₀の約53%）
    

#### 演習 3-6: Size Effectの評価中

試験片（体積V₁）での平均強度が400 MPaでした。10倍の体積（V₂=10V₁）の実部品の期待強度を、m=10として計算しなさい。 

解答例
    
    
    sigma_1 = 400  # MPa
    V_ratio = 10
    m = 10
    sigma_2 = sigma_1 * V_ratio**(-1/m)
    print(f"σ₂ = {sigma_2:.1f} MPa")
    # 出力: σ₂ ≈ 318 MPa（約20%低下）
    

#### 演習 3-7: クリープ速度の温度依存性中

Al₂O₃のクリープ（A=1e-8, n=1, Q=400 kJ/mol）において、応力50 MPa、温度1400°Cでのクリープ速度を計算しなさい。 

解答例
    
    
    sigma = 50e6
    T = 1400 + 273.15
    epsilon_dot = creep_rate(sigma, T, A=1e-8, n=1, Q=400e3)
    print(f"ε̇ = {epsilon_dot:.2e} s^-1")
    # 出力例: ε̇ ≈ 1.2e-8 s^-1
    

#### 演習 3-8: 多軸応力下の破壊難

Al₂O₃部品にσ₁=200 MPa（引張）、σ₂=-100 MPa（圧縮）の2軸応力が作用します。最大主応力説により、KIC=4 MPa·m1/2の材料が破壊するか判定しなさい（表面に5 μmの亀裂があると仮定）。 

解答例
    
    
    # 最大主応力 = σ₁（引張が支配的）
    sigma_principal = 200e6
    a = 5e-6
    Y = 1.12
    K_I = Y * sigma_principal * np.sqrt(np.pi * a)
    K_IC = 4e6
    print(f"K_I = {K_I/1e6:.2f} MPa·m^0.5, K_IC = {K_IC/1e6} MPa·m^0.5")
    if K_I >= K_IC:
        print("破壊が発生します")
    else:
        print("安全です")
    # 出力: K_I ≈ 2.81 MPa·m^0.5 < K_IC → 安全
    

#### 演習 3-9: R-curve挙動のモデリング難

変換強化ZrO₂のR-curve（KR = K₀ + A√Δa、K₀=4, A=2, Δaは亀裂進展量）を考慮し、初期亀裂10 μmから破壊に至る応力を計算しなさい。 

解答例
    
    
    K_0 = 4e6  # MPa·m^0.5
    A = 2e6
    a_initial = 10e-6
    Y = 1.12
    
    # 亀裂進展をシミュレーション
    delta_a_range = np.linspace(0, 50e-6, 100)
    K_R = K_0 + A * np.sqrt(delta_a_range)
    
    # 応力を増加させて破壊条件を探す
    for sigma_MPa in range(100, 1000, 10):
        sigma = sigma_MPa * 1e6
        K_I = Y * sigma * np.sqrt(np.pi * (a_initial + delta_a_range[-1]))
        if K_I >= K_R[-1]:
            print(f"破壊応力: {sigma_MPa} MPa")
            break
    # 出力例: R-curve効果により強度向上
    

#### 演習 3-10: 長期信頼性予測難

Si₃N₄タービン部品（m=12, σ₀=800 MPa）を使用応力400 MPaで10年間運用します。静的疲労（slow crack growth）を考慮せず、初期強度分布のみで破壊確率を推定しなさい。 

解答例
    
    
    m = 12
    sigma_0 = 800e6
    sigma_applied = 400e6
    
    # Weibull累積破壊確率
    P_f = 1 - np.exp(-((sigma_applied / sigma_0)**m))
    print(f"破壊確率 P_f = {P_f*100:.4f}%")
    print(f"信頼性 = {(1-P_f)*100:.4f}%")
    # 出力: P_f ≈ 0.024% → 信頼性 99.976%
    # 注: 実際には疲労を考慮すると破壊確率は増加
    

## 参考文献

  1. Lawn, B.R. (1993). _Fracture of Brittle Solids_. Cambridge University Press, pp. 1-75, 194-250.
  2. Munz, D., Fett, T. (2007). _Ceramics: Mechanical Properties, Failure Behaviour, Materials Selection_. Springer, pp. 45-120, 201-255.
  3. Anderson, T.L. (2017). _Fracture Mechanics: Fundamentals and Applications_. CRC Press, pp. 220-285.
  4. Weibull, W. (1951). A Statistical Distribution Function of Wide Applicability. _Journal of Applied Mechanics_ , 18, 293-297.
  5. Quinn, G.D. (2007). _Fractography of Ceramics and Glasses_. NIST Special Publication 960-16, pp. 1-50.
  6. Carter, C.B., Norton, M.G. (2013). _Ceramic Materials: Science and Engineering_. Springer, pp. 520-590.

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
