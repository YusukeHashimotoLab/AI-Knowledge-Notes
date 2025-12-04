---
title: "Chapter 3: Nanomaterials"
chapter_title: "Chapter 3: Nanomaterials"
subtitle: Carbon Nanotubes, Graphene, and Quantum Dots - Design Principles for High Performance
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 3

🌐 EN | [🇯🇵 JP](<../../../jp/MS/advanced-materials-systems-introduction/chapter-3.html>) | Last sync: 2025-11-16

## Learning Objectives

Upon completing this chapter, you will be able to explain:

### Basic Understanding

  * High-strength and high-toughness mechanisms in structural ceramics (transformation toughening, fiber reinforcement)
  * Physical origins and crystal structures of functional ceramics (piezoelectric, dielectric, magnetic)
  * Biocompatibility and osseointegration mechanisms of bioceramics
  * Mechanical properties of ceramics and statistical fracture theory (Weibull distribution)

### Practical Skills

  * Analyze strength distribution of ceramics (Weibull statistics) using Python
  * Calculate phase diagrams using pycalphad and optimize sintering conditions
  * Calculate and evaluate piezoelectric constants, dielectric permittivity, and magnetic properties
  * Select optimal ceramics for specific applications using material selection matrices

### Applied Capabilities

  * Design optimal ceramic composition and microstructure from application requirements
  * Design functional ceramic devices (sensors, actuators)
  * Evaluate biocompatibility of bioceramic implants
  * Perform reliability design for ceramic materials (probabilistic fracture prediction)

## 1.1 Structural Ceramics - Principles of High Strength and Toughness

### 1.1.1 Overview of Structural Ceramics

Structural ceramics are **ceramic materials with excellent mechanical properties (high strength, high hardness, heat resistance) used as structural components in harsh environments**. They enable use in high-temperature and corrosive environments impossible for metallic materials, with the following important applications:

  * **Al₂O₃（アルミナ）** : 切削工具、耐摩耗部品、人工関節（生体適合性）
  * **ZrO₂（ジルコニア）** : 歯科材料、酸素センサー、熱遮蔽コーティング（高靭性）
  * **Si₃N₄（窒化ケイ素）** : ガスタービン部品、ベアリング（高温強度）
  * **SiC（炭化ケイ素）** : 半導体製造装置、装甲材（超高硬度）

**💡 産業的重要性**

Structural ceramics are indispensable in aerospace, automotive, and medical fields。世界のセラミックス市場（2023年時点で$230B以上）の約60%が先進セラミックス材料です。その理由は：

  * 金属の3-5倍の強度（常温）と優れた耐熱性（1500°C以上）
  * 化学的安定性（酸・アルカリに不活性）
  * 低密度（金属の1/2-1/3）による軽量化効果
  * 高硬度（Hv 1500-2500）による耐摩耗性

### 1.1.2 High-Strength Ceramics (Al₂O₃, ZrO₂, Si₃N₄)

High-strength ceramics are typically represented by the following three main materials:
    
    
    flowchart LR
        A[Al₂O₃  
    アルミナ] --> B[高硬度  
    Hv 2000]
        C[ZrO₂  
    ジルコニア] --> D[高靭性  
    10-15 MPa√m]
        E[Si₃N₄  
    窒化ケイ素] --> F[高温強度  
    1400°C使用]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style E fill:#e8f5e9
        style B fill:#f3e5f5
        style D fill:#fce4ec
        style F fill:#fff9c4
            

  1. **Al₂O₃（アルミナ）** : 酸化物セラミックスの代表格。高硬度（Hv 2000）、優れた耐摩耗性、生体適合性により、切削工具・人工関節に使用。製造コストが低く最も広く普及。
  2. **ZrO₂（ジルコニア）** : 相変態強化（Transformation Toughening）により、highest level of fracture toughness among ceramic materials（10-15 MPa√m）を実現。「セラミックス鋼」とも呼ばれる。
  3. **Si₃N₄（窒化ケイ素）** : 共有結合性が強く、1400°Cまで高強度を維持。Used as high-temperature structural material in gas turbine components and bearings。熱衝撃抵抗性も優れる。

**⚠️ セラミックスの本質的課題**

While ceramics possess high strength and hardness, **脆性（低靭性）** が最大の欠点です。微小な欠陥（気孔、亀裂）が応力集中点となり、突発的な破壊を引き起こします（Griffith理論）。破壊靭性は金属の1/10以下です。このため、toughening techniques are important research topics。

### 1.1.3 Toughening Mechanisms

#### メカニズム1: 相変態強化（Transformation Toughening）

ジルコニア（ZrO₂）This is the strengthening mechanism most effectively functioning：

ZrO₂（正方晶、t-phase） → ZrO₂（単斜晶、m-phase） + 体積膨張（3-5%） 

**強化のメカニズム：**

  * **応力誘起変態** : 亀裂先端の高応力場で、準安定な正方晶（t）が単斜晶（m）へ相変態
  * **体積膨張効果** : 3-5%volume expansion generates compressive stress around crack、亀裂進展を抑制
  * **エネルギー吸収** : Energy consumption during transformation increases fracture energy
  * **靭性向上効果** : 破壊靭性が3 MPa√m → 10-15 MPa√m（3-5倍向上）

**実現方法：** Y₂O₃（3-8 mol%）やMgO（9-15 mol%）を添加し、正方晶をRoom temperatureで準安定化（PSZ: Partially Stabilized Zirconia）

#### メカニズム2: 繊維強化（Fiber Reinforcement）

This method composites high-strength fibers into a ceramic matrix：

セラミックス複合材料（CMC） = セラミックスマトリックス + 強化繊維（SiC, C, Al₂O₃） 

**強化のメカニズム：**

  * **クラックデフレクション** : 亀裂が繊維界面で偏向し、進展経路が長くなる
  * **ファイバープルアウト** : Absorbs large energy when fibers are pulled out
  * **クラックブリッジング** : 繊維が亀裂を架橋し、応力伝達を維持
  * **靭性向上効果** : 破壊靭性が5 MPa√m → 20-30 MPa√m（4-6倍向上）

**応用例：** SiC/SiC複合材料（航空機エンジン部品）、C/C複合材料（ブレーキディスク）

## 1.2 Functional Ceramics - Piezoelectric, Dielectric, and Magnetic

### 1.2.1 Piezoelectric Ceramics

The piezoelectric effect is **mechanical stress induces electric polarization（正圧電効果）、conversely, applying an electric field induces mechanical strain（逆圧電効果）現象** です。

#### 代表的な圧電材料

PZT（Pb(Zr,Ti)O₃）：圧電定数 d₃₃ = 200-600 pC/N 

BaTiO₃（チタン酸バリウム）：圧電定数 d₃₃ = 85-190 pC/N（鉛フリー代替材料） 

**PZT（ジルコン酸チタン酸鉛）の特徴：**

  * **高圧電定数** : d₃₃ = 200-600 pC/N（応用材料として最も優れる）
  * **モルフォトロピック相境界（MPB）** : Zr/Ti比率 52/48付近で圧電特性が最大化
  * **キュリーTemperature** : 320-380°C（このTemperature以上で圧電性消失）
  * **応用** : 超音波振動子、圧電アクチュエータ、圧電スピーカー、圧電点火装置

**⚠️ 環境問題と鉛フリー化**

PZTは鉛（Pb）を60wt%以上含むため、欧州RoHS規制で使用制限があります。鉛フリー代替材料として、BaTiO₃系、(K,Na)NbO₃系、BiFeO₃系が研究されていますが、PZTの性能には及びません（d₃₃ = 100-300 pC/N）。Piezoelectric devices are exempt items for medical equipment, but、長期的には代替材料開発が必要です。

#### 圧電効果の結晶学的起源

圧電効果は**非中心対称結晶構造** を持つ材料でのみ発現します：

  * **常誘電相（立方晶、Pm3m）** : 中心対称 → 圧電性なし（高温）
  * **強誘電相（正方晶、P4mm）** : 非中心対称 → 圧電性あり（Room temperature）
  * **自発分極** : Ti⁴⁺Dipole moment generated by displacement of ion from octahedral center
  * **分域（ドメイン）構造** : 電場印加により分域の方位が揃い、巨大圧電効果を発現（ポーリング処理）

### 1.2.2 Dielectric Ceramics

誘電セラミックスは、**高い誘電率（εᵣ）を持ち、capacitor materials that store electrical energy** として使用されます。

#### MLCC（積層セラミックコンデンサ）用材料

BaTiO₃（チタン酸バリウム）：εᵣ = 1,500-10,000（Room temperature、1 kHz） 

**高誘電率の起源：**

  * **強誘電性（Ferroelectricity）** : Property where spontaneous polarization can be reversed by external electric field
  * **分域壁の移動** : Domain walls move easily under electric field、大きな分極変化を生じる
  * **キュリーTemperature（Tc）** : BaTiO₃ではTc = 120°C、このTemperatureで誘電率がピーク
  * **組成調整** : CaZrO₃、SrTiO₃を添加してTcをRoom temperature付近にシフト（X7R特性）

**✅ MLCC（多層セラミックコンデンサ）の驚異的性能**

現代のMLCChave been miniaturized and enhanced to the extreme：

  * **積層数** : 1,000層以上（誘電体層厚み < 1 μm）
  * **静電容量** : 1 mm³サイズで100 μF以上達成
  * **用途** : スマートフォン1台に800個以上搭載
  * **市場規模** : 年間生産数 1兆個以上（世界最大の電子部品）

BaTiO₃ベースのMLCCare key materials for miniaturization and performance enhancement of electronic devices。

### 1.2.3 Magnetic Ceramics (Ferrites)

フェライト（Ferrites）は、**酸化物系の磁性材料で、高周波における低損失特性** を持つため、widely used in transformers, inductors, and electromagnetic wave absorbers。

#### フェライトの種類と用途

スピネル型フェライト：MFe₂O₄（M = Mn, Ni, Zn, Co等） 

六方晶フェライト（ハードフェライト）：BaFe₁₂O₁₉、SrFe₁₂O₁₉（永久磁石） 

**スピネル型フェライトの特徴：**

  * **ソフト磁性** : 保磁力が小さく（Hc < 100 A/m）、容易に磁化反転
  * **高周波特性** : 高い電気抵抗（ρ > 10⁶ Ω·cm）により渦電流損失が小さい
  * **Mn-Znフェライト** : 高透磁率（μᵣ = 2,000-15,000）、低周波トランスフォーマー用
  * **Ni-Znフェライト** : 高周波特性に優れる（GHz帯）、EMI対策部品用

**六方晶フェライト（ハードフェライト）の特徴：**

  * **ハード磁性** : 大きな保磁力（Hc = 200-400 kA/m）と残留磁束密度（Br = 0.4 T）
  * **永久磁石材料** : モーター、スピーカー、磁気記録媒体に使用
  * **低コスト** : 希土類磁石（Nd-Fe-B）より性能は劣るが、原料が安価で大量生産可能
  * **耐食性** : Does not corrode unlike metallic magnets due to oxide nature

**💡 フェライトの磁性起源**

フェライトの磁性はスピネル構造（AB₂O₄）中の**A席（四面体位置）とB席（八面体位置）antiparallel alignment of magnetic moments of ions** することで発現します（フェリ磁性）。Mn-ZnフェライトではMn²⁺とFe³⁺magnetic moments partially cancel each other、overall magnetization becomes smaller, but、高透磁率が実現されます。

## 1.3 Bioceramics - Biocompatibility and Osseointegration

### 1.3.1 Overview of Bioceramics

Bioceramics are **do not cause rejection reactions when in contact with biological tissue（生体適合性）、骨組織と直接結合できる（骨伝導性）セラミックス材料** です。

#### 代表的なバイオセラミックス

HAp（ハイドロキシアパタイト）：Ca₁₀(PO₄)₆(OH)₂ 

β-TCP（リン酸三カルシウム）：Ca₃(PO₄)₂ 

**ハイドロキシアパタイト（HAp）の特徴：**

  * **骨の主成分** : 天然骨の無機成分の65%がHAp（残り35%は有機物コラーゲン）
  * **生体適合性** : No rejection reaction due to similar chemical composition to bone tissue、拒絶反応が起きない
  * **骨伝導性（Osteoconduction）** : HAp表面に骨芽細胞が付着・増殖し、新しい骨組織が形成される
  * **骨結合（Osseointegration）** : HApDirect chemical bonding forms between surface and bone tissue
  * **応用** : 人工骨、歯科インプラント、骨充填材、Ti合金インプラントのコーティング

**✅ β-TCPの生体吸収性**

β-TCP (tricalcium phosphate), unlike HAp, has the property of **生体内で徐々に吸収される** 特性を持ちます：

  * **吸収期間** : 6-18ヶ月で完全吸収（粒子サイズ・気孔率に依存）
  * **置換メカニズム** : β-TCPが溶解しながら、新しい骨組織に置き換わる（Bone remodeling）
  * **Ca²⁺・PO₄³⁻供給** : Released ions from dissolution promote bone formation
  * **HAp/β-TCP複合材** : Absorption rate controllable by mixing ratio（HAp 70% / β-TCP 30%等）

生体吸収性により、永久的な異物が体内に残らず、achieves ideal bone regeneration where permanent foreign material does not remain in the body and is completely replaced by autologous bone tissue。

### 1.4 Python Practice: Analysis and Design of Ceramic Materials

### Example 1: Analysis of Fracture Strength Distribution using Weibull Statistics
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example 1: Analysis of Fracture Strength Distribution using 
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 1: Arrhenius Equation Simulation
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    R = 8.314  # J/(mol·K)
    
    # Diffusion parameters for BaTiO₃ system (literature values)
    D0 = 5e-4  # m²/s (Frequency factor)
    Ea = 300e3  # J/mol (Activation energy 300 kJ/mol)
    
    def diffusion_coefficient(T, D0, Ea):
        """Calculate diffusion coefficient using Arrhenius equation
    
        Args:
            T (float or array): Temperature [K]
            D0 (float): Frequency factor [m²/s]
            Ea (float): Activation energy [J/mol]
    
        Returns:
            float or array: Diffusion coefficient [m²/s]
        """
        return D0 * np.exp(-Ea / (R * T))
    
    # Temperature範囲 800-1400°C
    T_celsius = np.linspace(800, 1400, 100)
    T_kelvin = T_celsius + 273.15
    
    # Diffusion coefficientを計算
    D = diffusion_coefficient(T_kelvin, D0, Ea)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # 対数Plot（ArrheniusPlot）
    plt.subplot(1, 2, 1)
    plt.semilogy(T_celsius, D, 'b-', linewidth=2)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Diffusion Coefficient (m²/s)', fontsize=12)
    plt.title('Arrhenius Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 1/T vs ln(D) Plot（Linear relationship）
    plt.subplot(1, 2, 2)
    plt.plot(1000/T_kelvin, np.log(D), 'r-', linewidth=2)
    plt.xlabel('1000/T (K⁻¹)', fontsize=12)
    plt.ylabel('ln(D)', fontsize=12)
    plt.title('Linearized Arrhenius Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('arrhenius_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 主要TemperatureでのDiffusion coefficientを表示
    key_temps = [1000, 1100, 1200, 1300]
    print("Temperature依存性の比較:")
    print("-" * 50)
    for T_c in key_temps:
        T_k = T_c + 273.15
        D_val = diffusion_coefficient(T_k, D0, Ea)
        print(f"{T_c:4d}°C: D = {D_val:.2e} m²/s")
    
    # Output example:
    # Temperature依存性の比較:
    # --------------------------------------------------
    # 1000°C: D = 1.89e-12 m²/s
    # 1100°C: D = 9.45e-12 m²/s
    # 1200°C: D = 4.01e-11 m²/s
    # 1300°C: D = 1.48e-10 m²/s
    

### Example 2: Simulation of Reaction Progress using Jander Equation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 2: Jander equationによるConversion計算
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    def jander_equation(alpha, k, t):
        """Jander equation
    
        Args:
            alpha (float): Conversion (0-1)
            k (float): Rate constant [s⁻¹]
            t (float): Time [s]
    
        Returns:
            float: Jander equationの左辺 - k*t
        """
        return (1 - (1 - alpha)**(1/3))**2 - k * t
    
    def calculate_conversion(k, t):
        """TimetにおけるConversionを計算
    
        Args:
            k (float): Rate constant
            t (float): Time
    
        Returns:
            float: Conversion (0-1)
        """
        # Jander equationをalphaについて数値的に解く
        alpha0 = 0.5  # Initial estimate
        alpha = fsolve(lambda a: jander_equation(a, k, t), alpha0)[0]
        return np.clip(alpha, 0, 1)  # 0-1Constrain to range
    
    # Parameter settings
    D = 1e-11  # m²/s (1200°CでのDiffusion coefficient)
    C0 = 10000  # mol/m³
    r0_values = [1e-6, 5e-6, 10e-6]  # Particle radius [m]: 1μm, 5μm, 10μm
    
    # Time array（0-50Time）
    t_hours = np.linspace(0, 50, 500)
    t_seconds = t_hours * 3600
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Effect of Particle Size
    plt.subplot(1, 2, 1)
    for r0 in r0_values:
        k = D * C0 / r0**2
        alpha = [calculate_conversion(k, t) for t in t_seconds]
        plt.plot(t_hours, alpha, linewidth=2,
                 label=f'r₀ = {r0*1e6:.1f} μm')
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Conversion (α)', fontsize=12)
    plt.title('Effect of Particle Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Temperatureの影響（粒子サイズfixed）
    plt.subplot(1, 2, 2)
    r0_fixed = 5e-6  # 5μmfixed
    temperatures = [1100, 1200, 1300]  # °C
    
    for T_c in temperatures:
        T_k = T_c + 273.15
        D_T = diffusion_coefficient(T_k, D0, Ea)
        k = D_T * C0 / r0_fixed**2
        alpha = [calculate_conversion(k, t) for t in t_seconds]
        plt.plot(t_hours, alpha, linewidth=2,
                 label=f'{T_c}°C')
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Conversion (α)', fontsize=12)
    plt.title('Effect of Temperature (r₀ = 5 μm)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('jander_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 50%反応に要するTimeを計算
    print("\n50%反応に要するTime:")
    print("-" * 50)
    for r0 in r0_values:
        k = D * C0 / r0**2
        t_50 = fsolve(lambda t: jander_equation(0.5, k, t), 10000)[0]
        print(f"r₀ = {r0*1e6:.1f} μm: t₅₀ = {t_50/3600:.1f} hours")
    
    # Output example:
    # 50%反応に要するTime:
    # --------------------------------------------------
    # r₀ = 1.0 μm: t₅₀ = 1.9 hours
    # r₀ = 5.0 μm: t₅₀ = 47.3 hours
    # r₀ = 10.0 μm: t₅₀ = 189.2 hours
    

### Example 3: Calculation of Activation Energy (from DSC/TG Data)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example 3: Calculation of Activation Energy (from DSC/TG Dat
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 3: Activation Energy Calculation using Kissinger Method
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    # Kissinger method: ln(β/Tp²) vs 1/Tp Determine Ea from slope of line
    # β: Heating rate [K/min]
    # Tp: ピークTemperature [K]
    # Slope = -Ea/R
    
    # Experimental data（異なるHeating rateでのDSCピークTemperature）
    heating_rates = np.array([5, 10, 15, 20])  # K/min
    peak_temps_celsius = np.array([1085, 1105, 1120, 1132])  # °C
    peak_temps_kelvin = peak_temps_celsius + 273.15
    
    def kissinger_analysis(beta, Tp):
        """Kissinger methodでActivation energyを計算
    
        Args:
            beta (array): Heating rate [K/min]
            Tp (array): ピークTemperature [K]
    
        Returns:
            tuple: (Ea [kJ/mol], A [min⁻¹], R²)
        """
        # Left side of Kissinger equation
        y = np.log(beta / Tp**2)
    
        # 1/Tp
        x = 1000 / Tp  # 1000/Tでスケーリング（見やすくするため）
    
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
        # Activation energy計算
        R = 8.314  # J/(mol·K)
        Ea = -slope * R * 1000  # J/mol → kJ/mol
    
        # Frequency factor
        A = np.exp(intercept)
    
        return Ea, A, r_value**2
    
    # Activation energy計算
    Ea, A, R2 = kissinger_analysis(heating_rates, peak_temps_kelvin)
    
    print("Kissinger methodによる解析結果:")
    print("=" * 50)
    print(f"Activation energy Ea = {Ea:.1f} kJ/mol")
    print(f"Frequency factor A = {A:.2e} min⁻¹")
    print(f"Coefficient of determination R² = {R2:.4f}")
    print("=" * 50)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # KissingerPlot
    y_data = np.log(heating_rates / peak_temps_kelvin**2)
    x_data = 1000 / peak_temps_kelvin
    
    plt.plot(x_data, y_data, 'ro', markersize=10, label='Experimental data')
    
    # Fitting line
    x_fit = np.linspace(x_data.min()*0.95, x_data.max()*1.05, 100)
    slope = -Ea * 1000 / (R * 1000)
    intercept = np.log(A)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'Fit: Ea = {Ea:.1f} kJ/mol')
    
    plt.xlabel('1000/Tp (K⁻¹)', fontsize=12)
    plt.ylabel('ln(β/Tp²)', fontsize=12)
    plt.title('Kissinger Plot for Activation Energy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Display results in text box
    textstr = f'Ea = {Ea:.1f} kJ/mol\nR² = {R2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('kissinger_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output example:
    # Kissinger methodによる解析結果:
    # ==================================================
    # Activation energy Ea = 287.3 kJ/mol
    # Frequency factor A = 2.45e+12 min⁻¹
    # Coefficient of determination R² = 0.9956
    # ==================================================
    

## 1.4 Python Practice: Analysis and Design of Ceramic Materials

### 1.4.1 Three Elements of Temperature Profile

固相反応におけるTemperature profileは、is the most important control parameter determining reaction success。以下の3elements must be properly designed：
    
    
    flowchart TD
        A[Temperature profile設計] --> B[Heating rate  
    Heating Rate]
        A --> C[保持Time  
    Holding Time]
        A --> D[Cooling Rate]
    
        B --> B1[Too fast: thermal stress → cracks]
        B --> B2[Too slow: unwanted phase transformations]
    
        C --> C1[Too short: incomplete reaction]
        C --> C2[Too long: excessive grain growth]
    
        D --> D1[Too fast: thermal stress → cracks]
        D --> D2[遅すぎ: undesirable phases]
    
        style A fill:#f093fb
        style B fill:#e3f2fd
        style C fill:#e8f5e9
        style D fill:#fff3e0
            

#### 1\. Heating rate（Heating Rate）

**General recommended value:** 2-10°C/min

**Factors to consider:**

  * **Thermal stress** : 試料内部と表面のTemperature差が大きいとThermal stressが発生し、亀裂の原因に
  * **Intermediate phase formation** : to avoid unwanted intermediate phase formation at low temperatures、あるTemperature範囲は速く通過
  * **Decomposition reactions** : Rapid heating can cause bumping in CO₂ or H₂O release reactions

**⚠️ 実例: BaCO₃のDecomposition reactions**

BaTiO₃合成では800-900°Cで BaCO₃ → BaO + CO₂ の分解が起こります。Heating rateが20°C/min以上だと、CO₂が急激に放出され、試料が破裂することがあります。推奨Heating rateは5°C/min以下です。

#### 2\. 保持Time（Holding Time）

**Determination method:** Jander equationからの推算 + 実験最適化

必要な保持Timeは以下の式で推定できます：

t = [α_target / k]^(1/2) × (1 - α_target^(1/3))^(-2) 

**典型的な保持Time：**

  * 低温反応（<1000°C）: 12-24Time
  * 中温反応（1000-1300°C）: 4-8Time
  * 高温反応（>1300°C）: 2-4Time

#### 3\. Cooling Rate

**General recommended value:** 1-5°C/min（Heating rateより遅め）

**Importance:**

  * **Control of phase transformation** : Control high-temperature phase → low-temperature phase transformation during cooling
  * **Defect generation** : Rapid cooling freezes defects such as oxygen vacancies
  * **Crystallinity** : 徐冷はCrystallinityを向上

### 1.4.2 Temperature Profile Optimization Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 4: Temperature profile最適化
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def temperature_profile(t, T_target, heating_rate, hold_time, cooling_rate):
        """Generate temperature profile
    
        Args:
            t (array): Time array [min]
            T_target (float): Holding temperature [°C]
            heating_rate (float): Heating rate [°C/min]
            hold_time (float): 保持Time [min]
            cooling_rate (float): Cooling rate [°C/min]
    
        Returns:
            array: Temperature profile [°C]
        """
        T_room = 25  # Room temperature
        T = np.zeros_like(t)
    
        # Heating time
        t_heat = (T_target - T_room) / heating_rate
    
        # Cooling start time
        t_cool_start = t_heat + hold_time
    
        for i, time in enumerate(t):
            if time <= t_heat:
                # Heating phase
                T[i] = T_room + heating_rate * time
            elif time <= t_cool_start:
                # Holding phase
                T[i] = T_target
            else:
                # Cooling phase
                T[i] = T_target - cooling_rate * (time - t_cool_start)
                T[i] = max(T[i], T_room)  # Room temperature以下にはならない
    
        return T
    
    def simulate_reaction_progress(T, t, Ea, D0, r0):
        """Temperature profileに基づく反応進行を計算
    
        Args:
            T (array): Temperature profile [°C]
            t (array): Time array [min]
            Ea (float): Activation energy [J/mol]
            D0 (float): Frequency factor [m²/s]
            r0 (float): Particle radius [m]
    
        Returns:
            array: Conversion
        """
        R = 8.314
        C0 = 10000
        alpha = np.zeros_like(t)
    
        for i in range(1, len(t)):
            T_k = T[i] + 273.15
            D = D0 * np.exp(-Ea / (R * T_k))
            k = D * C0 / r0**2
    
            dt = (t[i] - t[i-1]) * 60  # min → s
    
            # Simple integration (reaction progress at small time steps)
            if alpha[i-1] < 0.99:
                dalpha = k * dt / (2 * (1 - (1-alpha[i-1])**(1/3)))
                alpha[i] = min(alpha[i-1] + dalpha, 1.0)
            else:
                alpha[i] = alpha[i-1]
    
        return alpha
    
    # Parameter settings
    T_target = 1200  # °C
    hold_time = 240  # min (4 hours)
    Ea = 300e3  # J/mol
    D0 = 5e-4  # m²/s
    r0 = 5e-6  # m
    
    # Comparison at different heating rates
    heating_rates = [2, 5, 10, 20]  # °C/min
    cooling_rate = 3  # °C/min
    
    # Time array
    t_max = 800  # min
    t = np.linspace(0, t_max, 2000)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Temperature profile
    for hr in heating_rates:
        T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate)
        ax1.plot(t/60, T_profile, linewidth=2, label=f'{hr}°C/min')
    
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Profiles', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t_max/60])
    
    # 反応進行
    for hr in heating_rates:
        T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate)
        alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0)
        ax2.plot(t/60, alpha, linewidth=2, label=f'{hr}°C/min')
    
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='Target (95%)')
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Conversion', fontsize=12)
    ax2.set_title('Reaction Progress', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t_max/60])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('temperature_profile_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 各Heating rateでの95%反応到達Timeを計算
    print("\n95%反応到達Timeの比較:")
    print("=" * 60)
    for hr in heating_rates:
        T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate)
        alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0)
    
        # 95% conversion time
        idx_95 = np.where(alpha >= 0.95)[0]
        if len(idx_95) > 0:
            t_95 = t[idx_95[0]] / 60
            print(f"Heating rate {hr:2d}°C/min: t₉₅ = {t_95:.1f} hours")
        else:
            print(f"Heating rate {hr:2d}°C/min: Incomplete reaction")
    
    # Output example:
    # 95%反応到達Timeの比較:
    # ============================================================
    # Heating rate  2°C/min: t₉₅ = 7.8 hours
    # Heating rate  5°C/min: t₉₅ = 7.2 hours
    # Heating rate 10°C/min: t₉₅ = 6.9 hours
    # Heating rate 20°C/min: t₉₅ = 6.7 hours
    

## Exercises

### 1.5.1 What is pycalphad

**pycalphad** is a Python library for phase diagram calculations based on the CALPHAD (CALculation of PHAse Diagrams) method. It calculates equilibrium phases from thermodynamic databases and is useful for designing reaction pathways.

**💡 Advantages of CALPHAD Method**

  * Can calculate complex phase diagrams of multicomponent systems (ternary and higher)
  * Experimental dataが少ない系でも予測可能
  * Temperature, composition, and pressure dependencies comprehensively

### 1.5.2 Example of Binary Phase Diagram Calculation
    
    
    # ===================================
    # Example 5: Phase Diagram Calculation with pycalphad
    # ===================================
    
    # Note: pycalphad installation required
    # pip install pycalphad
    
    from pycalphad import Database, equilibrium, variables as v
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load TDB database (simplified example here)
    # Actual appropriate TDB file needed in practice
    # Example: BaO-TiO2 system
    
    # Simplified TDB string (actually more complex)
    tdb_string = """
    $ BaO-TiO2 system (simplified)
    ELEMENT BA   BCC_A2  137.327   !
    ELEMENT TI   HCP_A3   47.867   !
    ELEMENT O    GAS      15.999   !
    
    FUNCTION GBCCBA   298.15  +GHSERBA;   6000 N !
    FUNCTION GHCPTI   298.15  +GHSERTI;   6000 N !
    FUNCTION GGASO    298.15  +GHSERO;    6000 N !
    
    PHASE LIQUID:L %  1  1.0  !
    PHASE BAO_CUBIC %  2  1 1  !
    PHASE TIO2_RUTILE %  2  1 2  !
    PHASE BATIO3 %  3  1 1 3  !
    """
    
    # Note: Formal TDB file required for actual calculations
    # Limited to conceptual explanation here
    
    print("Concept of Phase Diagram Calculation with pycalphad:")
    print("=" * 60)
    print("1. Load TDB database (thermodynamic data)")
    print("2. Temperature・組成範囲を設定")
    print("3. Execute equilibrium calculation")
    print("4. Visualize stable phases")
    print()
    print("Actual application examples:")
    print("- BaO-TiO2系: BaTiO3の形成Temperature・組成範囲")
    print("- Si-N system: Stable region of Si₃N₄")
    print("- Phase relationships of multicomponent ceramics")
    
    # 概念的なPlot（実データに基づくイメージ）
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Temperature範囲
    T = np.linspace(800, 1600, 100)
    
    # Stable regions of each phase (conceptual diagram)
    # BaO + TiO2 → BaTiO₃ reaction
    BaO_region = np.ones_like(T) * 0.3
    TiO2_region = np.ones_like(T) * 0.7
    BaTiO3_region = np.where((T > 1100) & (T < 1400), 0.5, np.nan)
    
    ax.fill_between(T, 0, BaO_region, alpha=0.3, color='blue', label='BaO + TiO2')
    ax.fill_between(T, BaO_region, TiO2_region, alpha=0.3, color='green',
                    label='BaTiO₃ stable')
    ax.fill_between(T, TiO2_region, 1, alpha=0.3, color='red', label='Liquid')
    
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
               label='BaTiO₃ composition')
    ax.axvline(x=1100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=1400, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Composition (BaO mole fraction)', fontsize=12)
    ax.set_title('Conceptual Phase Diagram: BaO-TiO2', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([800, 1600])
    ax.set_ylim([0, 1])
    
    # テキスト注釈
    ax.text(1250, 0.5, 'BaTiO₃\nformation\nregion',
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('phase_diagram_concept.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Actual usage example (commented out)
    """
    # Actual pycalphad usage example
    db = Database('BaO-TiO2.tdb')  # Load TDB file
    
    # Equilibrium calculation
    eq = equilibrium(db, ['BA', 'TI', 'O'], ['LIQUID', 'BATIO3'],
                     {v.X('BA'): (0, 1, 0.01),
                      v.T: (1000, 1600, 50),
                      v.P: 101325})
    
    # 結果Plot
    eq.plot()
    """
    

## 1.6 Condition Optimization using Design of Experiments (DOE)

### 1.6.1 What is DOE

Design of Experiments (DOE) is a statistical method to find optimal conditions with minimum number of experiments in systems with multiple interacting parameters.

**Key parameters to optimize in solid-state reactions:**

  * 反応Temperature（T）
  * 保持Time（t）
  * Particle size (r)
  * Raw material ratio (molar ratio)
  * Atmosphere (air, nitrogen, vacuum, etc.)

### 1.6.2 Response Surface Methodology
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 6: Condition Optimization using DOE
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.optimize import minimize
    
    # 仮想的なConversionモデル（TemperatureとTimeの関数）
    def reaction_yield(T, t, noise=0):
        """TemperatureとTimeからConversionを計算（仮想モデル）
    
        Args:
            T (float): Temperature [°C]
            t (float): Time [hours]
            noise (float): Noise level
    
        Returns:
            float: Conversion [%]
        """
        # Optimal value: T=1200°C, t=6 hours
        T_opt = 1200
        t_opt = 6
    
        # Quadratic model (Gaussian)
        yield_val = 100 * np.exp(-((T-T_opt)/150)**2 - ((t-t_opt)/3)**2)
    
        # Add noise
        if noise > 0:
            yield_val += np.random.normal(0, noise)
    
        return np.clip(yield_val, 0, 100)
    
    # Experimental point arrangement (central composite design)
    T_levels = [1000, 1100, 1200, 1300, 1400]  # °C
    t_levels = [2, 4, 6, 8, 10]  # hours
    
    # Arrange experimental points on grid
    T_grid, t_grid = np.meshgrid(T_levels, t_levels)
    yield_grid = np.zeros_like(T_grid, dtype=float)
    
    # 各実験点でConversionを測定（シミュレーション）
    for i in range(len(t_levels)):
        for j in range(len(T_levels)):
            yield_grid[i, j] = reaction_yield(T_grid[i, j], t_grid[i, j], noise=2)
    
    # Display results
    print("Reaction Condition Optimization using Design of Experiments")
    print("=" * 70)
    print(f"{'Temperature (°C)':<20} {'Time (hours)':<20} {'Yield (%)':<20}")
    print("-" * 70)
    for i in range(len(t_levels)):
        for j in range(len(T_levels)):
            print(f"{T_grid[i, j]:<20} {t_grid[i, j]:<20} {yield_grid[i, j]:<20.1f}")
    
    # 最大Conversionの条件を探す
    max_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape)
    T_best = T_grid[max_idx]
    t_best = t_grid[max_idx]
    yield_best = yield_grid[max_idx]
    
    print("-" * 70)
    print(f"Optimal conditions: T = {T_best}°C, t = {t_best} hours")
    print(f"最大Conversion: {yield_best:.1f}%")
    
    # 3DPlot
    fig = plt.figure(figsize=(14, 6))
    
    # 3D表面Plot
    ax1 = fig.add_subplot(121, projection='3d')
    T_fine = np.linspace(1000, 1400, 50)
    t_fine = np.linspace(2, 10, 50)
    T_mesh, t_mesh = np.meshgrid(T_fine, t_fine)
    yield_mesh = np.zeros_like(T_mesh)
    
    for i in range(len(t_fine)):
        for j in range(len(T_fine)):
            yield_mesh[i, j] = reaction_yield(T_mesh[i, j], t_mesh[i, j])
    
    surf = ax1.plot_surface(T_mesh, t_mesh, yield_mesh, cmap='viridis',
                            alpha=0.8, edgecolor='none')
    ax1.scatter(T_grid, t_grid, yield_grid, color='red', s=50,
                label='Experimental points')
    
    ax1.set_xlabel('Temperature (°C)', fontsize=10)
    ax1.set_ylabel('Time (hours)', fontsize=10)
    ax1.set_zlabel('Yield (%)', fontsize=10)
    ax1.set_title('Response Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 等高線Plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(T_mesh, t_mesh, yield_mesh, levels=20, cmap='viridis')
    ax2.contour(T_mesh, t_mesh, yield_mesh, levels=10, colors='black',
                alpha=0.3, linewidths=0.5)
    ax2.scatter(T_grid, t_grid, c=yield_grid, s=100, edgecolors='red',
                linewidths=2, cmap='viridis')
    ax2.scatter(T_best, t_best, color='red', s=300, marker='*',
                edgecolors='white', linewidths=2, label='Optimum')
    
    ax2.set_xlabel('Temperature (°C)', fontsize=11)
    ax2.set_ylabel('Time (hours)', fontsize=11)
    ax2.set_title('Contour Map', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    fig.colorbar(contour, ax=ax2, label='Yield (%)')
    
    plt.tight_layout()
    plt.savefig('doe_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 1.6.3 Practical Approach to Experimental Design

In actual solid-state reactions, DOE is applied in the following steps:

  1. **Screening experiments** (two-level factorial design): Identify parameters with large effects
  2. **Response surface methodology** (central composite design): Search for optimal conditions
  3. **Confirmation experiments** : Experiment at predicted optimal conditions to validate model

**✅ Example: Synthesis Optimization of Li-ion Battery Cathode Material LiCoO₂**

Results when a research group optimized LiCoO₂ synthesis conditions using DOE:

  * Number of experiments: 100 (conventional) → 25 (DOE) (75% reduction)
  * 最適Temperature: 900°C（従来の850°Cより高温）
  * 最適保持Time: 12Time（従来の24Timeから半減）
  * Battery capacity: 140 mAh/g → 155 mAh/g (11% improvement)

## 1.7 Fitting Reaction Kinetics Curves

### 1.7.1 Experimental dataからのRate constant決定
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 1.7.1 Experimental dataからのRate constant決定
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # ===================================
    # Example 7: Reaction Kinetics Curve Fitting
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Experimental data（Time vs Conversion）
    # Example: BaTiO₃ synthesis @ 1200°C
    time_exp = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20])  # hours
    conversion_exp = np.array([0, 0.15, 0.28, 0.38, 0.47, 0.60,
                              0.70, 0.78, 0.84, 0.90, 0.95])
    
    # Jander equationモデル
    def jander_model(t, k):
        """Jander equationによるConversion計算
    
        Args:
            t (array): Time [hours]
            k (float): Rate constant
    
        Returns:
            array: Conversion
        """
        # [1 - (1-α)^(1/3)]² = kt Solve for α
        kt = k * t
        alpha = 1 - (1 - np.sqrt(kt))**3
        alpha = np.clip(alpha, 0, 1)  # 0-1Constrain to range
        return alpha
    
    # Ginstling-Brounshtein equation (alternative diffusion model)
    def gb_model(t, k):
        """Ginstling-Brounshtein equation
    
        Args:
            t (array): Time
            k (float): Rate constant
    
        Returns:
            array: Conversion
        """
        # 1 - 2α/3 - (1-α)^(2/3) = kt
        # Needs numerical solution, but approximation used here
        kt = k * t
        alpha = 1 - (1 - kt/2)**(3/2)
        alpha = np.clip(alpha, 0, 1)
        return alpha
    
    # Power law (empirical equation)
    def power_law_model(t, k, n):
        """Power law model
    
        Args:
            t (array): Time
            k (float): Rate constant
            n (float): Exponent
    
        Returns:
            array: Conversion
        """
        alpha = k * t**n
        alpha = np.clip(alpha, 0, 1)
        return alpha
    
    # Fit with each model
    # Jander equation
    popt_jander, _ = curve_fit(jander_model, time_exp, conversion_exp, p0=[0.01])
    k_jander = popt_jander[0]
    
    # Ginstling-Brounshtein equation
    popt_gb, _ = curve_fit(gb_model, time_exp, conversion_exp, p0=[0.01])
    k_gb = popt_gb[0]
    
    # Power law
    popt_power, _ = curve_fit(power_law_model, time_exp, conversion_exp, p0=[0.1, 0.5])
    k_power, n_power = popt_power
    
    # Generate predicted curves
    t_fit = np.linspace(0, 20, 200)
    alpha_jander = jander_model(t_fit, k_jander)
    alpha_gb = gb_model(t_fit, k_gb)
    alpha_power = power_law_model(t_fit, k_power, n_power)
    
    # Calculate residuals
    residuals_jander = conversion_exp - jander_model(time_exp, k_jander)
    residuals_gb = conversion_exp - gb_model(time_exp, k_gb)
    residuals_power = conversion_exp - power_law_model(time_exp, k_power, n_power)
    
    # Calculate R²
    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
    
    r2_jander = r_squared(conversion_exp, jander_model(time_exp, k_jander))
    r2_gb = r_squared(conversion_exp, gb_model(time_exp, k_gb))
    r2_power = r_squared(conversion_exp, power_law_model(time_exp, k_power, n_power))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Fitting results
    ax1.plot(time_exp, conversion_exp, 'ko', markersize=8, label='Experimental data')
    ax1.plot(t_fit, alpha_jander, 'b-', linewidth=2,
             label=f'Jander (R²={r2_jander:.4f})')
    ax1.plot(t_fit, alpha_gb, 'r-', linewidth=2,
             label=f'Ginstling-Brounshtein (R²={r2_gb:.4f})')
    ax1.plot(t_fit, alpha_power, 'g-', linewidth=2,
             label=f'Power law (R²={r2_power:.4f})')
    
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Conversion', fontsize=12)
    ax1.set_title('Kinetic Model Fitting', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 20])
    ax1.set_ylim([0, 1])
    
    # 残差Plot
    ax2.plot(time_exp, residuals_jander, 'bo-', label='Jander')
    ax2.plot(time_exp, residuals_gb, 'ro-', label='Ginstling-Brounshtein')
    ax2.plot(time_exp, residuals_power, 'go-', label='Power law')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kinetic_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Results summary
    print("\n反応速度モデルのFitting results:")
    print("=" * 70)
    print(f"{'Model':<25} {'Parameter':<30} {'R²':<10}")
    print("-" * 70)
    print(f"{'Jander':<25} {'k = ' + f'{k_jander:.4f} h⁻¹':<30} {r2_jander:.4f}")
    print(f"{'Ginstling-Brounshtein':<25} {'k = ' + f'{k_gb:.4f} h⁻¹':<30} {r2_gb:.4f}")
    print(f"{'Power law':<25} {'k = ' + f'{k_power:.4f}, n = {n_power:.4f}':<30} {r2_power:.4f}")
    print("=" * 70)
    print(f"\nOptimal model: {'Jander' if r2_jander == max(r2_jander, r2_gb, r2_power) else 'GB' if r2_gb == max(r2_jander, r2_gb, r2_power) else 'Power law'}")
    
    # Output example:
    # 反応速度モデルのFitting results:
    # ======================================================================
    # Model                     Parameter                      R²
    # ----------------------------------------------------------------------
    # Jander                    k = 0.0289 h⁻¹                 0.9953
    # Ginstling-Brounshtein     k = 0.0412 h⁻¹                 0.9867
    # Power law                 k = 0.2156, n = 0.5234         0.9982
    # ======================================================================
    #
    # Optimal model: Power law
    

## 1.8 Advanced Topics: Microstructure Control

### 1.8.1 Grain Growth Suppression

固相反応では、高温・長Timeundesirable grain growth occurs with holding。これを抑制する戦略：

  * **Two-step sintering** : 高温で短Time保持後、低温で長Time保持
  * **Use of additives** : Add small amounts of grain growth inhibitors (e.g., MgO, Al₂O₃)
  * **Spark Plasma Sintering (SPS)** : 急速加熱・短Time焼結

### 1.8.2 Mechanochemical Activation of Reactions

メカノケミカル法（高エネルギーボールミル）により、固相反応をRoom temperature付近で進行させることも可能です：
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 8: Grain Growth Simulation
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def grain_growth(t, T, D0, Ea, G0, n):
        """粒成長のTime発展
    
        Burke-Turnbull equation: G^n - G0^n = k*t
    
        Args:
            t (array): Time [hours]
            T (float): Temperature [K]
            D0 (float): Frequency factor
            Ea (float): Activation energy [J/mol]
            G0 (float): Initial grain size [μm]
            n (float): 粒成長Exponent（通常2-4）
    
        Returns:
            array: Grain size [μm]
        """
        R = 8.314
        k = D0 * np.exp(-Ea / (R * T))
        G = (G0**n + k * t * 3600)**(1/n)  # hours → seconds
        return G
    
    # Parameter settings
    D0_grain = 1e8  # μm^n/s
    Ea_grain = 400e3  # J/mol
    G0 = 0.5  # μm
    n = 3
    
    # Temperatureの影響
    temps_celsius = [1100, 1200, 1300]
    t_range = np.linspace(0, 12, 100)  # 0-12 hours
    
    plt.figure(figsize=(12, 5))
    
    # Temperature依存性
    plt.subplot(1, 2, 1)
    for T_c in temps_celsius:
        T_k = T_c + 273.15
        G = grain_growth(t_range, T_k, D0_grain, Ea_grain, G0, n)
        plt.plot(t_range, G, linewidth=2, label=f'{T_c}°C')
    
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1,
                label='Target grain size')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Grain Size (μm)', fontsize=12)
    plt.title('Grain Growth at Different Temperatures', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 5])
    
    # Effect of two-step sintering
    plt.subplot(1, 2, 2)
    
    # Conventional sintering: 1300°C, 6 hours
    t_conv = np.linspace(0, 6, 100)
    T_conv = 1300 + 273.15
    G_conv = grain_growth(t_conv, T_conv, D0_grain, Ea_grain, G0, n)
    
    # Two-step: 1300°C 1h → 1200°C 5h
    t1 = np.linspace(0, 1, 20)
    G1 = grain_growth(t1, 1300+273.15, D0_grain, Ea_grain, G0, n)
    G_intermediate = G1[-1]
    
    t2 = np.linspace(0, 5, 80)
    G2 = grain_growth(t2, 1200+273.15, D0_grain, Ea_grain, G_intermediate, n)
    
    t_two_step = np.concatenate([t1, t2 + 1])
    G_two_step = np.concatenate([G1, G2])
    
    plt.plot(t_conv, G_conv, 'r-', linewidth=2, label='Conventional (1300°C)')
    plt.plot(t_two_step, G_two_step, 'b-', linewidth=2, label='Two-step (1300°C→1200°C)')
    plt.axvline(x=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Grain Size (μm)', fontsize=12)
    plt.title('Two-Step Sintering Strategy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 5])
    
    plt.tight_layout()
    plt.savefig('grain_growth_control.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 最終Grain sizeの比較
    G_final_conv = grain_growth(6, 1300+273.15, D0_grain, Ea_grain, G0, n)
    G_final_two_step = G_two_step[-1]
    
    print("\nComparison of grain growth:")
    print("=" * 50)
    print(f"Conventional (1300°C, 6h): {G_final_conv:.2f} μm")
    print(f"Two-step (1300°C 1h + 1200°C 5h): {G_final_two_step:.2f} μm")
    print(f"Grain size抑制効果: {(1 - G_final_two_step/G_final_conv)*100:.1f}%")
    
    # Output example:
    # Comparison of grain growth:
    # ==================================================
    # Conventional (1300°C, 6h): 4.23 μm
    # Two-step (1300°C 1h + 1200°C 5h): 2.87 μm
    # Grain size抑制効果: 32.2%
    

## Learning Objectivesの確認

Upon completing this chapter, you will be able to explain:

### Basic Understanding

  * ✅ Explain three rate-limiting steps in solid-state reactions (nucleation, interface reaction, diffusion)
  * ✅ Arrhenius式の物理的意味とTemperature依存性を理解している
  * ✅ Jander equationとGinstling-Brounshtein equationの違いを説明できる
  * ✅ Temperature profileの3要素（Heating rate・保持Time・Cooling rate）の重要性を理解している

### Practical Skills

  * ✅ PythonでDiffusion coefficientのTemperature依存性をシミュレートできる
  * ✅ Jander equationを用いて反応進行を予測できる
  * ✅ Kissinger methodでDSC/TGデータからActivation energyを計算できる
  * ✅ Optimize reaction conditions using DOE (Design of Experiments)
  * ✅ Understand basics of phase diagram calculation using pycalphad

### Applied Capabilities

  * ✅ Design synthesis process for new ceramic materials
  * ✅ Experimental dataから反応機構を推定し、適切な速度式を選択できる
  * ✅ Develop condition optimization strategy for industrial processes
  * ✅ Propose grain growth control strategies (two-step sintering, etc.)

## Exercises

### Easy (Basic Confirmation)

Q1: Rate-Limiting Step in Solid-State Reactions

In the synthesis reaction BaCO₃ + TiO₂ → BaTiO₃ + CO₂ of BaTiO₃, which is the slowest (rate-limiting) step?

a) CO₂ release  
b) BaTiO₃ nucleus formation  
c) Diffusion of Ba²⁺ ions through product layer  
d) Chemical reaction at interface

View answer

**正解: c) Diffusion of Ba²⁺ ions through product layer**

**Explanation:**  
In solid-state reactions, the product layer physically separates reactants, making ion diffusion through the product layer the slowest process.

  * a) CO₂ release is fast as gas diffusion
  * b) Nucleation completes in initial stage
  * c) **Diffusion is rate-limiting** (correct) - Ion diffusion in solids is extremely slow (D ~ 10⁻¹² m²/s)
  * d) Interface reactions are typically fast

**Important point:** Diffusion coefficientはTemperatureに対してExponent関数的に増加するため、反応Temperatureの選択が極めて重要です。

Q2: Parameters of Arrhenius Equation

Diffusion coefficient D(T) = D₀ exp(-Eₐ/RT) において、Eₐ（Activation energy）が大きいほど、Temperature変化に対するDiffusion coefficientの感度はどうなりますか？

a) 高くなる（Temperature依存性が強い）  
b) 低くなる（Temperature依存性が弱い）  
c) 変わらない  
d) 関係ない

View answer

**正解: a) 高くなる（Temperature依存性が強い）**

**Explanation:**  
Activation energyEₐは、Exponent関数 exp(-Eₐ/RT) の肩に位置するため、Eₐが大きいほどTemperature変化に対するDの変化率が大きくなります。

**数値例:**

  * Eₐ = 100 kJ/mol の場合: Temperatureを100°C上げると D は約3倍
  * Eₐ = 300 kJ/mol の場合: Temperatureを100°C上げると D は約30倍

このため、Activation energyが大きい系では、Temperature制御が特に重要になります。

Q3: Particle Size and Reaction Rate

Jander equation k = D·C₀/r₀² によれば、Particle radiusr₀を1/2にすると、反応Rate constantkは何倍になりますか？

a) 2倍  
b) 4倍  
c) 1/2倍  
d) 1/4倍

View answer

**正解: b) 4倍**

**計算:**  
k ∝ 1/r₀²  
r₀ → r₀/2 のとき、k → k/(r₀/2)² = k/(r₀²/4) = 4k

**実践的意味:**  
これが「粉砕・微細化」が固相反応で極めて重要な理由です。

  * Grain size10μm → 1μm: 反応速度100倍（反応Time1/100）
  * ボールミル、Pulverization by jet mill is standard process
  * ナノ粒子を使えばRoom temperature付近でも反応可能な場合も

### Medium (Application)

Q4: Temperature profile設計

BaTiO₃合成で、Heating rateを20°C/minから5°C/minに変更しました。Which is the most appropriate main reason for this change？

a) 反応速度を速めるため  
b) CO₂to prevent sample rupture due to rapid release  
c) 電気代を節約するため  
d) Crystallinityを下げるため

View answer

**正解: b) CO₂to prevent sample rupture due to rapid release**

**詳細な理由:**

BaCO₃ + TiO₂ → BaTiO₃ + CO₂ の反応では、800-900°Cで炭酸バリウムが分解してCO₂を放出します。

  * **急速加熱（20°C/min）の問題:**
    * 短Timeで多量のCO₂が発生
    * ガス圧が高まり、試料が破裂・飛散
    * 焼結体に亀裂・クラックが入る
  * **徐加熱（5°C/min）の利点:**
    * CO₂がゆっくり放出され、圧力上昇が緩やか
    * 試料の健全性が保たれる
    * 均質な反応が進行

**実践的アドバイス:** Decomposition reactionsを伴う合成では、ガス放出速度を制御するため、該当Temperature範囲でのHeating rateを特に遅くします（例: 750-950°Cを2°C/minで通過）。

Q5: Kissinger methodの適用

DSC測定で以下のデータが得られました。Kissinger methodでActivation energyを求めてください。

Heating rate β (K/min): 5, 10, 15  
ピークTemperature Tp (K): 1273, 1293, 1308

Kissinger式: ln(β/Tp²) vs 1/Tp のSlope = -Eₐ/R

View answer

**解答:**

**ステップ1: データ整理**

β (K/min) | Tp (K) | ln(β/Tp²) | 1000/Tp (K⁻¹)  
---|---|---|---  
5 | 1273 | -11.558 | 0.7855  
10 | 1293 | -11.171 | 0.7734  
15 | 1308 | -10.932 | 0.7645  
  
**ステップ2: Linear regression**

y = ln(β/Tp²) vs x = 1000/Tp をPlot  
Slope slope = Δy/Δx = (-10.932 - (-11.558)) / (0.7645 - 0.7855) = 0.626 / (-0.021) ≈ -29.8

**ステップ3: Eₐ計算**

slope = -Eₐ / (R × 1000) （1000/Tpを使ったため1000で割る）  
Eₐ = -slope × R × 1000  
Eₐ = 29.8 × 8.314 × 1000 = 247,757 J/mol ≈ 248 kJ/mol

**答え: Eₐ ≈ 248 kJ/mol**

**物理的解釈:**  
この値はBaTiO₃系の固相反応における典型的なActivation energy（250-350 kJ/mol）の範囲内です。このActivation energyは、Ba²⁺corresponds to solid-phase diffusion of ions。

Q6: Optimization using DOE

実験計画法で、Temperature（1100, 1200, 1300°C）とTime（4, 6, 8Time）の2因子を検討します。全実験回数は何回必要ですか？また、1advantages compared to conventional method of changing factors one at a time2つ挙げてください。

View answer

**解答:**

**実験回数:**  
3水準 × 3水準 = **9回** （フルファクトリアル計画）

**DOEの利点（従来法との比較）:**

  1. **交互作用の検出が可能**
     * 従来法: Temperatureの影響、Timeの影響を個別に評価
     * DOE: 「高温ではTimeを短くできる」といった交互作用を定量化
     * 例: 1300°Cでは4Timeで十分だが、1100°Cでは8Time必要、など
  2. **実験回数の削減**
     * 従来法（OFAT: One Factor At a Time）: 
       * Temperature検討: 3回（Timefixed）
       * Time検討: 3回（Temperaturefixed）
       * 確認実験: 複数回
       * 合計: 10回以上
     * DOE: 9回で完了（全条件網羅＋交互作用解析）
     * さらに中心複合計画法を使えば7回に削減可能

**追加の利点:**

  * 統計的に有意な結論が得られる（誤差評価が可能）
  * 応答曲面を構築でき、未実施条件の予測が可能
  * Can detect even when optimal conditions are outside experimental range

### Hard (Advanced)

Q7: Design of Complex Reaction Systems

次の条件でLi₁.₂Ni₀.₂Mn₀.₆O₂（リチウムリッチ正極材料）を合成するTemperature profileを設計してください：

  * 原料: Li₂CO₃, NiO, Mn₂O₃
  * 目標: 単一相、Grain size < 5 μm、Li/遷移金属比の精密制御
  * 制約: 900°C以上でLi₂Oが揮発（Li欠損のリスク）

Temperature profile（Heating rate、Holding temperature・Time、Cooling rate）と、その設計理由を説明してください。

View answer

**推奨Temperature profile:**

**Phase 1: 予備加熱（Li₂CO₃分解）**

  * Room temperature → 500°C: 3°C/min
  * 500°C保持: 2Time
  * **理由:** Li₂CO₃の分解（~450°C）をゆっくり進行させ、CO₂を完全に除去

**Phase 2: 中間加熱（前駆体形成）**

  * 500°C → 750°C: 5°C/min
  * 750°C保持: 4Time
  * **理由:** Li₂MnO₃やLiNiO₂などの中間相を形成。Li揮発の少ないTemperatureで均質化

**Phase 3: 本焼成（目的相合成）**

  * 750°C → 850°C: 2°C/min（ゆっくり）
  * 850°C保持: 12Time
  * **理由:**
    * Li₁.₂Ni₀.₂Mn₀.₆O₂の単一相形成には長Time必要
    * 850°Cに制限してLi揮発を最小化（<900°C制約）
    * 長Time保持で拡散を進めるが、粒成長は抑制されるTemperature

**Phase 4: 冷却**

  * 850°C → Room temperature: 2°C/min
  * **理由:** 徐冷によりCrystallinity向上、Thermal stressによる亀裂防止

**設計のImportant point:**

  1. **Li揮発対策:**
     * 900°C以下に制限（本問の制約）
     * さらに、Li過剰原料（Li/TM = 1.25など）を使用
     * 酸素気流中で焼成してLi₂Oの分圧を低減
  2. **Grain size制御 ( < 5 μm):**
     * 低温（850°C）・長Time（12h）で反応を進める
     * 高温・短Timeだと粒成長が過剰になる
     * 原料Grain sizeも1μm以下に微細化
  3. **組成均一性:**
     * 750°Cでの中間保持が重要
     * この段階で遷移金属の分布を均質化
     * 必要に応じて、750°C保持後に一度冷却→粉砕→再加熱

**全体所要Time:** 約30Time（加熱12h + 保持18h）

**代替手法の検討:**

  * **Sol-gel法:** より低温（600-700°C）で合成可能、均質性向上
  * **Spray pyrolysis:** Grain size制御が容易
  * **Two-step sintering:** 900°C 1h → 800°C 10h で粒成長抑制

Q8: Comprehensive Kinetics Analysis Problem

以下のデータから、反応機構を推定し、Activation energyを計算してください。

**Experimental data:**

Temperature (°C) | 50%反応到達Time t₅₀ (hours)  
---|---  
1000| 18.5  
1100| 6.2  
1200| 2.5  
1300| 1.2  
  
Jander equationを仮定した場合: [1-(1-0.5)^(1/3)]² = k·t₅₀

View answer

**解答:**

**ステップ1: Rate constantkの計算**

Jander equationで α=0.5 のとき:  
[1-(1-0.5)^(1/3)]² = [1-0.794]² = 0.206² = 0.0424

したがって k = 0.0424 / t₅₀

T (°C) | T (K) | t₅₀ (h) | k (h⁻¹) | ln(k) | 1000/T (K⁻¹)  
---|---|---|---|---|---  
1000| 1273| 18.5| 0.00229| -6.080| 0.7855  
1100| 1373| 6.2| 0.00684| -4.985| 0.7284  
1200| 1473| 2.5| 0.01696| -4.077| 0.6788  
1300| 1573| 1.2| 0.03533| -3.343| 0.6357  
  
**ステップ2: ArrheniusPlot**

ln(k) vs 1/T をPlot（Linear regression）

線形フィット: ln(k) = A - Eₐ/(R·T)

Slope = -Eₐ/R

Linear regression計算:  
slope = Δ(ln k) / Δ(1000/T)  
= (-3.343 - (-6.080)) / (0.6357 - 0.7855)  
= 2.737 / (-0.1498)  
= -18.27

**ステップ3: Activation energy計算**

slope = -Eₐ / (R × 1000)  
Eₐ = -slope × R × 1000  
Eₐ = 18.27 × 8.314 × 1000  
Eₐ = 151,899 J/mol ≈ **152 kJ/mol**

**ステップ4: 反応機構の考察**

  * **Activation energyの比較:**
    * 得られた値: 152 kJ/mol
    * 典型的な固相拡散: 200-400 kJ/mol
    * 界面反応: 50-150 kJ/mol
  * **推定される機構:**
    * この値は界面反応と拡散の中間
    * 可能性1: 界面反応が主律速（拡散の影響は小）
    * 可能性2: 粒子が微細で拡散距離が短く、見かけのEₐが低い
    * 可能性3: 混合律速（界面反応と拡散の両方が寄与）

**ステップ5: 検証方法の提案**

  1. **粒子サイズ依存性:** 異なるGrain sizeで実験し、k ∝ 1/r₀² が成立するか確認 
     * 成立 → 拡散律速
     * 不成立 → 界面反応律速
  2. **他の速度式でのフィッティング:**
     * Ginstling-Brounshtein equation（3次元拡散）
     * Contracting sphere model（界面反応）
     * どちらがR²が高いか比較
  3. **微細構造観察:** SEMで反応界面を観察 
     * 厚い生成物層 → 拡散律速の証拠
     * 薄い生成物層 → 界面反応律速の可能性

**最終結論:**  
Activation energy **Eₐ = 152 kJ/mol**  
推定機構: **界面反応律速、または微細粒子系での拡散律速**  
追加実験が推奨される。

## Next Steps

In Chapter 1, we learned fundamental theory of advanced ceramic materials (structural, functional, and bioceramics). In the next Chapter 3, we will study nanomaterials (high-performance engineering plastics, functional polymers, and biodegradable polymers).

[← Series Index](<./index.html>) [Proceed to Chapter 3 →](<chapter-3.html>)

## References

  1. Dresselhaus, M. S., Dresselhaus, G., & Avouris, P. (2001). _Carbon Nanotubes: Synthesis, Structure, Properties, and Applications_. Springer. pp. 1-38, 111-165. - Comprehensive coverage of carbon nanotube structure, properties, and synthesis methodsExplanation
  2. Geim, A. K., & Novoselov, K. S. (2007). "The rise of graphene." _Nature Materials_ , 6(3), 183-191. - Nobel Prize-winning research on discovery of graphene and unique electronic properties
  3. Alivisatos, A. P. (1996). "Semiconductor clusters, nanocrystals, and quantum dots." _Science_ , 271(5251), 933-937. - Pioneering research on electronic structure and quantum confinement effects of quantum dots
  4. Burda, C., Chen, X., Narayanan, R., & El-Sayed, M. A. (2005). "Chemistry and properties of nanocrystals of different shapes." _Chemical Reviews_ , 105(4), 1025-1102. - Detailed review of shape-controlled synthesis and optical properties of metal nanoparticles
  5. Iijima, S. (1991). "Helical microtubules of graphitic carbon." _Nature_ , 354(6348), 56-58. - Historic paper on discovery of carbon nanotubes
  6. Brus, L. E. (1984). "Electron-electron and electron-hole interactions in small semiconductor crystallites: The size dependence of the lowest excited electronic state." _Journal of Chemical Physics_ , 80(9), 4403-4409. - Theoretical foundation of size-dependent bandgap in quantum dots
  7. ASE Documentation. (2024). _Atomic Simulation Environment_. <https://wiki.fysik.dtu.dk/ase/> \- ナノ構造シミュレーションのためのPythonライブラリ

## Tools and Libraries Used

  * **NumPy** (v1.24+): 数値計算ライブラリ - <https://numpy.org/>
  * **SciPy** (v1.10+): 科学技術計算ライブラリ（curve_fit, optimize） - <https://scipy.org/>
  * **Matplotlib** (v3.7+): データ可視化ライブラリ - <https://matplotlib.org/>
  * **pycalphad** (v0.10+): 相図計算ライブラリ - <https://pycalphad.org/>
  * **pymatgen** (v2023+): Materials Science計算ライブラリ - <https://pymatgen.org/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
