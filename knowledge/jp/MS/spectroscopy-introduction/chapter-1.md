---
title: 第1章：分光分析の基礎
chapter_title: 第1章：分光分析の基礎
subtitle: 光と物質の相互作用から読み解く材料の本質
---

## イントロダクション

分光分析（Spectroscopy）は、光と物質の相互作用を利用して、材料の電子状態、化学結合、構造、組成を解明する強力な分析手法です。この章では、あらゆる分光法の基盤となる量子力学的原理を学び、Beer-Lambert則、遷移モーメント、選択則など、実験データを解釈するための理論的基礎を習得します。

**なぜ分光分析が重要なのか？**  
分光分析は非破壊的で、微量試料に適用でき、多様なエネルギー領域（X線からマイクロ波まで）で材料の異なる性質を探ることができます。半導体のバンドギャップ測定、有機分子の官能基同定、触媒表面の化学状態解析など、Materials Informaticsにおける材料探索・設計の基盤技術です。 

## 1\. 光と物質の相互作用の基礎

### 1.1 電磁波の性質

光は電磁波として波動性と粒子性の二重性を持ちます。波動としての光は波長 $\lambda$（nm）や波数 $\tilde{\nu}$（cm-1）で表現され、粒子としての光子は以下のエネルギーを持ちます：

$$E = h\nu = \frac{hc}{\lambda} = hc\tilde{\nu}$$

ここで、$h = 6.626 \times 10^{-34}$ J·s（プランク定数）、$c = 2.998 \times 10^8$ m/s（光速）です。

**エネルギー領域と対応する遷移**  

  * **X線（0.01-10 nm）** : 内殻電子励起（XPS, XRF）
  * **UV-Vis（200-800 nm）** : 価電子励起、HOMO-LUMO遷移
  * **赤外（2.5-25 μm）** : 分子振動
  * **マイクロ波（0.1-10 cm）** : 分子回転

#### コード例1: Planck関数とエネルギー計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # プランク定数と光速
    h = 6.62607015e-34  # J·s
    c = 2.99792458e8    # m/s
    eV = 1.602176634e-19  # J (1 eV)
    
    def wavelength_to_energy(wavelength_nm):
        """
        波長（nm）から光子エネルギー（eV）を計算
    
        Parameters:
        -----------
        wavelength_nm : float or array
            波長（nanometer）
    
        Returns:
        --------
        energy_eV : float or array
            光子エネルギー（electron volt）
        """
        wavelength_m = wavelength_nm * 1e-9
        energy_J = h * c / wavelength_m
        energy_eV = energy_J / eV
        return energy_eV
    
    def wavenumber_to_energy(wavenumber_cm):
        """
        波数（cm^-1）から光子エネルギー（eV）を計算
    
        Parameters:
        -----------
        wavenumber_cm : float or array
            波数（cm^-1）
    
        Returns:
        --------
        energy_eV : float or array
            光子エネルギー（eV）
        """
        energy_J = h * c * wavenumber_cm * 100  # cm^-1 to m^-1
        energy_eV = energy_J / eV
        return energy_eV
    
    # 可視光領域（380-780 nm）のエネルギー計算
    wavelengths = np.linspace(380, 780, 100)
    energies = wavelength_to_energy(wavelengths)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, energies, linewidth=2, color='#f093fb')
    plt.fill_between(wavelengths, energies, alpha=0.3, color='#f5576c')
    plt.xlabel('波長 (nm)', fontsize=12)
    plt.ylabel('光子エネルギー (eV)', fontsize=12)
    plt.title('可視光領域の波長-エネルギー関係', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wavelength_energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 具体例
    print(f"赤色光（650 nm）のエネルギー: {wavelength_to_energy(650):.3f} eV")
    print(f"青色光（450 nm）のエネルギー: {wavelength_to_energy(450):.3f} eV")
    print(f"IR振動（1000 cm^-1）のエネルギー: {wavenumber_to_energy(1000):.4f} eV")
    

### 1.2 吸収・発光・散乱の基本過程

光と物質の相互作用は主に3つの過程に分類されます：
    
    
    ```mermaid
    flowchart TD
                A[入射光] --> B{物質との相互作用}
                B -->|吸収| C[励起状態へ遷移E = E₂ - E₁]
                B -->|発光| D[基底状態へ遷移蛍光・燐光]
                B -->|散乱| E[Rayleigh散乱弾性散乱]
                B -->|散乱| F[Raman散乱非弾性散乱]
    
                style A fill:#f093fb,color:#fff
                style C fill:#f5576c,color:#fff
                style D fill:#f5576c,color:#fff
                style E fill:#a8e6cf,color:#000
                style F fill:#a8e6cf,color:#000
    ```

  * **吸収（Absorption）** : 光子のエネルギーが物質の2つのエネルギー準位の差 $\Delta E = E_2 - E_1$ と一致するとき、光子が吸収され、物質は励起状態へ遷移します。
  * **発光（Emission）** : 励起状態から基底状態へ遷移する際に光子を放出します。蛍光（Fluorescence）、燐光（Phosphorescence）、化学発光などがあります。
  * **散乱（Scattering）** : Rayleigh散乱（弾性、エネルギー変化なし）とRaman散乱（非弾性、振動エネルギー変化）に分類されます。

### 1.3 Beer-Lambert則

吸収分光法の基本法則であるBeer-Lambert則（Lambert-Beer則とも呼ばれる）は、吸光度 $A$ と試料の濃度 $c$、光路長 $l$ の関係を記述します：

$$A = \log_{10}\left(\frac{I_0}{I}\right) = \varepsilon c l$$

ここで、$I_0$ は入射光強度、$I$ は透過光強度、$\varepsilon$ はモル吸光係数（L mol-1 cm-1）です。透過率 $T$ は $T = I/I_0$ で定義され、$A = -\log_{10}(T)$ の関係があります。

#### コード例2: Beer-Lambert則シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def beer_lambert(I0, epsilon, concentration, path_length):
        """
        Beer-Lambert則による透過光強度計算
    
        Parameters:
        -----------
        I0 : float
            入射光強度
        epsilon : float
            モル吸光係数 (L mol^-1 cm^-1)
        concentration : float or array
            濃度 (mol/L)
        path_length : float
            光路長 (cm)
    
        Returns:
        --------
        I : float or array
            透過光強度
        A : float or array
            吸光度
        T : float or array
            透過率
        """
        A = epsilon * concentration * path_length
        T = 10**(-A)
        I = I0 * T
        return I, A, T
    
    # パラメータ設定
    I0 = 1.0  # 入射光強度（規格化）
    epsilon = 1000  # モル吸光係数（典型的な有機色素）
    path_length = 1.0  # 光路長 1 cm
    concentrations = np.linspace(0, 1e-4, 100)  # 濃度範囲（mol/L）
    
    # 計算
    I, A, T = beer_lambert(I0, epsilon, concentrations, path_length)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 吸光度 vs 濃度
    ax1.plot(concentrations * 1e6, A, linewidth=2, color='#f093fb', label='吸光度')
    ax1.set_xlabel('濃度 (μmol/L)', fontsize=12)
    ax1.set_ylabel('吸光度', fontsize=12)
    ax1.set_title('Beer-Lambert則：濃度依存性', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 透過率 vs 濃度
    ax2.plot(concentrations * 1e6, T * 100, linewidth=2, color='#f5576c', label='透過率')
    ax2.set_xlabel('濃度 (μmol/L)', fontsize=12)
    ax2.set_ylabel('透過率 (%)', fontsize=12)
    ax2.set_title('透過率の濃度依存性', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('beer_lambert.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 定量分析例：吸光度から濃度を逆算
    measured_A = 0.5
    calculated_c = measured_A / (epsilon * path_length)
    print(f"測定吸光度: {measured_A}")
    print(f"逆算された濃度: {calculated_c * 1e6:.2f} μmol/L")
    

## 2\. 量子力学的基礎

### 2.1 電子状態と振動状態

分子のエネルギー準位は、電子状態、振動状態、回転状態の3つの自由度に分離できます（Born-Oppenheimer近似）：

$$E_{\text{total}} = E_{\text{electronic}} + E_{\text{vibrational}} + E_{\text{rotational}}$$

典型的なエネルギースケールは以下の通りです：

  * $E_{\text{electronic}} \sim 1-10$ eV（UV-Vis領域）
  * $E_{\text{vibrational}} \sim 0.05-0.5$ eV（赤外領域）
  * $E_{\text{rotational}} \sim 10^{-4}-10^{-3}$ eV（マイクロ波領域）

### 2.2 遷移モーメントとFermiの黄金則

光吸収による状態 $\left|\psi_i\right\rangle$ から $\left|\psi_f\right\rangle$ への遷移確率は、Fermiの黄金則で与えられます：

$$W_{i \to f} = \frac{2\pi}{\hbar} \left| \left\langle \psi_f \right| \hat{H}_{\text{int}} \left| \psi_i \right\rangle \right|^2 \rho(E_f)$$

ここで、$\hat{H}_{\text{int}}$ は光と物質の相互作用ハミルトニアン、$\rho(E_f)$ は終状態の状態密度です。電気双極子近似では、遷移双極子モーメント $\boldsymbol{\mu}_{fi}$ が重要になります：

$$\boldsymbol{\mu}_{fi} = \left\langle \psi_f \right| \hat{\boldsymbol{\mu}} \left| \psi_i \right\rangle = \int \psi_f^* \hat{\boldsymbol{\mu}} \psi_i \, d\tau$$

遷移双極子モーメントが0でない（$\boldsymbol{\mu}_{fi} \neq 0$）とき、その遷移は「許容遷移（allowed transition）」、0のとき「禁制遷移（forbidden transition）」と呼ばれます。

#### コード例3: Fermiの黄金則による遷移確率計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    hbar = 1.054571817e-34  # J·s
    eV = 1.602176634e-19    # J
    
    def transition_probability(transition_dipole_moment, state_density, energy_eV):
        """
        Fermiの黄金則による遷移確率計算（簡略化版）
    
        Parameters:
        -----------
        transition_dipole_moment : float
            遷移双極子モーメント（Debye）
        state_density : float
            状態密度（eV^-1）
        energy_eV : float
            遷移エネルギー（eV）
    
        Returns:
        --------
        W : float
            遷移確率（s^-1）
        """
        # Debye to C·m変換（1 D ≈ 3.336e-30 C·m）
        mu = transition_dipole_moment * 3.336e-30
    
        # 簡略化された遷移確率（電気双極子近似）
        # 実際の計算では電場の強度も考慮
        W = (2 * np.pi / hbar) * (mu**2) * state_density * eV
        return W
    
    def franck_condon_factor(n_initial, n_final, displacement):
        """
        Franck-Condon因子の近似計算（調和振動子近似）
    
        Parameters:
        -----------
        n_initial : int
            初期振動量子数
        n_final : int
            終振動量子数
        displacement : float
            平衡位置のずれ（無次元）
    
        Returns:
        --------
        fc_factor : float
            Franck-Condon因子
        """
        # 簡略化：Gaussian近似
        delta_n = n_final - n_initial
        fc_factor = np.exp(-displacement**2 / 2) * (displacement**delta_n / np.math.factorial(abs(delta_n)))
        return abs(fc_factor)**2
    
    # 遷移双極子モーメントと遷移確率の関係
    dipole_moments = np.linspace(0.1, 5.0, 50)  # Debye
    state_density = 1e15  # eV^-1（典型的な固体の値）
    energy = 2.0  # eV
    
    transition_probs = [transition_probability(mu, state_density, energy) for mu in dipole_moments]
    
    # Franck-Condon因子の計算（v=0 → v'遷移）
    displacements = np.linspace(0, 3, 4)
    vibrational_levels = np.arange(0, 8)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 遷移双極子モーメントと遷移確率
    ax1.plot(dipole_moments, np.array(transition_probs) * 1e-15, linewidth=2, color='#f093fb')
    ax1.set_xlabel('遷移双極子モーメント (Debye)', fontsize=12)
    ax1.set_ylabel('遷移確率 (×10¹⁵ s⁻¹)', fontsize=12)
    ax1.set_title('遷移双極子モーメントと遷移確率', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Franck-Condon因子
    for d in displacements:
        fc_factors = [franck_condon_factor(0, v, d) for v in vibrational_levels]
        ax2.plot(vibrational_levels, fc_factors, marker='o', linewidth=2, label=f'Δq = {d:.1f}')
    
    ax2.set_xlabel("終振動量子数 v'", fontsize=12)
    ax2.set_ylabel('Franck-Condon因子', fontsize=12)
    ax2.set_title('Franck-Condon因子の計算', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transition_probability.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 2.3 選択則

選択則（Selection Rules）は、どの遷移が許容され、どの遷移が禁制されるかを決定する量子力学的規則です。主要な選択則を以下に示します：

**主要な選択則**

  * **電気双極子遷移（UV-Vis, IR）** : $\Delta l = \pm 1$（軌道角運動量量子数）、$\Delta S = 0$（スピン）、$\Delta v = \pm 1$（振動量子数、調和振動子近似）
  * **Laporte則（中心対称分子）** : $g \leftrightarrow u$ のみ許容（$g \leftrightarrow g$ や $u \leftrightarrow u$ は禁制）
  * **スピン選択則** : 一重項-一重項、三重項-三重項遷移は許容、一重項-三重項遷移は禁制（ただしスピン軌道相互作用で緩和）
  * **Raman散乱** : 分極率の変化 $\partial \alpha / \partial Q \neq 0$（IR吸収とは相補的）

### 2.4 Franck-Condon原理

Franck-Condon原理は、電子遷移が振動の時間スケール（~10-13 s）に比べて非常に高速（~10-15 s）であるため、電子遷移中に原子核の位置がほとんど変化しないことを述べています。これにより、吸収・発光スペクトルに振動構造が現れます。

#### コード例4: Franck-Condon原理による吸収スペクトルシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import factorial
    
    def harmonic_oscillator_wavefunction(x, n, omega=1.0, m=1.0, hbar=1.0):
        """
        調和振動子の波動関数
    
        Parameters:
        -----------
        x : array
            座標
        n : int
            量子数
        omega : float
            角振動数
        m : float
            質量
        hbar : float
            換算プランク定数
    
        Returns:
        --------
        psi : array
            波動関数
        """
        alpha = np.sqrt(m * omega / hbar)
        norm = (alpha / np.pi)**0.25 / np.sqrt(2**n * factorial(n))
        hermite = np.polynomial.hermite.hermval(alpha * x, [0]*n + [1])
        psi = norm * hermite * np.exp(-alpha**2 * x**2 / 2)
        return psi
    
    def franck_condon_spectrum(displacement=1.5, n_levels=6):
        """
        Franck-Condon原理に基づく吸収スペクトルシミュレーション
    
        Parameters:
        -----------
        displacement : float
            励起状態と基底状態の平衡位置のずれ
        n_levels : int
            考慮する振動準位の数
    
        Returns:
        --------
        energies : array
            遷移エネルギー
        intensities : array
            相対強度
        """
        # 座標グリッド
        x = np.linspace(-6, 6, 1000)
    
        # 基底状態 v=0 の波動関数
        psi_ground = harmonic_oscillator_wavefunction(x, 0)
    
        energies = []
        intensities = []
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # ポテンシャルエネルギー曲線
        V_ground = 0.5 * x**2
        V_excited = 0.5 * (x - displacement)**2 + 3.0  # 励起状態（上方シフト）
    
        ax1.plot(x, V_ground, 'b-', linewidth=2, label='基底状態')
        ax1.plot(x, V_excited, 'r-', linewidth=2, label='励起状態')
    
        # 各振動準位への遷移
        for n in range(n_levels):
            # 励起状態 v=n の波動関数（平衡位置がdisplacementだけずれている）
            psi_excited = harmonic_oscillator_wavefunction(x - displacement, n)
    
            # Franck-Condon積分（重なり積分）
            fc_integral = np.trapz(psi_ground * psi_excited, x)
            fc_factor = fc_integral**2
    
            # 遷移エネルギー（電子遷移 + 振動エネルギー）
            E_transition = 3.0 + n * 0.2  # eV単位
    
            energies.append(E_transition)
            intensities.append(fc_factor)
    
            # ポテンシャル曲線上に振動準位を描画
            E_vib_ground = 0.5
            E_vib_excited = 3.0 + n * 0.2
            ax1.axhline(y=E_vib_excited, xmin=0.5, xmax=0.9, color='red', alpha=0.3, linewidth=1)
    
            # 遷移の矢印
            if n < 4:
                ax1.annotate('', xy=(displacement, E_vib_excited), xytext=(0, E_vib_ground),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.5, lw=1.5))
    
        ax1.set_xlabel('核間座標 (a.u.)', fontsize=12)
        ax1.set_ylabel('エネルギー (eV)', fontsize=12)
        ax1.set_title('Franck-Condon原理', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3)
    
        # 吸収スペクトル
        energies = np.array(energies)
        intensities = np.array(intensities)
    
        # Gaussian broadening
        E_range = np.linspace(2.8, 4.5, 500)
        spectrum = np.zeros_like(E_range)
        broadening = 0.05  # eV
    
        for E, I in zip(energies, intensities):
            spectrum += I * np.exp(-(E_range - E)**2 / (2 * broadening**2))
    
        ax2.plot(E_range, spectrum, linewidth=2, color='#f093fb')
        ax2.fill_between(E_range, spectrum, alpha=0.3, color='#f5576c')
        ax2.set_xlabel('光子エネルギー (eV)', fontsize=12)
        ax2.set_ylabel('吸収強度 (a.u.)', fontsize=12)
        ax2.set_title('シミュレートされた吸収スペクトル', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('franck_condon_spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        return energies, intensities
    
    # 実行
    energies, intensities = franck_condon_spectrum(displacement=1.5)
    print("遷移エネルギー (eV):", energies)
    print("Franck-Condon因子:", intensities)
    

## 3\. 分光法の分類

### 3.1 エネルギー領域別分類
    
    
    ```mermaid
    flowchart LR
                A[電磁波スペクトル] --> B[X線0.01-10 nm]
                A --> C[UV-Vis200-800 nm]
                A --> D[赤外2.5-25 μm]
                A --> E[マイクロ波0.1-10 cm]
    
                B --> B1[XPS化学状態]
                B --> B2[XRF元素分析]
    
                C --> C1[UV-Vis電子遷移]
                C --> C2[PL発光]
    
                D --> D1[FTIR振動]
                D --> D2[Raman振動]
    
                E --> E1[ESR磁気共鳴]
                E --> E2[NMR核スピン]
    
                style A fill:#f093fb,color:#fff
                style B fill:#ff6b6b,color:#fff
                style C fill:#4ecdc4,color:#fff
                style D fill:#ffe66d,color:#000
                style E fill:#a8e6cf,color:#000
    ```

### 3.2 測定原理別分類

  * **吸収分光法（Absorption Spectroscopy）** : UV-Vis, FTIR, 原子吸光（AAS）
  * **発光分光法（Emission Spectroscopy）** : 蛍光（PL）, 燐光、化学発光
  * **散乱分光法（Scattering Spectroscopy）** : Raman, Brillouin散乱
  * **共鳴分光法（Resonance Spectroscopy）** : NMR, ESR

## 4\. スペクトルの読み方

### 4.1 横軸と縦軸の変換

分光データは様々な単位系で表現されます：

  * **横軸** : 波長 $\lambda$ (nm), 波数 $\tilde{\nu}$ (cm-1), エネルギー $E$ (eV), 周波数 $\nu$ (Hz)
  * **縦軸** : 透過率 $T$ (%), 吸光度 $A$, 強度 $I$ (a.u.), モル吸光係数 $\varepsilon$

#### コード例5: 波長・波数・エネルギー変換計算機
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class SpectroscopyUnitConverter:
        """分光分析の単位変換クラス"""
    
        def __init__(self):
            self.h = 6.62607015e-34  # J·s
            self.c = 2.99792458e8    # m/s
            self.eV = 1.602176634e-19  # J
    
        def wavelength_to_wavenumber(self, wavelength_nm):
            """波長（nm）→ 波数（cm^-1）"""
            wavelength_cm = wavelength_nm * 1e-7
            return 1 / wavelength_cm
    
        def wavenumber_to_wavelength(self, wavenumber_cm):
            """波数（cm^-1）→ 波長（nm）"""
            wavelength_cm = 1 / wavenumber_cm
            return wavelength_cm * 1e7
    
        def wavelength_to_energy_eV(self, wavelength_nm):
            """波長（nm）→ エネルギー（eV）"""
            wavelength_m = wavelength_nm * 1e-9
            energy_J = self.h * self.c / wavelength_m
            return energy_J / self.eV
    
        def energy_eV_to_wavelength(self, energy_eV):
            """エネルギー（eV）→ 波長（nm）"""
            energy_J = energy_eV * self.eV
            wavelength_m = self.h * self.c / energy_J
            return wavelength_m * 1e9
    
        def wavelength_to_frequency(self, wavelength_nm):
            """波長（nm）→ 周波数（Hz）"""
            wavelength_m = wavelength_nm * 1e-9
            return self.c / wavelength_m
    
        def transmittance_to_absorbance(self, T):
            """透過率（%）→ 吸光度"""
            T_fraction = T / 100
            return -np.log10(T_fraction)
    
        def absorbance_to_transmittance(self, A):
            """吸光度 → 透過率（%）"""
            return 10**(-A) * 100
    
    # 変換器のインスタンス化
    converter = SpectroscopyUnitConverter()
    
    # UV-Vis領域の変換表
    wavelengths_nm = np.array([200, 250, 300, 400, 500, 600, 700, 800])
    wavenumbers_cm = converter.wavelength_to_wavenumber(wavelengths_nm)
    energies_eV = converter.wavelength_to_energy_eV(wavelengths_nm)
    frequencies_THz = converter.wavelength_to_frequency(wavelengths_nm) / 1e12
    
    print("=" * 70)
    print("UV-Vis領域の単位変換表")
    print("=" * 70)
    print(f"{'波長 (nm)':<12} {'波数 (cm⁻¹)':<15} {'エネルギー (eV)':<15} {'周波数 (THz)':<12}")
    print("-" * 70)
    for wl, wn, E, f in zip(wavelengths_nm, wavenumbers_cm, energies_eV, frequencies_THz):
        print(f"{wl:<12.0f} {wn:<15.0f} {E:<15.2f} {f:<12.1f}")
    
    # IR領域の変換表
    print("\n" + "=" * 70)
    print("赤外領域の単位変換表")
    print("=" * 70)
    wavenumbers_IR = np.array([4000, 3000, 2000, 1500, 1000, 500])
    wavelengths_IR = converter.wavenumber_to_wavelength(wavenumbers_IR)
    energies_IR_eV = converter.wavelength_to_energy_eV(wavelengths_IR)
    
    print(f"{'波数 (cm⁻¹)':<15} {'波長 (μm)':<15} {'エネルギー (eV)':<15}")
    print("-" * 70)
    for wn, wl, E in zip(wavenumbers_IR, wavelengths_IR / 1000, energies_IR_eV):
        print(f"{wn:<15.0f} {wl:<15.2f} {E:<15.4f}")
    
    # 透過率と吸光度の関係
    transmittances = np.linspace(1, 100, 100)
    absorbances = converter.transmittance_to_absorbance(transmittances)
    
    plt.figure(figsize=(10, 6))
    plt.plot(transmittances, absorbances, linewidth=2, color='#f093fb')
    plt.xlabel('透過率 (%)', fontsize=12)
    plt.ylabel('吸光度', fontsize=12)
    plt.title('透過率と吸光度の関係', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('transmittance_absorbance.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 4.2 ピーク位置、強度、幅の解釈

スペクトルのピークから以下の情報が得られます：

  * **ピーク位置** : エネルギー準位の差 $\Delta E$、化学結合の種類、バンドギャップ
  * **ピーク強度** : 遷移確率、濃度、モル吸光係数
  * **ピーク幅（半値全幅, FWHM）** : 不均一広がり（結晶性、欠陥）、均一広がり（寿命）

#### コード例6: ガウス・ローレンツ線形フィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def gaussian(x, amplitude, center, width):
        """
        Gaussian線形関数
    
        Parameters:
        -----------
        x : array
            横軸（波長、エネルギーなど）
        amplitude : float
            ピーク高さ
        center : float
            ピーク中心位置
        width : float
            標準偏差（FWHM = 2.355 * width）
    
        Returns:
        --------
        y : array
            Gaussian曲線
        """
        return amplitude * np.exp(-(x - center)**2 / (2 * width**2))
    
    def lorentzian(x, amplitude, center, width):
        """
        Lorentzian線形関数
    
        Parameters:
        -----------
        x : array
            横軸
        amplitude : float
            ピーク高さ
        center : float
            ピーク中心位置
        width : float
            半値半幅（HWHM）
    
        Returns:
        --------
        y : array
            Lorentzian曲線
        """
        return amplitude * (width**2 / ((x - center)**2 + width**2))
    
    def voigt(x, amplitude, center, width_g, width_l):
        """
        Voigt線形関数（GaussianとLorentzianの畳み込み）
        簡略化版：pseudo-Voigt
    
        Parameters:
        -----------
        x : array
            横軸
        amplitude : float
            ピーク高さ
        center : float
            ピーク中心位置
        width_g : float
            Gaussian成分の幅
        width_l : float
            Lorentzian成分の幅
    
        Returns:
        --------
        y : array
            Voigt曲線
        """
        # pseudo-Voigt: GaussianとLorentzianの線形結合
        eta = 0.5  # mixing parameter
        g = gaussian(x, amplitude, center, width_g)
        l = lorentzian(x, amplitude, center, width_l)
        return eta * l + (1 - eta) * g
    
    # 合成スペクトルの生成（ノイズあり）
    x_data = np.linspace(400, 700, 300)  # 波長（nm）
    true_params = {
        'amplitude': 1.0,
        'center': 550,
        'width': 30
    }
    
    # Gaussian + ノイズ
    y_data = gaussian(x_data, **true_params) + np.random.normal(0, 0.02, len(x_data))
    
    # フィッティング
    initial_guess = [0.8, 540, 25]
    
    popt_gauss, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    popt_lorentz, _ = curve_fit(lorentzian, x_data, y_data, p0=initial_guess)
    
    # 結果の可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # フィッティング結果
    ax1.scatter(x_data, y_data, s=10, alpha=0.5, color='gray', label='実験データ')
    ax1.plot(x_data, gaussian(x_data, *popt_gauss), 'r-', linewidth=2, label='Gaussianフィット')
    ax1.plot(x_data, lorentzian(x_data, *popt_lorentz), 'b--', linewidth=2, label='Lorentzianフィット')
    ax1.set_xlabel('波長 (nm)', fontsize=12)
    ax1.set_ylabel('吸光度 (a.u.)', fontsize=12)
    ax1.set_title('ピーク形状のフィッティング', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 線形関数の比較
    x_comparison = np.linspace(-100, 100, 500)
    y_gauss = gaussian(x_comparison, 1.0, 0, 20)
    y_lorentz = lorentzian(x_comparison, 1.0, 0, 20)
    y_voigt = voigt(x_comparison, 1.0, 0, 20, 10)
    
    ax2.plot(x_comparison, y_gauss, 'r-', linewidth=2, label='Gaussian')
    ax2.plot(x_comparison, y_lorentz, 'b-', linewidth=2, label='Lorentzian')
    ax2.plot(x_comparison, y_voigt, 'g-', linewidth=2, label='Voigt (pseudo)')
    ax2.set_xlabel('相対位置', fontsize=12)
    ax2.set_ylabel('規格化強度', fontsize=12)
    ax2.set_title('線形関数の比較', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 2)
    
    plt.tight_layout()
    plt.savefig('peak_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # フィッティングパラメータの出力
    print("Gaussianフィット結果:")
    print(f"  振幅: {popt_gauss[0]:.3f}")
    print(f"  中心: {popt_gauss[1]:.1f} nm")
    print(f"  FWHM: {2.355 * popt_gauss[2]:.1f} nm")
    
    print("\nLorentzianフィット結果:")
    print(f"  振幅: {popt_lorentz[0]:.3f}")
    print(f"  中心: {popt_lorentz[1]:.1f} nm")
    print(f"  FWHM: {2 * popt_lorentz[2]:.1f} nm")
    

### 4.3 ベースライン処理の重要性

実測スペクトルには、試料ホルダーの吸収、散乱、装置のドリフトなどによるベースラインが含まれます。正確な定量分析にはベースライン補正が不可欠です。

#### コード例7: ベースライン補正（多項式フィッティング）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    def polynomial_baseline(x, y, degree=2, exclude_peaks=True):
        """
        多項式フィッティングによるベースライン補正
    
        Parameters:
        -----------
        x : array
            横軸データ
        y : array
            縦軸データ（スペクトル）
        degree : int
            多項式の次数
        exclude_peaks : bool
            ピーク領域を除外してフィッティング
    
        Returns:
        --------
        baseline : array
            推定されたベースライン
        corrected : array
            ベースライン補正後のスペクトル
        """
        if exclude_peaks:
            # ピーク検出
            peaks, _ = find_peaks(y, prominence=0.1 * np.max(y))
    
            # ピーク周辺を除外したマスク
            mask = np.ones(len(y), dtype=bool)
            window = int(len(y) * 0.05)  # ピーク周辺5%を除外
            for peak in peaks:
                start = max(0, peak - window)
                end = min(len(y), peak + window)
                mask[start:end] = False
    
            # マスクされた領域で多項式フィッティング
            coeffs = np.polyfit(x[mask], y[mask], degree)
        else:
            coeffs = np.polyfit(x, y, degree)
    
        baseline = np.polyval(coeffs, x)
        corrected = y - baseline
    
        return baseline, corrected
    
    # 合成スペクトル（ベースライン付き）
    x = np.linspace(400, 700, 500)
    
    # 真のスペクトル（2つのピーク）
    true_spectrum = gaussian(x, 0.8, 500, 25) + gaussian(x, 0.5, 600, 20)
    
    # ベースライン（2次多項式）
    baseline_true = 0.1 + 0.0005 * (x - 550) + 0.000001 * (x - 550)**2
    
    # 観測スペクトル = 真のスペクトル + ベースライン + ノイズ
    observed_spectrum = true_spectrum + baseline_true + np.random.normal(0, 0.01, len(x))
    
    # ベースライン補正
    baseline_estimated, corrected_spectrum = polynomial_baseline(x, observed_spectrum, degree=2, exclude_peaks=True)
    
    # 可視化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 観測スペクトル
    ax1.plot(x, observed_spectrum, 'k-', linewidth=1.5, label='観測スペクトル')
    ax1.plot(x, baseline_true, 'r--', linewidth=2, label='真のベースライン')
    ax1.plot(x, baseline_estimated, 'b--', linewidth=2, label='推定ベースライン')
    ax1.set_xlabel('波長 (nm)', fontsize=12)
    ax1.set_ylabel('強度 (a.u.)', fontsize=12)
    ax1.set_title('ベースライン補正：観測スペクトル', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 補正後のスペクトル
    ax2.plot(x, corrected_spectrum, 'g-', linewidth=2, label='補正後スペクトル')
    ax2.plot(x, true_spectrum, 'r--', linewidth=2, alpha=0.7, label='真のスペクトル')
    ax2.set_xlabel('波長 (nm)', fontsize=12)
    ax2.set_ylabel('強度 (a.u.)', fontsize=12)
    ax2.set_title('ベースライン補正後', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 残差解析
    residual = corrected_spectrum - true_spectrum
    ax3.plot(x, residual, 'purple', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.fill_between(x, residual, alpha=0.3, color='purple')
    ax3.set_xlabel('波長 (nm)', fontsize=12)
    ax3.set_ylabel('残差 (a.u.)', fontsize=12)
    ax3.set_title('補正の残差', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_correction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計量
    print(f"ベースライン補正前のRMS誤差: {np.sqrt(np.mean((observed_spectrum - true_spectrum)**2)):.4f}")
    print(f"ベースライン補正後のRMS誤差: {np.sqrt(np.mean(residual**2)):.4f}")
    

## 5\. 多ピークスペクトルの解析

#### コード例8: 複数ピークの自動検出とフィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    
    def multi_gaussian(x, *params):
        """
        複数のGaussian関数の和
    
        Parameters:
        -----------
        x : array
            横軸
        params : tuple
            (amplitude1, center1, width1, amplitude2, center2, width2, ...)
    
        Returns:
        --------
        y : array
            複数Gaussianの和
        """
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amplitude = params[i]
            center = params[i+1]
            width = params[i+2]
            y += gaussian(x, amplitude, center, width)
        return y
    
    def auto_peak_fitting(x, y, min_prominence=0.1):
        """
        自動ピーク検出とマルチGaussianフィッティング
    
        Parameters:
        -----------
        x : array
            横軸データ
        y : array
            縦軸データ
        min_prominence : float
            ピーク検出の最小プロミネンス（最大値に対する相対値）
    
        Returns:
        --------
        params : array
            フィッティングパラメータ
        fitted : array
            フィッティング曲線
        individual_peaks : list of arrays
            個別のピーク成分
        """
        # ベースライン補正
        baseline, y_corrected = polynomial_baseline(x, y, degree=1, exclude_peaks=True)
    
        # ピーク検出
        peaks, properties = find_peaks(y_corrected, prominence=min_prominence * np.max(y_corrected))
    
        print(f"検出されたピーク数: {len(peaks)}")
    
        # 初期推定値
        initial_params = []
        for peak in peaks:
            amplitude = y_corrected[peak]
            center = x[peak]
            width = 20  # 初期幅の推定
            initial_params.extend([amplitude, center, width])
    
        # マルチGaussianフィッティング
        try:
            popt, _ = curve_fit(multi_gaussian, x, y_corrected, p0=initial_params, maxfev=10000)
            fitted = multi_gaussian(x, *popt)
    
            # 個別ピーク成分
            individual_peaks = []
            for i in range(0, len(popt), 3):
                peak_component = gaussian(x, popt[i], popt[i+1], popt[i+2])
                individual_peaks.append(peak_component)
    
            return popt, fitted, individual_peaks, baseline
    
        except RuntimeError:
            print("フィッティング失敗：収束しませんでした")
            return None, None, None, baseline
    
    # 複雑なスペクトルの生成（4つのピーク）
    x_data = np.linspace(400, 700, 600)
    true_components = [
        gaussian(x_data, 0.7, 450, 20),
        gaussian(x_data, 1.0, 520, 25),
        gaussian(x_data, 0.6, 580, 18),
        gaussian(x_data, 0.4, 650, 22)
    ]
    true_spectrum = sum(true_components)
    baseline = 0.05 + 0.0001 * x_data
    observed = true_spectrum + baseline + np.random.normal(0, 0.02, len(x_data))
    
    # 自動フィッティング
    params, fitted, individual, baseline_est = auto_peak_fitting(x_data, observed, min_prominence=0.15)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # フィッティング結果
    ax1.plot(x_data, observed, 'k.', markersize=2, alpha=0.5, label='観測データ')
    if fitted is not None:
        ax1.plot(x_data, fitted + baseline_est, 'r-', linewidth=2, label='フィッティング曲線')
        for i, peak in enumerate(individual):
            ax1.plot(x_data, peak + baseline_est, '--', linewidth=1.5, alpha=0.7, label=f'ピーク {i+1}')
    ax1.plot(x_data, baseline_est, 'g--', linewidth=2, label='ベースライン')
    ax1.set_xlabel('波長 (nm)', fontsize=12)
    ax1.set_ylabel('強度 (a.u.)', fontsize=12)
    ax1.set_title('多ピークスペクトルの自動フィッティング', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 分離されたピーク成分
    if individual is not None:
        for i, peak in enumerate(individual):
            ax2.plot(x_data, peak, linewidth=2, label=f'ピーク {i+1}')
            ax2.fill_between(x_data, peak, alpha=0.3)
    ax2.set_xlabel('波長 (nm)', fontsize=12)
    ax2.set_ylabel('強度 (a.u.)', fontsize=12)
    ax2.set_title('分離されたピーク成分', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_peak_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # フィッティング結果の詳細
    if params is not None:
        print("\n" + "=" * 60)
        print("フィッティング結果")
        print("=" * 60)
        for i in range(0, len(params), 3):
            peak_num = i // 3 + 1
            amplitude = params[i]
            center = params[i+1]
            width = params[i+2]
            fwhm = 2.355 * width
            area = amplitude * width * np.sqrt(2 * np.pi)
            print(f"\nピーク {peak_num}:")
            print(f"  中心位置: {center:.1f} nm")
            print(f"  振幅: {amplitude:.3f}")
            print(f"  FWHM: {fwhm:.1f} nm")
            print(f"  面積: {area:.3f}")
    

## 演習問題

**演習問題（クリックして展開）**

### Easy レベル（基本計算）

**問題1** : 波長500 nmの光の光子エネルギーをeV単位で計算してください。また、この光は可視光のどの色に対応しますか？

解答を見る

**解答** :
    
    
    h = 6.626e-34  # J·s
    c = 2.998e8    # m/s
    eV = 1.602e-19  # J
    
    wavelength = 500e-9  # m
    E = h * c / wavelength / eV
    print(f"光子エネルギー: {E:.2f} eV")
    # 出力: 光子エネルギー: 2.48 eV
    # 500 nmは緑色の光に対応
    

**問題2** : Beer-Lambert則において、モル吸光係数 $\varepsilon = 50000$ L mol-1 cm-1、光路長 $l = 1$ cm、吸光度 $A = 0.8$ のとき、試料の濃度（mol/L）を求めてください。

解答を見る

**解答** :
    
    
    epsilon = 50000  # L mol^-1 cm^-1
    l = 1.0  # cm
    A = 0.8
    
    c = A / (epsilon * l)
    print(f"濃度: {c:.2e} mol/L = {c * 1e6:.2f} μmol/L")
    # 出力: 濃度: 1.60e-05 mol/L = 16.00 μmol/L
    

**問題3** : 赤外分光において、波数 1650 cm-1 のピークが観測されました。この波数に対応する波長（μm）とエネルギー（eV）を計算してください。

解答を見る

**解答** :
    
    
    wavenumber = 1650  # cm^-1
    
    # 波数から波長
    wavelength_cm = 1 / wavenumber
    wavelength_um = wavelength_cm * 1e4
    print(f"波長: {wavelength_um:.2f} μm")
    
    # エネルギー
    h = 6.626e-34
    c = 2.998e8
    eV = 1.602e-19
    energy_J = h * c * wavenumber * 100  # cm^-1 to m^-1
    energy_eV = energy_J / eV
    print(f"エネルギー: {energy_eV:.4f} eV")
    # 出力: 波長: 6.06 μm, エネルギー: 0.2045 eV
    

### Medium レベル（実践的計算）

**問題4** : 下記のデータは溶液の濃度と吸光度の測定結果です。Beer-Lambert則を用いて検量線を作成し、未知試料（吸光度 0.65）の濃度を推定してください。

濃度 (μmol/L)| 10| 20| 30| 40| 50  
---|---|---|---|---|---  
吸光度| 0.18| 0.35| 0.53| 0.71| 0.88  
解答を見る

**解答** :
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    # データ
    concentrations = np.array([10, 20, 30, 40, 50])  # μmol/L
    absorbances = np.array([0.18, 0.35, 0.53, 0.71, 0.88])
    
    # 線形回帰
    slope, intercept, r_value, p_value, std_err = linregress(concentrations, absorbances)
    
    print(f"検量線: A = {slope:.4f} * C + {intercept:.4f}")
    print(f"相関係数 R²: {r_value**2:.4f}")
    
    # 未知試料の濃度推定
    unknown_A = 0.65
    unknown_C = (unknown_A - intercept) / slope
    print(f"未知試料の濃度: {unknown_C:.1f} μmol/L")
    
    # 可視化
    plt.figure(figsize=(8, 6))
    plt.scatter(concentrations, absorbances, s=100, color='#f093fb', label='測定データ')
    plt.plot(concentrations, slope * concentrations + intercept, 'r-', linewidth=2, label=f'検量線 (R²={r_value**2:.3f})')
    plt.scatter([unknown_C], [unknown_A], s=150, color='#f5576c', marker='*', label='未知試料', zorder=5)
    plt.xlabel('濃度 (μmol/L)', fontsize=12)
    plt.ylabel('吸光度', fontsize=12)
    plt.title('検量線の作成', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 出力: 未知試料の濃度: 36.4 μmol/L
    

**問題5** : 吸収スペクトルが以下のGaussian関数で近似できるとします。ピーク中心、FWHM、積分強度を計算してください。

$$I(\lambda) = 0.8 \exp\left(-\frac{(\lambda - 520)^2}{2 \times 25^2}\right)$$

解答を見る

**解答** :
    
    
    import numpy as np
    from scipy.integrate import quad
    
    # Gaussian関数のパラメータ
    amplitude = 0.8
    center = 520  # nm
    sigma = 25  # nm
    
    # FWHM計算
    fwhm = 2.355 * sigma
    print(f"ピーク中心: {center} nm")
    print(f"FWHM: {fwhm:.1f} nm")
    
    # 積分強度（解析解）
    integral_analytical = amplitude * sigma * np.sqrt(2 * np.pi)
    print(f"積分強度（解析解）: {integral_analytical:.2f}")
    
    # 数値積分による検証
    def gaussian_func(x):
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    integral_numerical, error = quad(gaussian_func, 0, 1000)
    print(f"積分強度（数値積分）: {integral_numerical:.2f}")
    
    # 出力: ピーク中心: 520 nm, FWHM: 58.9 nm, 積分強度: 50.13
    

**問題6** : Franck-Condon原理に基づき、基底状態（v=0）から励起状態の異なる振動準位（v'=0, 1, 2, 3）への遷移強度を計算してください。励起状態の平衡位置が基底状態から無次元座標で1.2だけずれているとします。

解答を見る

**解答** :
    
    
    import numpy as np
    
    def franck_condon_factor_harmonic(n_i, n_f, displacement):
        """
        Franck-Condon因子（調和振動子近似）
        Simplified formula for n_i = 0
        """
        from scipy.special import factorial
    
        if n_i == 0:
            S = displacement**2 / 2  # Huang-Rhys factor
            fc = np.exp(-S) * (S**n_f) / factorial(n_f)
        return fc
    
    displacement = 1.2
    vibrational_levels = [0, 1, 2, 3]
    
    print("Franck-Condon因子（v=0 → v'遷移）:")
    print("=" * 40)
    for v_f in vibrational_levels:
        fc = franck_condon_factor_harmonic(0, v_f, displacement)
        print(f"v=0 → v'={v_f}: {fc:.4f}")
    
    # 正規化された相対強度
    fc_values = [franck_condon_factor_harmonic(0, v, displacement) for v in vibrational_levels]
    fc_normalized = np.array(fc_values) / np.max(fc_values)
    print("\n相対強度（最大値で正規化）:")
    for v, intensity in zip(vibrational_levels, fc_normalized):
        print(f"v'={v}: {intensity:.2f}")
    

### Hard レベル（高度な解析）

**問題7** : 2つのGaussianピーク（中心: 500 nm, 550 nm、幅: 20 nm、振幅比 2:1）からなる合成スペクトルを生成し、ノイズ（標準偏差0.05）を付加してください。その後、scipy.optimize.curve_fitを用いて2成分フィッティングを行い、元のパラメータを復元してください。

解答を見る

**解答** :
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def two_gaussian(x, A1, c1, w1, A2, c2, w2):
        """2つのGaussianの和"""
        g1 = A1 * np.exp(-(x - c1)**2 / (2 * w1**2))
        g2 = A2 * np.exp(-(x - c2)**2 / (2 * w2**2))
        return g1 + g2
    
    # 真のパラメータ
    true_params = [1.0, 500, 20, 0.5, 550, 20]
    
    # 合成スペクトル
    x = np.linspace(400, 650, 300)
    y_true = two_gaussian(x, *true_params)
    y_noisy = y_true + np.random.normal(0, 0.05, len(x))
    
    # フィッティング（初期推定値）
    initial_guess = [0.8, 495, 18, 0.4, 545, 22]
    popt, pcov = curve_fit(two_gaussian, x, y_noisy, p0=initial_guess)
    
    # 結果の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_noisy, 'k.', markersize=3, alpha=0.5, label='ノイズ付きデータ')
    plt.plot(x, y_true, 'g--', linewidth=2, label='真のスペクトル')
    plt.plot(x, two_gaussian(x, *popt), 'r-', linewidth=2, label='フィッティング結果')
    
    # 個別成分
    plt.plot(x, popt[0] * np.exp(-(x - popt[1])**2 / (2 * popt[2]**2)), 'b--', alpha=0.7, label='ピーク1')
    plt.plot(x, popt[3] * np.exp(-(x - popt[4])**2 / (2 * popt[5]**2)), 'm--', alpha=0.7, label='ピーク2')
    
    plt.xlabel('波長 (nm)', fontsize=12)
    plt.ylabel('強度 (a.u.)', fontsize=12)
    plt.title('2成分Gaussianフィッティング', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # パラメータ比較
    print("パラメータ復元結果:")
    print("=" * 60)
    labels = ['振幅1', '中心1', '幅1', '振幅2', '中心2', '幅2']
    for label, true, fitted in zip(labels, true_params, popt):
        error = abs(fitted - true) / true * 100
        print(f"{label:8s}: 真値={true:6.2f}, フィット={fitted:6.2f}, 誤差={error:5.2f}%")
    

**問題8** : 選択則に基づき、以下の遷移が許容か禁制かを判定してください。

  * (a) 水素原子の1s → 2p遷移
  * (b) 水素原子の1s → 2s遷移
  * (c) ベンゼンの π → π* 遷移（D6h 対称性）
  * (d) オクタヘドラル錯体のd-d遷移（Oh 対称性）

解答を見る

**解答** :

(a) **許容** : $\Delta l = +1$ (s → p)、電気双極子遷移の選択則を満たす。

(b) **禁制** : $\Delta l = 0$ (s → s)、電気双極子遷移では $\Delta l = \pm 1$ が必要。

(c) **許容** : π軌道（πu）から π*軌道（πg*）への遷移はLaporte則（g ↔ u）を満たす。

(d) **禁制** （Laporte禁制）: 両方ともd軌道（g対称性）なので g ↔ g 遷移。ただし、振動による対称性の崩れや配位子の効果で実際には弱い吸収が観測されることがある。

**問題9** : 実測の吸収スペクトルデータが与えられた場合、以下の手順で解析してください：

  1. ベースライン補正（2次多項式フィッティング）
  2. ピーク自動検出
  3. 各ピークのGaussianフィッティング
  4. ピーク中心、FWHM、積分強度の算出

解答を見る（完全な解析コード）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    from scipy.integrate import simps
    
    # サンプルデータ生成（実測を模擬）
    x_data = np.linspace(400, 700, 500)
    true_spectrum = (gaussian(x_data, 0.6, 480, 22) +
                     gaussian(x_data, 0.9, 540, 28) +
                     gaussian(x_data, 0.5, 620, 25))
    baseline = 0.08 + 0.0002 * x_data
    observed = true_spectrum + baseline + np.random.normal(0, 0.03, len(x_data))
    
    # ステップ1: ベースライン補正
    baseline_fit, _ = polynomial_baseline(x_data, observed, degree=2, exclude_peaks=True)
    corrected = observed - baseline_fit
    
    # ステップ2: ピーク検出
    peaks, properties = find_peaks(corrected, prominence=0.15)
    print(f"検出されたピーク数: {len(peaks)}")
    
    # ステップ3: 各ピークのフィッティング
    results = []
    for i, peak_idx in enumerate(peaks):
        # ピーク周辺のデータを抽出
        window = 80
        start = max(0, peak_idx - window)
        end = min(len(x_data), peak_idx + window)
    
        x_region = x_data[start:end]
        y_region = corrected[start:end]
    
        # 初期推定
        p0 = [corrected[peak_idx], x_data[peak_idx], 20]
    
        try:
            popt, _ = curve_fit(gaussian, x_region, y_region, p0=p0)
    
            # パラメータ抽出
            amplitude, center, sigma = popt
            fwhm = 2.355 * sigma
    
            # 積分強度（Gaussianの解析解）
            area = amplitude * sigma * np.sqrt(2 * np.pi)
    
            results.append({
                'peak_number': i + 1,
                'center': center,
                'amplitude': amplitude,
                'fwhm': fwhm,
                'area': area,
                'params': popt
            })
    
        except RuntimeError:
            print(f"ピーク {i+1} のフィッティング失敗")
    
    # ステップ4: 結果の可視化と出力
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 元のスペクトルとベースライン
    ax1.plot(x_data, observed, 'k-', linewidth=1, alpha=0.5, label='観測スペクトル')
    ax1.plot(x_data, baseline_fit, 'g--', linewidth=2, label='ベースライン')
    ax1.set_xlabel('波長 (nm)', fontsize=12)
    ax1.set_ylabel('強度 (a.u.)', fontsize=12)
    ax1.set_title('ステップ1: ベースライン補正', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 補正後とフィッティング結果
    ax2.plot(x_data, corrected, 'k-', linewidth=1, alpha=0.5, label='補正後スペクトル')
    
    colors = ['#f093fb', '#f5576c', '#4ecdc4']
    for i, result in enumerate(results):
        fitted_peak = gaussian(x_data, *result['params'])
        ax2.plot(x_data, fitted_peak, '--', linewidth=2, color=colors[i % len(colors)],
                 label=f"ピーク{result['peak_number']} ({result['center']:.1f} nm)")
        ax2.fill_between(x_data, fitted_peak, alpha=0.2, color=colors[i % len(colors)])
    
    ax2.set_xlabel('波長 (nm)', fontsize=12)
    ax2.set_ylabel('強度 (a.u.)', fontsize=12)
    ax2.set_title('ステップ2-3: ピーク検出とフィッティング', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_spectrum_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 解析結果の表
    print("\n" + "=" * 80)
    print("スペクトル解析結果")
    print("=" * 80)
    print(f"{'ピーク':<8} {'中心 (nm)':<12} {'振幅':<10} {'FWHM (nm)':<12} {'積分強度':<12}")
    print("-" * 80)
    for result in results:
        print(f"{result['peak_number']:<8} {result['center']:<12.1f} {result['amplitude']:<10.3f} "
              f"{result['fwhm']:<12.1f} {result['area']:<12.2f}")
    

**問題10** : Voigt線形（GaussianとLorentzianの畳み込み）を実装し、純粋なGaussian、純粋なLorentzian、Voigt線形の3つをプロットして比較してください。Voigt線形は以下の積分で定義されます：

$$V(x; \sigma, \gamma) = \int_{-\infty}^{\infty} G(x'; \sigma) L(x - x'; \gamma) \, dx'$$

解答を見る

**解答** :
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import wofz
    
    def voigt_profile(x, amplitude, center, sigma, gamma):
        """
        Voigt線形（Faddeeva関数を使用）
    
        Parameters:
        -----------
        x : array
            横軸
        amplitude : float
            ピーク高さ
        center : float
            中心位置
        sigma : float
            Gaussian成分の幅
        gamma : float
            Lorentzian成分の幅（HWHM）
    
        Returns:
        --------
        voigt : array
            Voigt線形
        """
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        voigt = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        return voigt
    
    # パラメータ設定
    x = np.linspace(-100, 100, 1000)
    amplitude = 1.0
    center = 0.0
    sigma = 15.0  # Gaussian幅
    gamma = 10.0  # Lorentzian HWHM
    
    # 3つの線形の計算
    y_gauss = gaussian(x, amplitude, center, sigma)
    y_lorentz = lorentzian(x, amplitude, center, gamma)
    y_voigt = voigt_profile(x, amplitude, center, sigma, gamma)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 線形スケール
    ax1.plot(x, y_gauss, 'b-', linewidth=2, label=f'Gaussian (σ={sigma})')
    ax1.plot(x, y_lorentz, 'r-', linewidth=2, label=f'Lorentzian (γ={gamma})')
    ax1.plot(x, y_voigt, 'g-', linewidth=2, label='Voigt (畳み込み)')
    ax1.set_xlabel('相対位置', fontsize=12)
    ax1.set_ylabel('規格化強度', fontsize=12)
    ax1.set_title('Voigt線形の比較（線形スケール）', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 対数スケール（裾の違いを強調）
    ax2.semilogy(x, y_gauss, 'b-', linewidth=2, label='Gaussian')
    ax2.semilogy(x, y_lorentz, 'r-', linewidth=2, label='Lorentzian')
    ax2.semilogy(x, y_voigt, 'g-', linewidth=2, label='Voigt')
    ax2.set_xlabel('相対位置', fontsize=12)
    ax2.set_ylabel('規格化強度（対数）', fontsize=12)
    ax2.set_title('Voigt線形の比較（対数スケール）', fontsize=14, fontweight='bold')
    ax2.set_ylim(1e-6, 2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('voigt_profile.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 裾の違いを定量化
    print("x = 50 での強度比（中心からの距離）:")
    idx = np.argmin(np.abs(x - 50))
    print(f"  Gaussian: {y_gauss[idx]:.2e}")
    print(f"  Lorentzian: {y_lorentz[idx]:.2e}")
    print(f"  Voigt: {y_voigt[idx]:.2e}")
    print(f"\nLorentzianはGaussianの {y_lorentz[idx] / y_gauss[idx]:.1f} 倍の裾を持つ")
    

## 学習目標の確認

この章で学んだ内容を振り返り、以下の項目を確認してください。

### 基本理解

  * ✅ 光子エネルギーと波長の関係（Planck関係式）を説明できる
  * ✅ 吸収・発光・散乱の違いと物理的機構を理解している
  * ✅ Beer-Lambert則の物理的意味と適用条件を説明できる
  * ✅ 選択則の概念と主要な選択則（$\Delta l = \pm 1$、Laporte則など）を理解している

### 実践スキル

  * ✅ 波長・波数・エネルギーを相互変換できる
  * ✅ Beer-Lambert則を用いて濃度計算や検量線作成ができる
  * ✅ スペクトルのピークフィッティング（Gaussian/Lorentzian）ができる
  * ✅ ベースライン補正を適切に実施できる

### 応用力

  * ✅ Fermiの黄金則から遷移確率を計算できる
  * ✅ Franck-Condon原理に基づいて振動構造を解釈できる
  * ✅ 複数ピークのスペクトルを成分分離・定量できる

## 参考文献

  1. Atkins, P., de Paula, J. (2010). _Physical Chemistry_ (9th ed.). Oxford University Press, pp. 465-468 (Beer-Lambert law), pp. 485-490 (transition dipole moments), pp. 501-506 (selection rules). - 分光学の量子力学的基礎と遷移モーメントの詳細な解説
  2. Banwell, C. N., McCash, E. M. (1994). _Fundamentals of Molecular Spectroscopy_ (4th ed.). McGraw-Hill, pp. 8-15 (electromagnetic radiation), pp. 28-35 (Beer-Lambert law applications). - 分光分析の基本原理とBeer-Lambert則の応用
  3. Hollas, J. M. (2004). _Modern Spectroscopy_ (4th ed.). Wiley, pp. 15-23 (selection rules), pp. 45-52 (Franck-Condon principle), pp. 78-85 (transition probabilities). - 選択則とFranck-Condon原理の包括的解説
  4. Beer, A. (1852). Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten. _Annalen der Physik und Chemie_ , 86, 78-88. DOI: 10.1002/andp.18521620505 - Beer-Lambert則のオリジナル論文（歴史的文献）
  5. Shirley, D. A. (1972). High-resolution X-ray photoemission spectrum of the valence bands of gold. _Physical Review B_ , 5(12), 4709-4714. DOI: 10.1103/PhysRevB.5.4709 - Shirleyバックグラウンド補正アルゴリズムの原著論文
  6. NumPy 1.24 and SciPy 1.11 Documentation. _Signal Processing (scipy.signal) and Optimization (scipy.optimize)_. Available at: https://docs.scipy.org/doc/scipy/reference/signal.html - Pythonによるスペクトルデータ解析の実践的手法
  7. Turrell, G., Corset, J. (Eds.). (1996). _Raman Microscopy: Developments and Applications_. Academic Press, pp. 25-34 (classical scattering theory), pp. 58-67 (selection rules for Raman). - 散乱分光法の理論と選択則の詳細
  8. Levine, I. N. (2013). _Quantum Chemistry_ (7th ed.). Pearson, pp. 580-595 (time-dependent perturbation theory), pp. 620-635 (transition dipole moments). - 遷移双極子モーメントと量子力学的計算手法

## 次のステップ

第1章では、分光分析の基礎となる光と物質の相互作用、Beer-Lambert則、量子力学的原理、選択則を学びました。Pythonによる実践的なデータ処理（単位変換、ピークフィッティング、ベースライン補正）のスキルも習得しました。

**第2章** では、これらの基礎知識を応用して、赤外・ラマン分光法の原理と実践を学びます。振動分光による官能基同定、結晶性評価、IRとRamanの相補的関係、群論による選択則の詳細など、材料科学で頻繁に使用される振動分光の全てをカバーします。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
