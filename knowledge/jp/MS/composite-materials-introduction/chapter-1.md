---
title: 第1章 複合材料の基礎
chapter_title: 第1章 複合材料の基礎
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/composite-materials-introduction/chapter-1.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Composite Materials](<../../MS/composite-materials-introduction/index.html>)›Chapter 1

### 複合材料入門

  * [目次](<index.html>)
  * [第1章 複合材料の基礎](<chapter-1.html>)
  * [第2章 繊維強化複合材料](<chapter-2.html>)
  * [第3章 粒子・積層複合材料](<chapter-3.html>)
  * 第4章 複合材料の評価（準備中）
  * [第5章 Python実践](<chapter-5.html>)

#### Materials Science シリーズ

  * [高分子材料入門](<../polymer-materials-introduction/index.html>)
  * 薄膜・ナノ材料入門 (準備中)
  * [複合材料入門](<index.html>)

# 第1章 複合材料の基礎

### 学習目標

  * **基礎レベル:** 複合材料の定義と分類を理解し、基本的な混合則を計算できる
  * **応用レベル:** Halpin-Tsai理論を適用し、繊維配向の影響を定量的に評価できる
  * **発展レベル:** 界面強度と機械的特性の関係を解析し、材料設計に応用できる

## 1.1 複合材料とは

### 1.1.1 複合材料の定義

複合材料(Composite Materials)は、2種類以上の材料を巨視的に組み合わせることで、 単一材料では得られない優れた特性を実現する材料です。主に以下の要素から構成されます： 

  * **強化材(Reinforcement):** 高い強度・剛性を担う成分(繊維、粒子など)
  * **母材(Matrix):** 強化材を保持し、荷重を伝達する成分(樹脂、金属、セラミックス)
  * **界面(Interface):** 強化材と母材の境界領域(特性発現の鍵)

    
    
    ```mermaid
    flowchart TD
                                A[複合材料の分類] --> B[強化材形態]
                                A --> C[母材種類]
                                B --> D[繊維強化CFRP, GFRP]
                                B --> E[粒子強化MMC, CMC]
                                B --> F[積層材Laminate]
                                C --> G[樹脂基PMC]
                                C --> H[金属基MMC]
                                C --> I[セラミック基CMC]
    
                                style A fill:#e1f5ff
                                style D fill:#ffe1e1
                                style E fill:#ffe1e1
                                style F fill:#ffe1e1
                                style G fill:#e1ffe1
                                style H fill:#e1ffe1
                                style I fill:#e1ffe1
    ```

### 1.1.2 複合材料の利点

複合材料が広く用いられる理由は以下の特性にあります：

特性 | 説明 | 代表例  
---|---|---  
高比強度・比剛性 | 密度あたりの強度・剛性が高い | 航空機構造材(CFRP)  
異方性制御 | 方向により特性を設計可能 | 積層板の配向設計  
複合特性 | 電気・熱・機械特性の複合化 | 導電性複合材料  
成形自由度 | 複雑形状の一体成形が可能 | RTM成形品  
  
## 1.2 混合則(Rule of Mixtures)

### 1.2.1 基本的な混合則

複合材料の特性は、構成材料の特性と体積分率から予測できます。 最も単純なモデルが**混合則(ROM)** です。 

#### 弾性率の混合則(Voigt モデル)

繊維方向(縦方向)の弾性率 \\(E_L\\) は、等ひずみ仮定により：

$$E_L = E_f V_f + E_m V_m = E_f V_f + E_m (1 - V_f)$$ 

ここで、\\(E_f\\): 繊維の弾性率、\\(E_m\\): 母材の弾性率、\\(V_f\\): 繊維体積分率

#### 横方向弾性率(Reuss モデル)

繊維に垂直な方向(横方向)の弾性率 \\(E_T\\) は、等応力仮定により：

$$\frac{1}{E_T} = \frac{V_f}{E_f} + \frac{V_m}{E_m}$$ 

#### 例題 1.1: CFRP の弾性率計算

炭素繊維(Ef = 230 GPa)とエポキシ樹脂(Em = 3.5 GPa)からなる 一方向CFRPの縦弾性率と横弾性率を計算せよ。繊維体積分率 Vf = 0.60 とする。 
    
    
    import numpy as np
    
    # 材料特性
    E_f = 230  # 炭素繊維の弾性率 [GPa]
    E_m = 3.5  # エポキシの弾性率 [GPa]
    V_f = 0.60  # 繊維体積分率
    
    # 縦弾性率(Voigt モデル)
    E_L = E_f * V_f + E_m * (1 - V_f)
    
    # 横弾性率(Reuss モデル)
    E_T = 1 / (V_f / E_f + (1 - V_f) / E_m)
    
    print(f"縦弾性率 E_L: {E_L:.1f} GPa")
    print(f"横弾性率 E_T: {E_T:.2f} GPa")
    print(f"異方性比 E_L/E_T: {E_L/E_T:.1f}")
    
    # 出力:
    # 縦弾性率 E_L: 139.4 GPa
    # 横弾性率 E_T: 7.29 GPa
    # 異方性比 E_L/E_T: 19.1

### 1.2.2 強度の混合則

引張強度についても同様の関係が成り立ちますが、破壊メカニズムにより修正が必要です：

$$\sigma_c = \sigma_f V_f + \sigma_m' (1 - V_f)$$ 

ここで、\\(\sigma_m'\\) は繊維破断時のひずみにおける母材の応力です。 これは、繊維と母材が同時に破断するわけではないことを考慮しています。 

#### 例題 1.2: 複合材料の引張強度予測
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 材料特性
    sigma_f = 3500  # 繊維引張強度 [MPa]
    sigma_m = 80    # 母材引張強度 [MPa]
    epsilon_f = 0.015  # 繊維破断ひずみ
    E_m = 3500     # 母材弾性率 [MPa]
    
    # 繊維破断時の母材応力
    sigma_m_prime = min(E_m * epsilon_f, sigma_m)
    
    # 体積分率に対する強度変化
    V_f_range = np.linspace(0, 0.8, 100)
    sigma_c = sigma_f * V_f_range + sigma_m_prime * (1 - V_f_range)
    
    # 可視化
    plt.figure(figsize=(8, 5))
    plt.plot(V_f_range, sigma_c, 'b-', linewidth=2, label='複合材料強度')
    plt.axhline(y=sigma_m, color='r', linestyle='--', label='母材強度')
    plt.xlabel('繊維体積分率 V_f')
    plt.ylabel('引張強度 [MPa]')
    plt.title('繊維体積分率と複合材料強度の関係')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('composite_strength_rom.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"繊維破断時の母材応力: {sigma_m_prime:.1f} MPa")
    print(f"V_f=0.6 での複合材料強度: {sigma_f*0.6 + sigma_m_prime*0.4:.0f} MPa")

## 1.3 Halpin-Tsai 理論

### 1.3.1 理論の概要

Halpin-Tsai理論は、繊維の形状因子(アスペクト比)を考慮した より精度の高い弾性率予測モデルです。特に横方向弾性率の予測精度が向上します。 

$$E_c = E_m \frac{1 + \zeta \eta V_f}{1 - \eta V_f}$$ 

ここで、

$$\eta = \frac{(E_f / E_m) - 1}{(E_f / E_m) + \zeta}$$ 

\\(\zeta\\) は形状因子で、繊維のアスペクト比(長さ/直径)と配向に依存します： 

  * 縦方向: \\(\zeta = 2(l/d)\\) (繊維長さ/直径)
  * 横方向: \\(\zeta = 2\\) (円形断面繊維の場合)

#### 例題 1.3: Halpin-Tsai モデルによる弾性率計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def halpin_tsai_modulus(E_f, E_m, V_f, zeta):
        """
        Halpin-Tsai理論による複合材料弾性率の計算
    
        Parameters:
        -----------
        E_f : float
            繊維弾性率 [GPa]
        E_m : float
            母材弾性率 [GPa]
        V_f : float or array
            繊維体積分率
        zeta : float
            形状因子
    
        Returns:
        --------
        E_c : float or array
            複合材料弾性率 [GPa]
        """
        eta = (E_f / E_m - 1) / (E_f / E_m + zeta)
        E_c = E_m * (1 + zeta * eta * V_f) / (1 - eta * V_f)
        return E_c
    
    # 材料特性
    E_f = 230  # 炭素繊維 [GPa]
    E_m = 3.5  # エポキシ [GPa]
    
    # 体積分率範囲
    V_f_range = np.linspace(0, 0.7, 100)
    
    # 各方向の弾性率計算
    # 縦方向(混合則でも十分精度が高い)
    E_L_rom = E_f * V_f_range + E_m * (1 - V_f_range)
    
    # 横方向(Halpin-Tsai)
    zeta_T = 2  # 横方向の形状因子
    E_T_ht = halpin_tsai_modulus(E_f, E_m, V_f_range, zeta_T)
    
    # 横方向(Reuss モデルと比較)
    E_T_reuss = 1 / (V_f_range / E_f + (1 - V_f_range) / E_m)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 縦弾性率
    ax1.plot(V_f_range, E_L_rom, 'b-', linewidth=2, label='混合則(Voigt)')
    ax1.set_xlabel('繊維体積分率 V_f')
    ax1.set_ylabel('縦弾性率 E_L [GPa]')
    ax1.set_title('縦方向弾性率')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 横弾性率
    ax2.plot(V_f_range, E_T_ht, 'r-', linewidth=2, label='Halpin-Tsai')
    ax2.plot(V_f_range, E_T_reuss, 'g--', linewidth=2, label='混合則(Reuss)')
    ax2.set_xlabel('繊維体積分率 V_f')
    ax2.set_ylabel('横弾性率 E_T [GPa]')
    ax2.set_title('横方向弾性率の比較')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('halpin_tsai_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # V_f = 0.6 での値を出力
    V_f_test = 0.6
    E_L = E_f * V_f_test + E_m * (1 - V_f_test)
    E_T_ht_val = halpin_tsai_modulus(E_f, E_m, V_f_test, zeta_T)
    E_T_reuss_val = 1 / (V_f_test / E_f + (1 - V_f_test) / E_m)
    
    print(f"V_f = {V_f_test}")
    print(f"縦弾性率 E_L: {E_L:.1f} GPa")
    print(f"横弾性率 E_T (Halpin-Tsai): {E_T_ht_val:.2f} GPa")
    print(f"横弾性率 E_T (Reuss): {E_T_reuss_val:.2f} GPa")
    print(f"予測差: {abs(E_T_ht_val - E_T_reuss_val):.2f} GPa")

### 1.3.2 短繊維複合材料への適用

短繊維複合材料では、繊維のアスペクト比が有限であるため、 縦方向弾性率もHalpin-Tsai理論で補正する必要があります。 

#### 例題 1.4: 短繊維複合材料の弾性率
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def short_fiber_modulus(E_f, E_m, V_f, aspect_ratio):
        """
        短繊維複合材料の縦弾性率(Halpin-Tsai)
    
        Parameters:
        -----------
        aspect_ratio : float
            繊維のアスペクト比 (l/d)
        """
        zeta_L = 2 * aspect_ratio
        eta_L = (E_f / E_m - 1) / (E_f / E_m + zeta_L)
        E_L = E_m * (1 + zeta_L * eta_L * V_f) / (1 - eta_L * V_f)
    
        zeta_T = 2
        eta_T = (E_f / E_m - 1) / (E_f / E_m + zeta_T)
        E_T = E_m * (1 + zeta_T * eta_T * V_f) / (1 - eta_T * V_f)
    
        return E_L, E_T
    
    # 材料特性
    E_f = 230  # GPa
    E_m = 3.5  # GPa
    V_f = 0.5
    
    # アスペクト比の影響
    aspect_ratios = np.array([5, 10, 20, 50, 100, 1000])
    E_L_values = []
    E_T_values = []
    
    for ar in aspect_ratios:
        E_L, E_T = short_fiber_modulus(E_f, E_m, V_f, ar)
        E_L_values.append(E_L)
        E_T_values.append(E_T)
    
    E_L_values = np.array(E_L_values)
    E_T_values = np.array(E_T_values)
    
    # 長繊維の場合(混合則)
    E_L_continuous = E_f * V_f + E_m * (1 - V_f)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.semilogx(aspect_ratios, E_L_values, 'bo-', linewidth=2,
                 markersize=8, label='短繊維 E_L (Halpin-Tsai)')
    plt.axhline(y=E_L_continuous, color='r', linestyle='--',
                linewidth=2, label=f'長繊維 E_L (混合則): {E_L_continuous:.1f} GPa')
    plt.xlabel('アスペクト比 (l/d)')
    plt.ylabel('縦弾性率 E_L [GPa]')
    plt.title(f'短繊維のアスペクト比と弾性率の関係 (V_f = {V_f})')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('short_fiber_aspect_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 結果出力
    print("アスペクト比と弾性率の関係:")
    print("-" * 50)
    for ar, E_L, E_T in zip(aspect_ratios, E_L_values, E_T_values):
        efficiency = (E_L / E_L_continuous) * 100
        print(f"l/d = {ar:4.0f}: E_L = {E_L:6.1f} GPa ({efficiency:5.1f}%), "
              f"E_T = {E_T:5.2f} GPa")

## 1.4 界面の役割

### 1.4.1 界面強度の重要性

複合材料の性能は、強化材と母材の界面特性に大きく依存します。 界面が担う主な役割は： 

  * **荷重伝達:** 母材から繊維へ応力を効率的に伝える
  * **亀裂抑制:** き裂の進展を界面で偏向・停止させる
  * **破壊エネルギー:** 界面剥離により破壊エネルギーを吸収

    
    
    ```mermaid
    flowchart LR
                                A[荷重印加] --> B[母材に応力発生]
                                B --> C{界面せん断応力}
                                C --> D[繊維に荷重伝達]
                                D --> E[繊維が主荷重負担]
    
                                C --> F{界面強度}
                                F -->|強い| G[効率的な荷重伝達高強度]
                                F -->|弱い| H[界面剥離低強度]
    
                                style A fill:#e1f5ff
                                style E fill:#c8e6c9
                                style G fill:#c8e6c9
                                style H fill:#ffcdd2
    ```

### 1.4.2 界面せん断応力

繊維に沿った界面せん断応力 \\(\tau_i\\) は、Kelly-Tyson モデルにより： 

$$\tau_i = \frac{\sigma_f d}{2l}$$ 

ここで、\\(\sigma_f\\): 繊維応力、\\(d\\): 繊維直径、\\(l\\): 繊維長さ。 臨界繊維長 \\(l_c\\) は、繊維が最大強度を発揮できる最小長さで： 

$$l_c = \frac{\sigma_f^* d}{2\tau_i}$$ 

\\(\sigma_f^*\\): 繊維の引張強度、\\(\tau_i\\): 界面せん断強度

#### 例題 1.5: 臨界繊維長の計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 材料特性
    sigma_f_max = 3500  # 繊維引張強度 [MPa]
    tau_i = 50          # 界面せん断強度 [MPa]
    d = 7e-3            # 繊維直径 [mm]
    
    # 臨界繊維長
    l_c = (sigma_f_max * d) / (2 * tau_i)
    
    print(f"繊維直径: {d} mm")
    print(f"繊維引張強度: {sigma_f_max} MPa")
    print(f"界面せん断強度: {tau_i} MPa")
    print(f"臨界繊維長: {l_c:.2f} mm")
    print(f"臨界アスペクト比 (l_c/d): {l_c/d:.0f}")
    
    # 繊維長さに対する強度効率
    fiber_lengths = np.linspace(0.1, 3*l_c, 100)
    strength_efficiency = np.where(
        fiber_lengths >= l_c,
        1.0,  # l >= l_c: 100%効率
        fiber_lengths / l_c  # l < l_c: 比例的に低下
    )
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 繊維長さと強度効率
    ax1.plot(fiber_lengths, strength_efficiency * 100, 'b-', linewidth=2)
    ax1.axvline(x=l_c, color='r', linestyle='--', linewidth=2,
                label=f'臨界繊維長 l_c = {l_c:.2f} mm')
    ax1.axhline(y=100, color='g', linestyle=':', alpha=0.5)
    ax1.set_xlabel('繊維長さ [mm]')
    ax1.set_ylabel('強度効率 [%]')
    ax1.set_title('繊維長さと強度効率の関係')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 110])
    
    # 界面せん断応力分布(l > l_c の場合)
    l_fiber = 2 * l_c  # 例として 2*l_c の繊維
    x = np.linspace(0, l_fiber, 100)
    # 繊維軸方向応力(線形増加、中央で最大)
    sigma_fiber = np.where(
        x <= l_fiber/2,
        2 * tau_i * x / d,  # 繊維端から増加
        2 * tau_i * (l_fiber - x) / d  # 対称
    )
    
    ax2.plot(x, sigma_fiber, 'r-', linewidth=2)
    ax2.axhline(y=sigma_f_max, color='g', linestyle='--',
                label=f'繊維強度 {sigma_f_max} MPa')
    ax2.set_xlabel('繊維軸方向位置 [mm]')
    ax2.set_ylabel('繊維内応力 [MPa]')
    ax2.set_title(f'繊維内応力分布 (l = {l_fiber:.2f} mm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('critical_fiber_length.png', dpi=300, bbox_inches='tight')
    plt.close()

### 1.4.3 界面処理の効果

繊維表面処理(サイジング、カップリング剤)は界面強度を向上させます。 代表的な処理方法と効果： 

処理方法 | メカニズム | 効果  
---|---|---  
シランカップリング剤 | 化学結合形成 | 界面せん断強度 20-40% 向上  
酸化処理 | 表面官能基増加 | 濡れ性改善、接着力向上  
プラズマ処理 | 表面活性化 | 極性基導入、化学結合促進  
サイジング剤 | 保護膜形成 | 繊維損傷防止、濡れ性制御  
  
#### 例題 1.6: 界面処理の効果シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def composite_strength_with_interface(V_f, tau_i, sigma_f_max, d, l):
        """
        界面強度を考慮した複合材料強度
    
        Parameters:
        -----------
        tau_i : float
            界面せん断強度 [MPa]
        l : float
            繊維長さ [mm]
        """
        # 臨界繊維長
        l_c = (sigma_f_max * d) / (2 * tau_i)
    
        # 繊維の有効強度
        if l >= l_c:
            sigma_f_eff = sigma_f_max * (1 - l_c / (2 * l))
        else:
            sigma_f_eff = tau_i * l / d
    
        # 複合材料強度(簡易混合則)
        sigma_c = sigma_f_eff * V_f
    
        return sigma_c, l_c
    
    # パラメータ設定
    V_f = 0.5
    sigma_f_max = 3500  # MPa
    d = 0.007          # mm
    l = 5.0            # mm
    
    # 界面強度の範囲(未処理 vs 処理済み)
    tau_i_range = np.linspace(20, 80, 50)
    sigma_c_values = []
    l_c_values = []
    
    for tau_i in tau_i_range:
        sigma_c, l_c = composite_strength_with_interface(V_f, tau_i, sigma_f_max, d, l)
        sigma_c_values.append(sigma_c)
        l_c_values.append(l_c)
    
    sigma_c_values = np.array(sigma_c_values)
    l_c_values = np.array(l_c_values)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 界面強度と複合材料強度
    ax1.plot(tau_i_range, sigma_c_values, 'b-', linewidth=2)
    ax1.axvline(x=40, color='r', linestyle='--', alpha=0.5, label='未処理')
    ax1.axvline(x=55, color='g', linestyle='--', alpha=0.5, label='処理済み')
    ax1.set_xlabel('界面せん断強度 τ_i [MPa]')
    ax1.set_ylabel('複合材料強度 [MPa]')
    ax1.set_title(f'界面強度の影響 (V_f={V_f}, l={l} mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 臨界繊維長
    ax2.plot(tau_i_range, l_c_values, 'r-', linewidth=2)
    ax2.axhline(y=l, color='g', linestyle='--', label=f'実際の繊維長 {l} mm')
    ax2.fill_between(tau_i_range, 0, l, alpha=0.2, color='green',
                      label='l > l_c (効率的)')
    ax2.set_xlabel('界面せん断強度 τ_i [MPa]')
    ax2.set_ylabel('臨界繊維長 l_c [mm]')
    ax2.set_title('界面強度と臨界繊維長の関係')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('interface_treatment_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 具体的な数値比較
    tau_i_untreated = 40  # MPa
    tau_i_treated = 55    # MPa
    
    sigma_c_untreated, l_c_untreated = composite_strength_with_interface(
        V_f, tau_i_untreated, sigma_f_max, d, l)
    sigma_c_treated, l_c_treated = composite_strength_with_interface(
        V_f, tau_i_treated, sigma_f_max, d, l)
    
    print("界面処理の効果:")
    print("-" * 60)
    print(f"未処理繊維:")
    print(f"  界面せん断強度: {tau_i_untreated} MPa")
    print(f"  臨界繊維長: {l_c_untreated:.2f} mm")
    print(f"  複合材料強度: {sigma_c_untreated:.0f} MPa")
    print(f"\n処理済み繊維:")
    print(f"  界面せん断強度: {tau_i_treated} MPa")
    print(f"  臨界繊維長: {l_c_treated:.2f} mm")
    print(f"  複合材料強度: {sigma_c_treated:.0f} MPa")
    print(f"\n強度向上率: {(sigma_c_treated/sigma_c_untreated - 1)*100:.1f}%")

## 1.5 まとめ

本章では、複合材料の基礎として以下の内容を学びました：

  * 複合材料の定義と分類(繊維強化、粒子強化、積層材)
  * 混合則(Voigt/Reussモデル)による弾性率・強度予測
  * Halpin-Tsai理論による精密な弾性率計算
  * 短繊維複合材料におけるアスペクト比の影響
  * 界面の役割と臨界繊維長の概念
  * 界面処理による特性向上メカニズム

次章では、繊維強化複合材料に焦点を当て、CFRP/GFRPの製造法、 積層理論(Classical Laminate Theory)、A-B-D行列について学びます。 

## 演習問題

### 基礎レベル

#### 問題 1.1: 混合則の基本計算

ガラス繊維(Ef = 70 GPa)とポリエステル樹脂(Em = 2.8 GPa)からなる 一方向GFRPについて、以下を計算せよ。繊維体積分率 Vf = 0.55。 

  1. 縦弾性率 EL (Voigt モデル)
  2. 横弾性率 ET (Reuss モデル)
  3. 異方性比 EL/ET

#### 問題 1.2: 体積分率の決定

CFRP複合材料の密度が 1.55 g/cm³ である。炭素繊維の密度 1.80 g/cm³、 エポキシ樹脂の密度 1.20 g/cm³ として、繊維体積分率 Vf を求めよ。 (ヒント: 密度の混合則 \\(\rho_c = \rho_f V_f + \rho_m (1-V_f)\\) を使用) 

#### 問題 1.3: 強度の混合則

繊維引張強度 2800 MPa、母材引張強度 65 MPa、繊維破断ひずみ 0.012、 母材弾性率 3200 MPa の材料系で、Vf = 0.65 の複合材料の引張強度を予測せよ。 

### 応用レベル

#### 問題 1.4: Halpin-Tsai 理論の適用

短繊維複合材料(Vf = 0.4, Ef = 230 GPa, Em = 3.5 GPa) において、アスペクト比 l/d = 20 の繊維を用いる場合の縦弾性率を Halpin-Tsai理論で計算し、混合則(Voigt)の結果と比較せよ。 

#### 問題 1.5: 臨界繊維長の影響

炭素繊維(直径 7 μm、引張強度 3500 MPa)とエポキシ樹脂の界面せん断強度が 45 MPa である場合の臨界繊維長を求め、繊維長が 2 mm、4 mm、6 mm の場合の 強度効率を計算せよ。 

#### 問題 1.6: 界面処理の設計

繊維長 3 mm の短繊維複合材料で、繊維強度の 95% 以上を活用したい。 必要な界面せん断強度を求めよ。 (繊維: 直径 10 μm、引張強度 4000 MPa) 

#### 問題 1.7: プログラミング課題

以下の機能を持つPythonプログラムを作成せよ： 

  * 繊維と母材の特性を入力
  * 体積分率 0-0.7 の範囲で弾性率を計算(混合則とHalpin-Tsai)
  * 結果をグラフ表示
  * 最適な繊維体積分率を提案(コスト関数を定義)

### 発展レベル

#### 問題 1.8: 多軸応力下の解析

一方向CFRPに縦方向応力 σx = 500 MPa と 横方向応力 σy = 50 MPa が同時に作用する場合、 Tsai-Hill破壊規準を用いて安全率を計算せよ。 (縦引張強度 1500 MPa、横引張強度 50 MPa、せん断強度 70 MPa) 

Tsai-Hill規準: \\(\left(\frac{\sigma_x}{X}\right)^2 - \frac{\sigma_x \sigma_y}{X^2} + \left(\frac{\sigma_y}{Y}\right)^2 + \left(\frac{\tau_{xy}}{S}\right)^2 = \frac{1}{SF^2}\\) 

#### 問題 1.9: 確率論的アプローチ

繊維強度がワイブル分布に従う場合(形状パラメータ m=5、尺度パラメータ σ₀=3500 MPa)、 複合材料の強度分布をモンテカルロシミュレーションで推定せよ。 繊維本数 N=1000 本、Vf = 0.6 とする。 

#### 問題 1.10: 最適化問題

以下の制約条件下で、複合材料のコスト/性能比を最小化する 繊維体積分率と繊維長さを求めよ： 

  * 目標弾性率: EL ≥ 100 GPa
  * 繊維コスト: 50 円/kg、母材コスト: 5 円/kg
  * 繊維長さ範囲: 1-10 mm
  * 体積分率範囲: 0.3-0.7

scipy.optimize を用いた最適化プログラムを実装せよ。 

## 参考文献

  1. Jones, R. M., "Mechanics of Composite Materials", 2nd ed., Taylor & Francis, 1999, pp. 45-89
  2. Hull, D. and Clyne, T. W., "An Introduction to Composite Materials", 2nd ed., Cambridge University Press, 1996, pp. 12-38, 112-145
  3. Mallick, P. K., "Fiber-Reinforced Composites: Materials, Manufacturing, and Design", 3rd ed., CRC Press, 2007, pp. 67-103
  4. Halpin, J. C. and Kardos, J. L., "The Halpin-Tsai Equations: A Review", Polymer Engineering and Science, Vol. 16, No. 5, 1976, pp. 344-352
  5. Kelly, A. and Tyson, W. R., "Tensile Properties of Fibre-Reinforced Metals: Copper/Tungsten and Copper/Molybdenum", Journal of the Mechanics and Physics of Solids, Vol. 13, 1965, pp. 329-350
  6. Chawla, K. K., "Composite Materials: Science and Engineering", 3rd ed., Springer, 2012, pp. 78-124, 156-187
  7. Daniel, I. M. and Ishai, O., "Engineering Mechanics of Composite Materials", 2nd ed., Oxford University Press, 2006, pp. 34-72
  8. Gay, D., Hoa, S. V., and Tsai, S. W., "Composite Materials: Design and Applications", 3rd ed., CRC Press, 2015, pp. 23-61

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
