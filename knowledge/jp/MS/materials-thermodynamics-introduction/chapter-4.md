---
title: "第4章: 二元系相図の読み方と解析"
chapter_title: "第4章: 二元系相図の読み方と解析"
subtitle: Cu-Ni全率固溶体、Al-Si共晶系、Pb-Sn包晶系など実用合金の相図を読み解く
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/materials-thermodynamics-introduction/chapter-4.html>) | Last sync: 2025-11-16

## 学習目標

この章では、材料科学で最も重要なツールの一つである**二元系相図** の読み方と解析方法を学びます。相図は材料の組成と温度に応じた相状態を視覚的に示し、合金設計や熱処理プロセスの設計に不可欠です。

#### この章で習得するスキル

  * 全率固溶体系（Cu-Ni系など）の相図の読み方
  * 共晶系（Al-Si、Pb-Sn系など）の特徴と組織形成
  * 包晶系・偏晶系の反応メカニズム
  * 金属間化合物を含む複雑な相図の解析
  * レバールールによる相分率の定量的計算
  * 冷却曲線と組織形成の関係
  * 実用合金の組成-組織マッピング

#### 💡 二元系相図の重要性

二元系相図は、2つの成分からなる合金系の平衡状態を示します。鉄鋼材料のFe-C系、アルミニウム合金のAl-Cu系、Al-Si系など、工業的に重要な材料の多くは二元系相図を基礎としています。相図を正確に読み解くスキルは、材料開発者にとって必須の能力です。

## 二元系相図の読み方フロー

相図を読む際は、以下のステップを順に実行します：
    
    
    ```mermaid
    graph TD
        A[1. 軸の確認] --> B[2. 相領域の識別]
        B --> C[3. 相境界線の追跡]
        C --> D[4. 不変点の確認]
        D --> E[5. 組成点の設定]
        E --> F[6. 温度での平衡状態決定]
        F --> G[7. レバールール適用]
        G --> H[8. 冷却経路の追跡]
    
        style A fill:#f093fb,stroke:#f5576c,color:#fff
        style H fill:#f093fb,stroke:#f5576c,color:#fff
    ```

#### 相図読解の基本原則

  1. **横軸（組成）** : 左端が成分A（0 wt%）、右端が成分B（100 wt%）
  2. **縦軸（温度）** : 下から上に向かって温度が上昇
  3. **相領域** : α、β、L（液相）などのラベルで区別
  4. **相境界線** : 液相線（liquidus）、固相線（solidus）、溶解度曲線など
  5. **不変点** : 共晶点、包晶点、偏晶点など特定の温度・組成での反応点

## 1\. 全率固溶体系（Cu-Ni系）

全率固溶体系では、液相から固相まで全組成範囲で連続的に固溶します。Cu-Ni系はこの典型例で、両成分が同じFCC構造を持ち、原子半径が近いため全率固溶が可能です。

### 1.1 Cu-Ni系相図の特徴

  * **液相線（Liquidus）** : この線より上では完全に液体（L相）
  * **固相線（Solidus）** : この線より下では完全に固体（α相）
  * **二相領域（L + α）** : 液相と固相が共存する温度範囲
  * **平衡凝固** : 極めて遅い冷却で液相線から固相線に向かって徐々に凝固

#### コード例1: Cu-Ni全率固溶体系相図の作成
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    
    # Cu-Ni系の実験データ（近似値）
    compositions = np.array([0, 20, 40, 60, 80, 100])  # wt% Ni
    liquidus = np.array([1085, 1160, 1260, 1350, 1410, 1455])  # ℃
    solidus = np.array([1085, 1130, 1220, 1310, 1390, 1455])   # ℃
    
    # スプライン補間で滑らかな曲線を生成
    comp_smooth = np.linspace(0, 100, 200)
    liquidus_interp = interp1d(compositions, liquidus, kind='cubic')
    solidus_interp = interp1d(compositions, solidus, kind='cubic')
    
    liquidus_smooth = liquidus_interp(comp_smooth)
    solidus_smooth = solidus_interp(comp_smooth)
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 相領域の塗りつぶし
    ax.fill_between(comp_smooth, liquidus_smooth, 1600,
                    alpha=0.3, color='red', label='液相 (L)')
    ax.fill_between(comp_smooth, solidus_smooth, liquidus_smooth,
                    alpha=0.3, color='yellow', label='L + α (二相領域)')
    ax.fill_between(comp_smooth, 0, solidus_smooth,
                    alpha=0.3, color='blue', label='α固溶体')
    
    # 液相線・固相線のプロット
    ax.plot(comp_smooth, liquidus_smooth, 'r-', linewidth=2.5, label='液相線 (Liquidus)')
    ax.plot(comp_smooth, solidus_smooth, 'b-', linewidth=2.5, label='固相線 (Solidus)')
    
    # 実験データ点
    ax.scatter(compositions, liquidus, c='red', s=80, zorder=5, edgecolors='black')
    ax.scatter(compositions, solidus, c='blue', s=80, zorder=5, edgecolors='black')
    
    # レバールール適用例（60 wt% Ni, 1300℃）
    example_comp = 60
    example_temp = 1300
    ax.plot(example_comp, example_temp, 'ko', markersize=10, zorder=10,
            label=f'例: {example_comp} wt% Ni, {example_temp}℃')
    
    # 水平タイライン（tie line）
    liquidus_comp = np.interp(example_temp, liquidus_smooth[::-1], comp_smooth[::-1])
    solidus_comp = np.interp(example_temp, solidus_smooth[::-1], comp_smooth[::-1])
    ax.plot([solidus_comp, liquidus_comp], [example_temp, example_temp],
            'k--', linewidth=1.5, alpha=0.7)
    ax.plot([solidus_comp, liquidus_comp], [example_temp, example_temp],
            'ko', markersize=6)
    
    # ラベルとフォーマット
    ax.set_xlabel('組成 (wt% Ni)', fontsize=13, fontweight='bold')
    ax.set_ylabel('温度 (℃)', fontsize=13, fontweight='bold')
    ax.set_title('Cu-Ni全率固溶体系相図', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(1000, 1600)
    
    # 純成分の融点を注釈
    ax.annotate('Cu融点\n1085℃', xy=(0, 1085), xytext=(15, 1050),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center')
    ax.annotate('Ni融点\n1455℃', xy=(100, 1455), xytext=(85, 1500),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('cu_ni_phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Cu-Ni系相図を作成しました。")
    print(f"\nレバールール適用例:")
    print(f"組成: {example_comp} wt% Ni, 温度: {example_temp}℃")
    print(f"液相の組成: {liquidus_comp:.1f} wt% Ni")
    print(f"固相の組成: {solidus_comp:.1f} wt% Ni")
    
    # レバールール計算
    f_liquid = (example_comp - solidus_comp) / (liquidus_comp - solidus_comp)
    f_solid = 1 - f_liquid
    print(f"液相の分率: {f_liquid:.3f} ({f_liquid*100:.1f}%)")
    print(f"固相の分率: {f_solid:.3f} ({f_solid*100:.1f}%)")
    

#### 💡 レバールール（Lever Rule）の原理

レバールールは、二相領域で各相の分率を計算する方法です。テコの原理と同様に、組成点から各相境界までの距離の逆比が相分率になります：

$$f_L = \frac{C_0 - C_\alpha}{C_L - C_\alpha}, \quad f_\alpha = \frac{C_L - C_0}{C_L - C_\alpha}$$

ここで、\\(C_0\\)は全体組成、\\(C_L\\)は液相組成、\\(C_\alpha\\)は固相組成です。

## 2\. 共晶系（Al-Si系、Pb-Sn系）

共晶系では、特定の組成（共晶組成）で液相から2つの固相が同時に晶出する**共晶反応** が起こります：

$$L \rightarrow \alpha + \beta \quad (\text{冷却時})$$

### 2.1 共晶系の特徴

  * **共晶点（Eutectic Point）** : 共晶反応が起こる温度・組成
  * **共晶温度** : 系内で最も低い融点
  * **固溶度限（Solubility Limit）** : α相とβ相にそれぞれ溶け込む最大組成
  * **共晶組織** : α相とβ相が層状または棒状に混在した微細組織

#### コード例2: Al-Si共晶系相図とレバールール計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Al-Si系の相図データ（簡略化）
    def create_al_si_phase_diagram():
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # 組成範囲
        comp = np.linspace(0, 100, 300)
    
        # 液相線（Al側とSi側）
        Al_side = 660 - 5.5 * comp[:150]  # Al側の液相線
        Si_side = 1414 - 9.0 * (100 - comp[150:])  # Si側の液相線
        liquidus = np.concatenate([Al_side, Si_side])
    
        # 共晶点
        eutectic_comp = 12.6  # wt% Si
        eutectic_temp = 577   # ℃
    
        # 固溶度限（温度依存性を簡略化）
        solubility_Al = np.full(300, 1.65)  # α相中のSi固溶度限
        solubility_Si = np.full(300, 0.05)  # Si相中のAl固溶度限（ほぼ0）
    
        # 相領域の塗りつぶし
        ax.fill_between(comp, liquidus, 1500, alpha=0.3, color='red', label='液相 (L)')
        ax.fill_between(comp[:150], eutectic_temp, Al_side,
                        alpha=0.3, color='yellow', label='L + α')
        ax.fill_between(comp[150:], eutectic_temp, Si_side,
                        alpha=0.3, color='orange', label='L + β(Si)')
        ax.fill_between(comp, 0, eutectic_temp, alpha=0.2, color='blue', label='α + β')
    
        # 液相線のプロット
        ax.plot(comp, liquidus, 'r-', linewidth=2.5, label='液相線')
    
        # 共晶線
        ax.plot([0, 100], [eutectic_temp, eutectic_temp],
                'g--', linewidth=2, label='共晶線')
    
        # 共晶点
        ax.plot(eutectic_comp, eutectic_temp, 'ro', markersize=12,
                zorder=10, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'共晶点\n({eutectic_comp} wt% Si, {eutectic_temp}℃)',
                    xy=(eutectic_comp, eutectic_temp),
                    xytext=(eutectic_comp + 15, eutectic_temp + 80),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
        # 固溶度限
        ax.plot([solubility_Al[0], solubility_Al[0]], [0, eutectic_temp],
                'b--', linewidth=1.5, alpha=0.7)
        ax.plot([100-solubility_Si[0], 100-solubility_Si[0]], [0, eutectic_temp],
                'b--', linewidth=1.5, alpha=0.7)
    
        # レバールール適用例（8 wt% Si, 600℃）
        example_comp = 8.0
        example_temp = 600
        ax.plot(example_comp, example_temp, 'ko', markersize=10, zorder=10)
    
        # タイライン
        liquidus_comp_at_temp = eutectic_comp + (example_temp - eutectic_temp) * 0.2
        ax.plot([solubility_Al[0], liquidus_comp_at_temp],
                [example_temp, example_temp], 'k--', linewidth=1.5)
        ax.plot([solubility_Al[0], liquidus_comp_at_temp],
                [example_temp, example_temp], 'ko', markersize=6)
    
        # ラベルとフォーマット
        ax.set_xlabel('組成 (wt% Si)', fontsize=13, fontweight='bold')
        ax.set_ylabel('温度 (℃)', fontsize=13, fontweight='bold')
        ax.set_title('Al-Si共晶系相図', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
        ax.set_ylim(500, 750)
    
        # 純成分の融点
        ax.annotate('Al融点\n660℃', xy=(0, 660), xytext=(5, 720),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=10)
    
        plt.tight_layout()
        plt.savefig('al_si_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # レバールール計算
        C0 = example_comp
        C_alpha = solubility_Al[0]
        C_L = liquidus_comp_at_temp
    
        f_L = (C0 - C_alpha) / (C_L - C_alpha)
        f_alpha = 1 - f_L
    
        print(f"\n=== レバールール計算結果 ===")
        print(f"全体組成: {C0} wt% Si")
        print(f"温度: {example_temp}℃")
        print(f"α相組成: {C_alpha} wt% Si")
        print(f"液相組成: {C_L:.1f} wt% Si")
        print(f"液相分率: {f_L:.3f} ({f_L*100:.1f}%)")
        print(f"α相分率: {f_alpha:.3f} ({f_alpha*100:.1f}%)")
    
    create_al_si_phase_diagram()
    

### 2.2 Pb-Sn共晶系と冷却曲線

Pb-Sn系は、はんだ材料として広く使用される共晶合金です。共晶組成（61.9 wt% Sn）では、液相から直接共晶組織に変態するため、冷却曲線に明確な停留（thermal arrest）が現れます。

#### コード例3: Pb-Sn共晶系の冷却曲線シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_cooling_curve(composition_Sn, initial_temp=300, final_temp=100,
                               cooling_rate=0.5, latent_heat_factor=20):
        """
        Pb-Sn系の冷却曲線をシミュレート
    
        Parameters:
        -----------
        composition_Sn : float
            Sn組成（wt%）
        initial_temp : float
            初期温度（℃）
        final_temp : float
            最終温度（℃）
        cooling_rate : float
            冷却速度（℃/s）
        latent_heat_factor : float
            潜熱による温度停留の強さ
        """
        # 相図パラメータ
        eutectic_comp = 61.9  # wt% Sn
        eutectic_temp = 183   # ℃
        Pb_melting = 327      # ℃
        Sn_melting = 232      # ℃
    
        # 液相線温度を組成から推定
        if composition_Sn < eutectic_comp:
            liquidus_temp = Pb_melting - (Pb_melting - eutectic_temp) * (composition_Sn / eutectic_comp)
        else:
            liquidus_temp = Sn_melting - (Sn_melting - eutectic_temp) * ((100 - composition_Sn) / (100 - eutectic_comp))
    
        # 時間配列
        time = np.arange(0, (initial_temp - final_temp) / cooling_rate, 0.1)
        temperature = np.zeros_like(time)
    
        for i, t in enumerate(time):
            # 基本冷却
            temp = initial_temp - cooling_rate * t
    
            # 液相線での停留（凝固開始）
            if abs(temp - liquidus_temp) < 5:
                temp += latent_heat_factor * np.exp(-((temp - liquidus_temp)**2) / 10)
    
            # 共晶温度での停留
            if abs(temp - eutectic_temp) < 5:
                # 共晶組成に近いほど停留が顕著
                eutectic_factor = 1.0 - abs(composition_Sn - eutectic_comp) / eutectic_comp
                temp += latent_heat_factor * eutectic_factor * np.exp(-((temp - eutectic_temp)**2) / 10)
    
            temperature[i] = temp
    
        return time, temperature
    
    # 異なる組成での冷却曲線を計算
    compositions = [20, 40, 61.9, 80]  # wt% Sn
    colors = ['blue', 'green', 'red', 'purple']
    labels = [f'{comp} wt% Sn' for comp in compositions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 冷却曲線のプロット
    for comp, color, label in zip(compositions, colors, labels):
        time, temp = simulate_cooling_curve(comp)
        if comp == 61.9:
            ax1.plot(time, temp, color=color, linewidth=2.5, label=label + ' (共晶)', linestyle='-')
        else:
            ax1.plot(time, temp, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax1.axhline(y=183, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='共晶温度 (183℃)')
    ax1.set_xlabel('時間 (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('温度 (℃)', fontsize=12, fontweight='bold')
    ax1.set_title('Pb-Sn系の冷却曲線', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(100, 350)
    
    # 冷却速度（温度の時間微分）
    ax2.set_title('冷却速度 (dT/dt)', fontsize=14, fontweight='bold')
    for comp, color, label in zip(compositions, colors, labels):
        time, temp = simulate_cooling_curve(comp)
        cooling_rate = np.gradient(temp, time)
        if comp == 61.9:
            ax2.plot(temp, -cooling_rate, color=color, linewidth=2.5, label=label + ' (共晶)')
        else:
            ax2.plot(temp, -cooling_rate, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax2.axvline(x=183, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('温度 (℃)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('冷却速度の絶対値 (℃/s)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(100, 350)
    
    plt.tight_layout()
    plt.savefig('pb_sn_cooling_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== 冷却曲線の特徴 ===")
    print("1. 液相線での停留: 凝固開始により潜熱放出")
    print("2. 共晶温度での停留: 共晶反応（L → α + β）による顕著な潜熱放出")
    print("3. 共晶組成（61.9 wt% Sn）で最も明確な停留")
    print("4. 共晶から離れた組成では、二段階の停留が観察される")
    

## 3\. 包晶系と偏晶系

### 3.1 包晶反応（Peritectic Reaction）

包晶反応は、液相と固相が反応して新しい固相を生成する反応です：

$$L + \alpha \rightarrow \beta \quad (\text{冷却時})$$

Pt-Ag系、Fe-Ni系、Cu-Zn系など多くの合金系で観察されます。

#### コード例4: 包晶反応の可視化（Pt-Ag系モデル）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    
    def create_peritectic_diagram():
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # 組成範囲
        comp = np.linspace(0, 100, 500)
    
        # 包晶系の液相線（簡略化モデル）
        # Pt側
        liquidus_left = 1769 - 15 * comp[:200]
        # Ag側
        liquidus_right = 961 + 7 * (100 - comp[300:])
        # 包晶点付近
        liquidus_middle = np.linspace(liquidus_left[-1], liquidus_right[0], 100)
        liquidus = np.concatenate([liquidus_left, liquidus_middle, liquidus_right])
    
        # 包晶点パラメータ
        peritectic_comp = 42  # wt% Ag
        peritectic_temp = 1186  # ℃
    
        # 固相線
        solidus_left = np.full(200, peritectic_temp)
        solidus_middle = np.full(100, peritectic_temp)
        solidus_right = 961 + 5 * (100 - comp[300:])
        solidus = np.concatenate([solidus_left, solidus_middle, solidus_right])
    
        # α相の固溶度限
        alpha_limit = 15  # wt% Ag
    
        # 相領域の塗りつぶし
        ax.fill_between(comp, liquidus, 2000, alpha=0.3, color='red', label='液相 (L)')
        ax.fill_between(comp[:200], solidus[:200], liquidus[:200],
                        alpha=0.3, color='yellow', label='L + α')
        ax.fill_between(comp[200:], solidus[200:], liquidus[200:],
                        alpha=0.3, color='orange', label='L + β')
        ax.fill_between(comp, 800, solidus, alpha=0.2, color='blue', label='β相')
    
        # α相領域
        ax.fill_between(comp[:100], 800, peritectic_temp,
                        where=(comp[:100] <= alpha_limit),
                        alpha=0.3, color='cyan', label='α相')
        ax.fill_between(comp[:100], 800, peritectic_temp,
                        where=(comp[:100] > alpha_limit),
                        alpha=0.3, color='lightblue', label='α + β')
    
        # 液相線・固相線のプロット
        ax.plot(comp, liquidus, 'r-', linewidth=2.5, label='液相線')
        ax.plot(comp, solidus, 'b-', linewidth=2.5, label='固相線')
    
        # 包晶線
        ax.plot([0, 100], [peritectic_temp, peritectic_temp],
                'g--', linewidth=2, label='包晶線')
    
        # 包晶点
        ax.plot(peritectic_comp, peritectic_temp, 'ro', markersize=14,
                zorder=10, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'包晶点\n({peritectic_comp} wt% Ag, {peritectic_temp}℃)',
                    xy=(peritectic_comp, peritectic_temp),
                    xytext=(peritectic_comp + 15, peritectic_temp + 150),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
        # 包晶反応の矢印
        arrow1 = FancyArrowPatch((peritectic_comp - 10, peritectic_temp + 50),
                                (peritectic_comp, peritectic_temp + 10),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='red', label='液相')
        arrow2 = FancyArrowPatch((peritectic_comp - 20, peritectic_temp + 50),
                                (peritectic_comp, peritectic_temp + 10),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='cyan', label='α相')
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
    
        ax.text(peritectic_comp + 5, peritectic_temp - 30, 'L + α → β',
                fontsize=13, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2))
    
        # ラベルとフォーマット
        ax.set_xlabel('組成 (wt% Ag)', fontsize=13, fontweight='bold')
        ax.set_ylabel('温度 (℃)', fontsize=13, fontweight='bold')
        ax.set_title('包晶系相図（Pt-Ag系モデル）', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 80)
        ax.set_ylim(900, 1900)
    
        # 純成分の融点
        ax.annotate('Pt融点\n1769℃', xy=(0, 1769), xytext=(10, 1850),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=10)
    
        plt.tight_layout()
        plt.savefig('peritectic_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        print("\n=== 包晶反応の特徴 ===")
        print("1. L + α → β: 液相と先に晶出したα相が反応してβ相を生成")
        print("2. 包晶点より低温では、α相はβ相に包まれる（コアシェル構造）")
        print("3. 完全な平衡状態には長時間の拡散が必要")
        print("4. 急冷すると、α相が残留してコアとして残る")
    
    create_peritectic_diagram()
    

#### 💡 包晶反応の実用的重要性

包晶反応は、鋼の凝固（δ-Fe + L → γ-Fe）や、高温超伝導体（YBCO）の合成など、工業的に重要なプロセスで頻繁に現れます。包晶反応では、先に晶出した相が後から生成する相に包まれるため、拡散が律速となり平衡状態に達するのに時間がかかります。

## 4\. 金属間化合物を含む系

金属間化合物（Intermetallic Compound）は、特定の化学量論的組成で形成される秩序構造を持つ化合物です。Al-Cu系のθ相（Al₂Cu）、Ni-Al系のNi₃Alなどが代表例です。

### 4.1 Al-Cu系の特徴

  * **α相** : Al中のCu固溶体（FCC構造）
  * **θ相（Al₂Cu）** : 化学量論的化合物
  * **共晶反応** : L → α + θ（548℃）
  * **時効硬化** : 過飽和固溶体からθ'析出により高強度化

#### コード例5: 金属間化合物を含む系（Al-Cu系モデル）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def create_al_cu_phase_diagram():
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # 組成範囲（wt% Cu）
        comp = np.linspace(0, 60, 600)
    
        # Al側液相線
        liquidus_Al = 660 - 8 * comp[:250]
        # θ相（Al2Cu）近傍
        theta_comp = 33.2  # Al2Cuの化学量論組成（wt% Cu）
        liquidus_theta = np.full(100, 591)  # θ相の融点付近
        # Cu側液相線
        liquidus_Cu = 1085 - 15 * (60 - comp[350:])
        # 中間部分
        liquidus_mid = np.linspace(liquidus_Al[-1], liquidus_theta[0], 50)
        liquidus_mid2 = np.linspace(liquidus_theta[-1], liquidus_Cu[0], 50)
        liquidus = np.concatenate([liquidus_Al, liquidus_mid, liquidus_theta, liquidus_mid2, liquidus_Cu])
    
        # 共晶温度
        eutectic_temp = 548  # ℃
        eutectic_comp = 32.5  # wt% Cu
    
        # 固溶度限（温度依存性を簡略化）
        solubility_Al_at_eutectic = 5.65  # wt% Cu at 548℃
        solubility_Al_at_room_temp = 0.1   # wt% Cu at 25℃
    
        # 相領域の塗りつぶし
        ax.fill_between(comp, liquidus, 1200, alpha=0.3, color='red', label='液相 (L)')
        ax.fill_between(comp[:250], eutectic_temp, liquidus[:250],
                        alpha=0.3, color='yellow', label='L + α')
        ax.fill_between(comp[350:], eutectic_temp, liquidus[350:],
                        alpha=0.3, color='orange', label='L + θ')
    
        # α + θ 二相領域
        ax.fill_between(comp[:400], 0, eutectic_temp,
                        where=(comp[:400] > solubility_Al_at_room_temp),
                        alpha=0.2, color='purple', label='α + θ')
    
        # α単相領域
        ax.fill_between(comp[:100], 0, eutectic_temp,
                        where=(comp[:100] <= solubility_Al_at_eutectic),
                        alpha=0.3, color='cyan', label='α (Al固溶体)')
    
        # θ相領域
        ax.fill_between(comp[300:500], 0, liquidus_theta[0],
                        alpha=0.3, color='lightgreen', label='θ (Al₂Cu)')
    
        # 液相線のプロット
        ax.plot(comp, liquidus, 'r-', linewidth=2.5, label='液相線')
    
        # 共晶線
        ax.plot([0, 60], [eutectic_temp, eutectic_temp],
                'g--', linewidth=2, label='共晶線')
    
        # 共晶点
        ax.plot(eutectic_comp, eutectic_temp, 'ro', markersize=12,
                zorder=10, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'共晶点\n({eutectic_comp} wt% Cu, {eutectic_temp}℃)',
                    xy=(eutectic_comp, eutectic_temp),
                    xytext=(eutectic_comp - 10, eutectic_temp + 80),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
        # θ相の化学量論組成
        ax.axvline(x=theta_comp, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(theta_comp + 1, 1000, 'Al₂Cu\n(33.2 wt% Cu)',
                fontsize=10, fontweight='bold', color='green')
    
        # 固溶度限曲線（簡略化）
        solubility_curve_T = np.linspace(25, eutectic_temp, 100)
        solubility_curve_C = solubility_Al_at_room_temp + \
                             (solubility_Al_at_eutectic - solubility_Al_at_room_temp) * \
                             ((solubility_curve_T - 25) / (eutectic_temp - 25))**0.5
        ax.plot(solubility_curve_C, solubility_curve_T, 'b-', linewidth=2,
                label='α相の固溶度限')
    
        # 時効硬化処理の例（4 wt% Cu合金）
        aging_comp = 4.0
        solution_treat_temp = 520  # 溶体化処理温度
        aging_temp = 190           # 時効処理温度
    
        # 処理経路の矢印
        ax.annotate('', xy=(aging_comp, solution_treat_temp),
                    xytext=(aging_comp, 25),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2.5, linestyle='--'))
        ax.plot(aging_comp, solution_treat_temp, 'bs', markersize=10,
                label='溶体化処理', zorder=10)
        ax.plot(aging_comp, 25, 'b^', markersize=10,
                label='急冷（過飽和固溶体）', zorder=10)
        ax.plot(aging_comp, aging_temp, 'b*', markersize=15,
                label='時効処理（析出硬化）', zorder=10)
    
        # ラベルとフォーマット
        ax.set_xlabel('組成 (wt% Cu)', fontsize=13, fontweight='bold')
        ax.set_ylabel('温度 (℃)', fontsize=13, fontweight='bold')
        ax.set_title('Al-Cu系相図と時効硬化処理', fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 750)
    
        plt.tight_layout()
        plt.savefig('al_cu_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        print("\n=== Al-Cu系の時効硬化処理 ===")
        print(f"1. 溶体化処理: {solution_treat_temp}℃で加熱してCuを完全固溶")
        print(f"2. 急冷: 室温まで急冷して過飽和α固溶体を形成")
        print(f"3. 時効処理: {aging_temp}℃で加熱してθ'相（準安定相）を析出")
        print("4. θ'析出物が転位の運動を阻害 → 高強度化（ジュラルミン）")
        print(f"5. 過時効: 長時間時効でθ相（平衡相）に成長 → 強度低下")
    
    create_al_cu_phase_diagram()
    

#### 時効硬化の原理

Al-Cu系合金（ジュラルミン）は、以下のプロセスで高強度化されます：

  1. **溶体化処理** : α単相領域（520℃程度）で加熱し、Cuを完全にα相に固溶させる
  2. **急冷** : 水中急冷により過飽和固溶体（非平衡状態）を形成
  3. **時効処理** : 190℃程度で加熱し、θ'（GP zone → θ''）の微細析出物を形成
  4. **析出強化** : nm～数十nmの析出物が転位の運動を阻害し、強度が2～3倍に向上

## 5\. 固溶度限の温度依存性

固溶度限（Solubility Limit）は、ある温度で固溶体に溶け込むことができる最大の溶質濃度です。温度が上昇すると、多くの系で固溶度限が増加します。

#### コード例6: 固溶度限の温度依存性（Al-Cu系α相）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # 実験データ（Al-Cu系α相の固溶度限）
    temp_data = np.array([25, 100, 200, 300, 400, 500, 548])  # ℃
    solubility_data = np.array([0.1, 0.5, 1.5, 2.8, 4.0, 5.2, 5.65])  # wt% Cu
    
    # Arrhenius型のフィッティング関数
    def solubility_model(T, A, B):
        """
        固溶度の温度依存性（簡略化モデル）
        C = A * exp(B / T)
        """
        T_kelvin = T + 273.15
        return A * np.exp(B / T_kelvin)
    
    # フィッティング
    popt, pcov = curve_fit(solubility_model, temp_data, solubility_data, p0=[0.01, 3000])
    A_fit, B_fit = popt
    
    # 滑らかな曲線
    temp_smooth = np.linspace(25, 548, 200)
    solubility_smooth = solubility_model(temp_smooth, A_fit, B_fit)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 固溶度限曲線
    ax1.plot(temp_smooth, solubility_smooth, 'b-', linewidth=2.5, label='フィッティング曲線')
    ax1.scatter(temp_data, solubility_data, s=100, c='red', zorder=5,
                edgecolors='black', linewidths=2, label='実験データ')
    
    ax1.set_xlabel('温度 (℃)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('固溶度限 (wt% Cu)', fontsize=13, fontweight='bold')
    ax1.set_title('Al-Cu系α相の固溶度限', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 600)
    ax1.set_ylim(0, 6.5)
    
    # 時効硬化可能な組成範囲を示す
    ax1.fill_between(temp_smooth, 0, solubility_smooth, alpha=0.2, color='green',
                     label='時効硬化可能領域')
    ax1.axhline(y=4.0, color='orange', linestyle='--', linewidth=2,
                label='典型的な合金組成（4 wt% Cu）')
    
    # Arrhenius プロット（ln(C) vs 1/T）
    temp_kelvin = temp_data + 273.15
    ln_solubility = np.log(solubility_data)
    
    ax2.scatter(1000/temp_kelvin, ln_solubility, s=100, c='red', zorder=5,
                edgecolors='black', linewidths=2, label='実験データ')
    
    # フィッティング直線
    temp_kelvin_smooth = temp_smooth + 273.15
    ln_solubility_smooth = np.log(solubility_model(temp_smooth, A_fit, B_fit))
    ax2.plot(1000/temp_kelvin_smooth, ln_solubility_smooth, 'b-', linewidth=2.5,
             label='線形フィット')
    
    ax2.set_xlabel('1000/T (K⁻¹)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('ln(固溶度限)', fontsize=13, fontweight='bold')
    ax2.set_title('Arrheniusプロット', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # フィッティングパラメータを表示
    textstr = f'C = A·exp(B/T)\nA = {A_fit:.4f}\nB = {B_fit:.1f} K'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('al_cu_solubility_limit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 時効硬化処理の最適温度範囲を計算
    print("\n=== 固溶度限データ解析 ===")
    print(f"フィッティングパラメータ: A = {A_fit:.4f}, B = {B_fit:.1f} K")
    print(f"\n温度別の固溶度限:")
    for T in [25, 100, 200, 300, 400, 500]:
        C = solubility_model(T, A_fit, B_fit)
        print(f"  {T}℃: {C:.2f} wt% Cu")
    
    print(f"\n4 wt% Cu合金の場合:")
    target_composition = 4.0
    # 固溶度限が4.0 wt%となる温度を逆算
    def find_solvus_temp(C_target):
        T_kelvin = B_fit / np.log(C_target / A_fit)
        return T_kelvin - 273.15
    
    solvus_temp = find_solvus_temp(target_composition)
    print(f"  固溶度限線（solvus）温度: {solvus_temp:.1f}℃")
    print(f"  溶体化処理温度: {solvus_temp + 20:.1f}℃以上（通常520℃）")
    print(f"  時効処理温度: 150-200℃（過飽和度を保持）")
    print(f"  析出可能なCu量: {target_composition - solubility_model(190, A_fit, B_fit):.2f} wt%")
    

## 6\. 実材料の組成-組織マッピング

相図から予測される組織と実際の組織を対応付けることは、材料設計において極めて重要です。ここでは、Al-Si系鋳造合金の組成と組織の関係をマッピングします。

#### コード例7: 実材料の組成-組織マッピング（Al-Si鋳造合金）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Al-Si鋳造合金の組成と組織データ
    alloy_data = {
        'Alloy': ['A356', 'A380', 'A383', 'A390', 'Silumin', '過共晶Al-Si'],
        'Si_content': [7.0, 8.5, 10.5, 17.0, 12.6, 18.0],  # wt% Si
        'Mg_content': [0.35, 0.0, 0.0, 0.55, 0.0, 0.0],
        'Microstructure': ['亜共晶', '亜共晶', '亜共晶', '過共晶', '共晶', '過共晶'],
        'Primary_phase': ['α-Al', 'α-Al', 'α-Al', 'Primary Si', 'Eutectic', 'Primary Si'],
        'UTS_MPa': [228, 317, 310, 283, 180, 250],  # Ultimate Tensile Strength
        'Elongation': [8.0, 3.5, 3.0, 1.0, 5.0, 0.5],  # %
        'Application': ['自動車部品', 'ダイカスト', 'ダイカスト', 'エンジン', '一般鋳物', '耐摩耗']
    }
    
    df = pd.DataFrame(alloy_data)
    
    # 相図ベースのプロット
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. 組成と組織の関係
    ax1 = fig.add_subplot(gs[0, :])
    
    # 相図の背景
    eutectic_comp = 12.6
    ax1.axvline(x=eutectic_comp, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label='共晶組成')
    ax1.axvspan(0, eutectic_comp, alpha=0.2, color='blue', label='亜共晶領域')
    ax1.axvspan(eutectic_comp, 25, alpha=0.2, color='orange', label='過共晶領域')
    
    # 各合金の位置
    colors = {'亜共晶': 'blue', '共晶': 'red', '過共晶': 'orange'}
    for idx, row in df.iterrows():
        color = colors[row['Microstructure']]
        marker = 'o' if row['Mg_content'] == 0 else '^'
        ax1.scatter(row['Si_content'], idx + 1, s=200, c=color, marker=marker,
                    edgecolors='black', linewidths=2, zorder=5)
        ax1.text(row['Si_content'] + 0.5, idx + 1, row['Alloy'],
                 fontsize=11, va='center', fontweight='bold')
    
    ax1.set_xlabel('Si含有量 (wt%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('合金番号（順序のみ）', fontsize=13, fontweight='bold')
    ax1.set_title('Al-Si鋳造合金の組成分布', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 22)
    ax1.set_ylim(0, len(df) + 1)
    ax1.set_yticks([])
    
    # 2. 引張強度とSi含有量の関係
    ax2 = fig.add_subplot(gs[1, 0])
    
    for idx, row in df.iterrows():
        color = colors[row['Microstructure']]
        marker = 'o' if row['Mg_content'] == 0 else '^'
        ax2.scatter(row['Si_content'], row['UTS_MPa'], s=150, c=color,
                    marker=marker, edgecolors='black', linewidths=2)
        ax2.text(row['Si_content'], row['UTS_MPa'] + 10, row['Alloy'],
                 fontsize=9, ha='center')
    
    ax2.axvline(x=eutectic_comp, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Si含有量 (wt%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('引張強度 (MPa)', fontsize=12, fontweight='bold')
    ax2.set_title('機械的強度とSi含有量', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(5, 20)
    ax2.set_ylim(150, 350)
    
    # 3. 伸びとSi含有量の関係
    ax3 = fig.add_subplot(gs[1, 1])
    
    for idx, row in df.iterrows():
        color = colors[row['Microstructure']]
        marker = 'o' if row['Mg_content'] == 0 else '^'
        ax3.scatter(row['Si_content'], row['Elongation'], s=150, c=color,
                    marker=marker, edgecolors='black', linewidths=2)
        ax3.text(row['Si_content'], row['Elongation'] + 0.3, row['Alloy'],
                 fontsize=9, ha='center')
    
    ax3.axvline(x=eutectic_comp, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Si含有量 (wt%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('伸び (%)', fontsize=12, fontweight='bold')
    ax3.set_title('延性とSi含有量', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(5, 20)
    ax3.set_ylim(0, 10)
    
    # 凡例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=10, label='亜共晶（α-Al初晶）'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, label='共晶（微細組織）'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=10, label='過共晶（Si初晶）'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=10, label='Mg添加合金'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.savefig('al_si_composition_microstructure_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # データ表の出力
    print("\n=== Al-Si鋳造合金の組成-組織-特性マッピング ===\n")
    print(df.to_string(index=False))
    
    print("\n\n=== 組成と組織の関係 ===")
    print("1. 亜共晶合金（< 12.6 wt% Si）:")
    print("   - α-Al初晶 + Al-Si共晶組織")
    print("   - 比較的高い延性（3-8%）")
    print("   - 用途: 自動車部品、ダイカスト")
    
    print("\n2. 共晶合金（12.6 wt% Si）:")
    print("   - 完全な共晶組織（α-Al + Si層状構造）")
    print("   - 最も低い融点（577℃）→鋳造性良好")
    print("   - 用途: 一般鋳物、装飾品")
    
    print("\n3. 過共晶合金（> 12.6 wt% Si）:")
    print("   - 粗大なSi初晶 + Al-Si共晶組織")
    print("   - 高い耐摩耗性、低い延性（< 1%）")
    print("   - 用途: エンジンブロック、耐摩耗部品")
    print("   - P添加により初晶Si微細化可能")
    
    print("\n4. Mg添加の効果（A356, A390）:")
    print("   - Mg₂Si析出による時効硬化")
    print("   - 強度向上（200 → 280 MPa）")
    print("   - T6処理（溶体化 + 時効）で最大強度")
    

#### 実用合金の選択基準

要求特性 | 推奨組成範囲 | 代表合金 | 注意点  
---|---|---|---  
高強度 | 7-10 wt% Si + Mg | A356, A380 | 時効硬化処理必要  
高延性 | 7-9 wt% Si | A356 | Mg添加で延性低下  
鋳造性 | 10-13 wt% Si | Silumin, A380 | 共晶組成付近が最良  
耐摩耗性 | 17-18 wt% Si | A390 | P添加で初晶微細化  
  
## 演習問題

#### 演習1: Cu-Ni系のレバールール計算

**問題:** Cu-60wt%Ni合金を1300℃から極めてゆっくり冷却します。1250℃での平衡状態を求めなさい。

**与えられた情報:**

  * 1250℃での液相線組成: 55 wt% Ni
  * 1250℃での固相線組成: 68 wt% Ni
  * 全体組成: 60 wt% Ni

**求めるもの:**

  1. 液相の分率
  2. 固相の分率
  3. それぞれの相の組成

解答例を見る

**解答:**

レバールールを適用：

$$f_L = \frac{C_\alpha - C_0}{C_\alpha - C_L} = \frac{68 - 60}{68 - 55} = \frac{8}{13} = 0.615$$

$$f_\alpha = \frac{C_0 - C_L}{C_\alpha - C_L} = \frac{60 - 55}{68 - 55} = \frac{5}{13} = 0.385$$

**答え:**

  * 液相の分率: 61.5%（組成: 55 wt% Ni）
  * 固相の分率: 38.5%（組成: 68 wt% Ni）

#### 演習2: Al-Si共晶合金の組織予測

**問題:** Al-10wt%Si合金を700℃から室温まで平衡冷却した場合の最終組織を予測しなさい。

**与えられた情報:**

  * 共晶温度: 577℃
  * 共晶組成: 12.6 wt% Si
  * 室温でのα相中のSi固溶度限: 0.1 wt% Si
  * 577℃でのα相中のSi固溶度限: 1.65 wt% Si

**求めるもの:**

  1. 577℃直上での相状態
  2. 577℃直下での相状態と各相の分率
  3. 室温での最終組織

解答例を見る

**解答:**

**1\. 577℃直上:** L + α（二相領域）

**2\. 577℃直下（共晶反応直後）:**

  * 初晶α相（577℃以上で晶出）+ 共晶組織（α + Si層状構造）
  * レバールール: \\( f_{\text{初晶}\alpha} = \frac{12.6 - 10}{12.6 - 1.65} = 0.237 \\) (23.7%)
  * 共晶組織: 76.3%

**3\. 室温での最終組織:**

  * 初晶α相（Si: 0.1 wt%に減少）+ 共晶α（Si: 0.1 wt%）+ 共晶Si（変化なし）+ 析出Si（α相から析出）
  * 冷却中にα相からSiが析出（1.65 → 0.1 wt%）し、微細なSi粒子が分散

#### 演習3: 時効硬化処理の設計

**問題:** Al-4.5wt%Cu合金を時効硬化処理します。最適な熱処理条件を設計しなさい。

**与えられた情報:**

  * 共晶温度: 548℃
  * 548℃でのα相中のCu固溶度限: 5.65 wt% Cu
  * 室温でのα相中のCu固溶度限: 0.1 wt% Cu

**求めるもの:**

  1. 溶体化処理温度の範囲
  2. 時効処理温度の推奨範囲
  3. 理論的に析出可能なCu量

解答例を見る

**解答:**

**1\. 溶体化処理温度:**

  * 4.5 wt% Cuが完全に固溶する温度以上（約500℃以上）
  * 推奨: 520-540℃（共晶温度より10℃以上低く設定）
  * 理由: 共晶温度以上では部分溶融の危険性あり

**2\. 時効処理温度:**

  * 推奨: 150-200℃（一般的には190℃）
  * 理由: 過飽和度を維持しつつ、適度な拡散速度でθ'を析出
  * 低温（< 150℃）: 時効時間が長くなる
  * 高温（> 200℃）: θ'がθに成長し強度低下（過時効）

**3\. 理論析出量:**

  * 溶体化処理後: 4.5 wt% Cu が過飽和状態で固溶
  * 190℃での平衡固溶度: 約0.5 wt% Cu
  * 析出可能量: 4.5 - 0.5 = 4.0 wt% Cu
  * この量がθ'（Al₂Cu準安定相）として析出し、強度を向上させる

#### 演習4: 相図からの冷却経路追跡

**問題:** Pb-30wt%Sn合金を300℃から100℃まで平衡冷却した場合の相変態シーケンスを記述しなさい。

**与えられた情報:**

  * Pb融点: 327℃
  * 共晶温度: 183℃
  * 共晶組成: 61.9 wt% Sn
  * 183℃でのα相（Pb-rich）中のSn固溶度限: 19.2 wt% Sn

**求めるもの:**

  1. 各温度範囲での相状態
  2. 主要な相変態反応
  3. 最終組織の構成

解答例を見る

**解答:**

**相変態シーケンス:**

  1. **300℃:** 完全液相（L）
  2. **約270℃（液相線）:** L → L + α（α相の晶出開始）
  3. **270℃ - 183℃:** L + α（二相領域） 
     * 冷却に伴いα相が増加、液相が減少
     * α相の組成: Snが徐々に増加
     * 液相の組成: 共晶組成（61.9 wt% Sn）に向かう
  4. **183℃（共晶温度）:** L → α + β（共晶反応） 
     * 残留液相が共晶組織に変態
     * レバールール: \\( f_{\text{初晶}\alpha} = \frac{61.9 - 30}{61.9 - 19.2} = 0.747 \\) (74.7%)
     * 共晶組織: 25.3%
  5. **183℃ - 100℃:** α + β（二相領域） 
     * 固溶度限の温度依存性により、α相からSnがβ相に析出
     * 微細なβ粒子がα相中に分散

**最終組織（100℃）:**

  * 初晶α相（粗大）: 約75%
  * 共晶組織（α + β層状構造）: 約25%
  * 析出β相（微細、α相中に分散）: 少量

## まとめ

この章では、二元系相図の読み方と解析方法を、実用合金系を例に詳しく学びました。

#### 重要ポイントの復習

  1. **全率固溶体系（Cu-Ni）** : 液相線と固相線の間で連続的に凝固。レバールールで各相の分率を計算。
  2. **共晶系（Al-Si, Pb-Sn）** : 共晶点で液相から2つの固相が同時に晶出。共晶組成で最低融点。冷却曲線に明確な停留。
  3. **包晶系（Pt-Ag）** : L + α → β の包晶反応。拡散律速で平衡到達に時間を要する。
  4. **金属間化合物系（Al-Cu）** : 特定組成での化合物形成。時効硬化処理による高強度化（ジュラルミン）。
  5. **固溶度限の温度依存性** : 温度上昇とともに固溶度限が増加。時効硬化の原理。
  6. **組成-組織-特性マッピング** : 相図から組織を予測し、機械的特性と関連付ける。

#### 💡 次のステップ

次章では、**三元系相図と複雑な相変態** を学びます。Fe-C-Cr系（ステンレス鋼）、Al-Cu-Mg系（高強度アルミニウム合金）など、より実用的な合金系での相平衡を理解します。また、準安定相図、連続冷却変態（CCT）図、TTT図など、非平衡状態を扱う高度な相図の読み方も習得します。

#### 学習の確認

以下の質問に答えられるか確認しましょう：

  * Cu-Ni系で60 wt% Ni合金の1250℃での液相分率をレバールールで計算できますか？
  * Al-Si系で亜共晶、共晶、過共晶の組織の違いを説明できますか？
  * Pb-Sn共晶合金の冷却曲線で共晶温度での停留が顕著な理由を説明できますか？
  * 包晶反応と共晶反応の違いを化学式と相図で説明できますか？
  * Al-Cu系ジュラルミンの時効硬化のメカニズムを相図と対応付けて説明できますか？
  * 固溶度限の温度依存性がArrhenius型である理由を説明できますか？
  * 実用Al-Si合金の組成選択（亜共晶 vs 過共晶）を用途に応じて提案できますか？

[← 第3章: 相平衡と相図の基礎](<chapter-3.html>) [第5章: 三元系相図と複雑な相変態 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
