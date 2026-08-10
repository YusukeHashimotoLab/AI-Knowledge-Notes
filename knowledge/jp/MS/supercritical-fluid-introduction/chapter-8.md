---
title: "第8章: 超臨界流体の輸送物性"
chapter_title: "第8章: 超臨界流体の輸送物性"
subtitle: 粘度、拡散係数、熱伝導率、そして超臨界溶媒の選択
---

第6章と第7章では平衡を扱いました。すなわち、超臨界流体がどのような状態にあり、それをどう計算するかです。本章が扱うのは速さです。運動量、物質、熱が実際にどれだけ速く動くのか。粘度・拡散係数・熱伝導率は圧力損失、抽出時間、熱交換器の伝熱面積を決める量であり、臨界点近傍では平衡的な直感が外れるふるまいを示します。熱伝導率は発散するのに温度拡散率は崩壊し、粘度は臨界点に気づくことすらなく通過していきます。最後に、CO₂と水だけでなく超臨界溶媒全体を見渡して、輸送物性の描像を溶媒選択の手順へと落とし込みます。以下に掲載した出力はすべて実際にコードを実行して得たものです。

## 学習目標

本章を修了すると、以下のことができるようになります：

  * 輸送物性と熱力学物性を区別し、それぞれがどの設計量を支配するかを指摘できる
  * 粘度を希薄気体項と密度依存の残差項に分解し、前者をLucas式で実装できる
  * 粘度に臨界異常がほぼ現れない一方で熱伝導率には大きな異常が現れる理由を説明できる
  * 超臨界流体中の溶質拡散係数をStokes-Einstein式とWilke-Chang式で推定し、その不確かさを正直に述べられる
  * 超臨界プロセスに対するシュミット数、プラントル数、レイノルズ数、シャーウッド数を計算し解釈できる
  * 熱伝導率の臨界増大と温度拡散率の臨界減速を定量化できる
  * 超臨界抽出充填層の圧力損失と物質移動係数を見積もり、どの抵抗が律速かを判定できる
  * CO₂、水、エタノール、プロパン、窒素、キセノン、フッ素化合物を同じ換算状態で比較できる
  * 指定された分離課題に対し、制約による絞り込みと順位づけの手順で超臨界溶媒を選択できる

* * *

## 8.1 輸送物性に独立した章が必要な理由

### 平衡は「起こるか」を、速度論は「どれだけかかるか」を答える

状態方程式が答えるのは「この温度・圧力で密度はいくらか、溶質はどれだけ溶けるか」という問いです。抽出にどれだけ時間がかかるか、層の送液にどれだけ動力を要するか、サイクルにどれだけの伝熱面積が必要かについては何も語りません。それらは、いずれも流束と勾配を結びつける3つの係数から導かれます。

係数 | 構成方程式 | 運ぶもの | 決めるもの  
---|---|---|---  
粘度 $\eta$ (Pa·s) | $\tau = -\eta \, \partial u/\partial y$ | 運動量 | 圧力損失、送液動力、流動状態  
二成分拡散係数 $D_{12}$ (m²/s) | $J = -D_{12} \, \partial c/\partial y$ | 物質 | 抽出時間、クロマトグラフィーのピーク幅  
熱伝導率 $\lambda$ (W/(m·K)) | $q = -\lambda \, \partial T/\partial y$ | 熱 | 熱交換器の伝熱面積、再生器の効率  
  
第1章と第2章では定性的な要約を示しました。超臨界流体は液体に近い密度をもちながら、粘度は気体に近く、拡散係数は両者の中間にある、というものです。この要約はオーダーとしては正しく、超臨界プロセスが魅力的である理由そのものです。しかし本章が数値で示すように、工業が実際に使う高密度条件では、よく使われる「液体の10-100倍」という表現よりもかなり穏やかな差になります。

### 輸送係数の3つの寄与

3つの係数はいずれも、物理的起源の異なる寄与の和として書けます。粘度と熱伝導率について、あらゆる参照相関式が採用している標準的な分解は次の形です。

$$ \eta(T,\rho) = \eta_0(T) + \Delta\eta(T,\rho) + \eta_c(T,\rho) $$ $$ \lambda(T,\rho) = \lambda_0(T) + \Delta\lambda(T,\rho) + \lambda_c(T,\rho) $$ 

  * **希薄気体項** $\eta_0$、$\lambda_0$：密度ゼロの極限であり、二体衝突に支配され、気体分子運動論（Chapman-Enskog）または対応状態相関式で予測できます。温度のみに依存します。
  * **残差項（過剰項）** $\Delta\eta$、$\Delta\lambda$：有限密度の効果です。ほとんどの流体でこれは密度に支配され、温度依存性は弱い。例2でこのことを数%の精度で検証します。
  * **臨界増大** $\eta_c$、$\lambda_c$：臨界点で相関長 $\xi$ が発散する長距離密度ゆらぎに由来する異常です。この項があるために、超臨界流体の輸送物性を気体側からも液体側からも外挿できません。

#### 誰もが意外に思う非対称性

臨界増大の大きさは3つの係数で _同じではありません_ 。モード結合理論では発散の指数が大きく異なります。熱伝導率はおおむね $\xi$ に比例して強く発散し、粘度は指数が0.04程度という極めて弱い発散にとどまり（$T_c$ から数ミリケルビンの範囲で数%程度）、相互拡散係数は _ゼロに向かいます_ 。つまり同じ状態点で、1つは発散し、1つは無関係で、1つは消えます。8.2節から8.4節でこの3つすべてを数値で示します。

* * *

## 8.2 粘度

### 希薄気体項とLucas式

密度ゼロにおける気体の粘度は気体分子運動論から導かれます。実務的な手段はLucasの対応状態相関式であり、必要なのは $T_c$、$P_c$、$M$、$Z_c$、双極子モーメントだけです。

$$ \eta_0 \xi = \left[0.807 T_r^{0.618} - 0.357 e^{-0.449 T_r} + 0.340 e^{-4.058 T_r} + 0.018\right] F_P^{\circ} F_Q^{\circ} $$ $$ \xi = 0.176 \left(\frac{T_c}{M^3 P_c^4}\right)^{1/6} $$ 

ここで $T_c$ はK、$M$ はg/mol、$P_c$ はbar、$\xi$ は $(\mu\text{P})^{-1}$、$\eta_0$ はマイクロポアズ（$1\,\mu\text{P} = 10^{-7}$ Pa·s）です。$F_P^{\circ}$ は換算双極子モーメント $\mu_r = 52.46\, \mu^2 P_c / T_c^2$ を通じて分子の極性を補正し、$F_Q^{\circ}$ はヘリウム、水素、重水素にのみ必要な量子補正です。

**$\xi$ の向きに注意すること。** $\xi$ は粘度の _逆数_ スケールなので、相関式は $\eta_0 = [\ldots]/\xi$ であり、$\eta_0 = [\ldots]\times\xi$ ではありません。割る代わりに掛けると、答えは4桁ほど小さくなります。しかも結果は「それらしい指数をもつ小さな数」に見えるため、ざっと眺めただけでは誤りが生き残ってしまいます。これはまさに、本章の元になった保管済み原稿に見つかった不具合です。

コード例1: Lucas式による希薄気体粘度と参照相関式との比較
    
    
    """例1: Lucas式による希薄気体粘度の推定とCoolPropとの比較。"""
    import numpy as np
    import CoolProp.CoolProp as CP
    
    R = 8.314462618  # J/(mol·K)
    
    
    def lucas_dilute_gas_viscosity(T, Tc, Pc_bar, M, Zc, mu_debye=0.0):
        """低圧（希薄気体）粘度を与えるLucas式。
    
        出典: Poling, Prausnitz & O'Connell, "The Properties of Gases and
        Liquids", 5th ed., 式 9-4.15 - 9-4.18
    
        Parameters
        ----------
        T        : 温度 (K)
        Tc       : 臨界温度 (K)
        Pc_bar   : 臨界圧力 (bar)
        M        : モル質量 (g/mol)
        Zc       : 臨界圧縮因子 (-)
        mu_debye : 双極子モーメント (debye、無極性流体は0)
    
        Returns
        -------
        eta : 粘度 (Pa·s)
        """
        Tr = T / Tc
        # 粘度の逆数スケーリング量。単位は 1/マイクロポアズ
        xi = 0.176 * (Tc / (M ** 3 * Pc_bar ** 4)) ** (1.0 / 6.0)
    
        # 換算双極子モーメントと極性補正係数 F_P
        mu_r = 52.46 * mu_debye ** 2 * Pc_bar / Tc ** 2
        if mu_r < 0.022:
            F_P = 1.0
        elif mu_r < 0.075:
            F_P = 1.0 + 30.55 * max(0.0, 0.292 - Zc) ** 1.72
        else:
            F_P = 1.0 + 30.55 * max(0.0, 0.292 - Zc) ** 1.72 * abs(0.96 + 0.1 * (Tr - 0.7))
    
        bracket = (0.807 * Tr ** 0.618 - 0.357 * np.exp(-0.449 * Tr)
                   + 0.340 * np.exp(-4.058 * Tr) + 0.018)
    
        eta_micropoise = bracket * F_P / xi
        return eta_micropoise * 1e-7  # 1マイクロポアズ = 1e-7 Pa·s
    
    
    # 双極子モーメント (debye)。CRC Handbookより
    FLUIDS = {
        'CO2':      0.0,
        'Nitrogen': 0.0,
        'Propane':  0.084,
        'Ethanol':  1.69,
        'Water':    1.85,
    }
    
    print("=== Lucas式の希薄気体粘度とCoolPropの比較（Tr = 1.05で評価）===")
    print(f"{'Fluid':10s} {'T (K)':>8s} {'Lucas':>12s} {'CoolProp':>12s} {'error':>8s}  {'F_P':>5s}")
    print(f"{'':10s} {'':>8s} {'(uPa.s)':>12s} {'(uPa.s)':>12s} {'(%)':>8s}")
    print("-" * 62)
    
    for fluid, mu in FLUIDS.items():
        Tc = CP.PropsSI('Tcrit', fluid)
        Pc = CP.PropsSI('pcrit', fluid)
        rho_c = CP.PropsSI('rhocrit', fluid)
        M = CP.PropsSI('molar_mass', fluid) * 1000.0        # g/mol
        Zc = Pc * (M / 1000.0) / (rho_c * R * Tc)
        T = 1.05 * Tc
    
        eta_lucas = lucas_dilute_gas_viscosity(T, Tc, Pc / 1e5, M, Zc, mu)
        # 参照相関式の希薄気体極限：密度をほぼゼロにして評価する
        eta_ref = CP.PropsSI('V', 'T', T, 'D', 1e-6, fluid)
        err = 100.0 * (eta_lucas - eta_ref) / eta_ref
    
        F_P = eta_lucas / lucas_dilute_gas_viscosity(T, Tc, Pc / 1e5, M, Zc, 0.0)
    
        print(f"{fluid:10s} {T:8.2f} {eta_lucas*1e6:12.3f} {eta_ref*1e6:12.3f} "
              f"{err:+8.2f}  {F_P:5.3f}")
    
    print()
    print("換算双極子モーメント (mu_r) と適用された極性分岐:")
    for fluid, mu in FLUIDS.items():
        Tc = CP.PropsSI('Tcrit', fluid)
        Pc_bar = CP.PropsSI('pcrit', fluid) / 1e5
        mu_r = 52.46 * mu ** 2 * Pc_bar / Tc ** 2
        branch = '無極性' if mu_r < 0.022 else ('弱極性' if mu_r < 0.075 else '極性')
        print(f"  {fluid:10s} mu = {mu:4.2f} D  ->  mu_r = {mu_r:8.5f}  ({branch})")

=== Lucas式の希薄気体粘度とCoolPropの比較（Tr = 1.05で評価）=== Fluid T (K) Lucas CoolProp error F_P (uPa.s) (uPa.s) (%) \-------------------------------------------------------------- CO2 319.33 16.151 15.912 +1.50 1.000 Nitrogen 132.50 8.895 8.985 -1.01 1.000 Propane 388.38 10.836 10.510 +3.10 1.000 Ethanol 540.44 15.583 15.898 -1.98 1.148 Water 679.45 23.803 24.716 -3.69 1.259 換算双極子モーメント (mu_r) と適用された極性分岐: CO2 mu = 0.00 D -> mu_r = 0.00000 (無極性) Nitrogen mu = 0.00 D -> mu_r = 0.00000 (無極性) Propane mu = 0.08 D -> mu_r = 0.00012 (無極性) Ethanol mu = 1.69 D -> mu_r = 0.03545 (弱極性) Water mu = 1.85 D -> mu_r = 0.09461 (極性)

臨界定数と双極子モーメントだけを必要とする相関式で3-4%という結果は良好であり、しかも誤差の悪化の仕方が予想どおりです。水素結合をもつ2つの流体が最も悪く、いずれも極性因子で扱われている部分であって、その下にある気体分子運動論の側の問題ではありません。また、5つの希薄気体粘度がいずれも9-25 µPa·sという狭い帯に収まっていることにも注目してください。密度ゼロでは、どの気体もほとんど同じです。

### 密度依存性：仕事をしているのは残差項

実際の超臨界運転は $\rho/\rho_c \approx 1$-$2$ の領域であり、そこでは残差項が支配的です。希薄気体項を引き算すると残差項が取り出され、結果はほぼ密度だけの関数になります。

コード例2: CO₂の残差粘度と、現れない臨界スパイク
    
    
    """例2: CO₂の残差粘度と、臨界点に現れない発散。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import CoolProp.CoolProp as CP
    
    FLUID = 'CO2'
    Tc = CP.PropsSI('Tcrit', FLUID)
    Pc = CP.PropsSI('pcrit', FLUID)
    rho_c = CP.PropsSI('rhocrit', FLUID)
    
    
    def eta0(T):
        """希薄気体粘度：参照相関式を密度ほぼゼロで評価した値。"""
        return CP.PropsSI('V', 'T', T, 'D', 1e-6, FLUID)
    
    
    def eta(T, rho):
        return CP.PropsSI('V', 'T', T, 'D', rho, FLUID)
    
    
    print(f"CO2: Tc = {Tc:.2f} K, Pc = {Pc/1e6:.3f} MPa, rho_c = {rho_c:.1f} kg/m3")
    print()
    print("=== 残差粘度  d_eta = eta(T, rho) - eta0(T) ===")
    print("d_etaが密度の一本の曲線に重なるなら、それは温度ではなく")
    print("密度に支配されているということです。")
    print()
    header = f"{'rho/rho_c':>10s}" + ''.join(f"{f'{t-273.15:.0f} C':>12s}" for t in
                                            [310.0, 320.0, 340.0, 380.0, 450.0])
    print(header)
    print(f"{'':>10s}" + ''.join(f"{'(uPa.s)':>12s}" for _ in range(5)))
    print("-" * (10 + 12 * 5))
    for rho_r in [0.2, 0.5, 1.0, 1.5, 2.0]:
        rho = rho_r * rho_c
        row = f"{rho_r:10.1f}"
        for T in [310.0, 320.0, 340.0, 380.0, 450.0]:
            row += f"{(eta(T, rho) - eta0(T)) * 1e6:12.3f}"
        print(row)
    
    print()
    print("密度を固定したときの310-450 Kにおけるd_etaのばらつき:")
    for rho_r in [0.2, 0.5, 1.0, 1.5, 2.0]:
        rho = rho_r * rho_c
        vals = np.array([eta(T, rho) - eta0(T) for T in [310.0, 320.0, 340.0, 380.0, 450.0]])
        print(f"  rho/rho_c = {rho_r:3.1f}:  平均 = {vals.mean()*1e6:7.3f} uPa.s, "
              f"ばらつき = 平均の {(vals.max()-vals.min())/vals.mean()*100:5.1f} %")
    
    print()
    print("=== 臨界点で粘度は発散するのか ===")
    print("Tr = 1.001の等温線上をrho_cを横切って進む:")
    T_near = 1.001 * Tc
    print(f"{'rho/rho_c':>10s} {'eta (uPa.s)':>14s} {'d(eta)/d(rho) (uPa.s per kg/m3)':>34s}")
    print("-" * 60)
    rhos = rho_c * np.array([0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10])
    etas = np.array([eta(T_near, r) for r in rhos])
    grad = np.gradient(etas, rhos)
    for r, e, g in zip(rhos, etas, grad):
        print(f"{r/rho_c:10.2f} {e*1e6:14.4f} {g*1e6:34.5f}")
    
    print()
    print("同じ等温線上で、実際に発散する定圧比熱と比較する:")
    for rho_r in [0.90, 1.00, 1.10]:
        rho = rho_r * rho_c
        cp = CP.PropsSI('C', 'T', T_near, 'D', rho, FLUID)
        print(f"  rho/rho_c = {rho_r:4.2f}:  cp = {cp:10.1f} J/(kg.K),  "
              f"eta = {eta(T_near, rho)*1e6:7.3f} uPa.s")
    
    # 図の作成：残差粘度の重ね合わせ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rho_grid = np.linspace(1.0, 2.2 * rho_c, 300)
    for T in [310.0, 320.0, 340.0, 380.0, 450.0]:
        e0 = eta0(T)
        ax1.plot(rho_grid / rho_c, [(eta(T, r)) * 1e6 for r in rho_grid],
                 label=f'{T - 273.15:.0f} °C')
        ax2.plot(rho_grid / rho_c, [(eta(T, r) - e0) * 1e6 for r in rho_grid],
                 label=f'{T - 273.15:.0f} °C')
    ax1.set_xlabel(r'$\rho/\rho_c$'); ax1.set_ylabel(r'$\eta$ (µPa·s)')
    ax1.set_title('Total viscosity'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel(r'$\rho/\rho_c$'); ax2.set_ylabel(r'$\eta-\eta_0(T)$ (µPa·s)')
    ax2.set_title('Residual viscosity vs density')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('co2_residual_viscosity.png', dpi=150)
    print("\n図を 'co2_residual_viscosity.png' に保存しました")

CO2: Tc = 304.13 K, Pc = 7.377 MPa, rho_c = 467.6 kg/m3 === 残差粘度 d_eta = eta(T, rho) - eta0(T) === d_etaが密度の一本の曲線に重なるなら、それは温度ではなく 密度に支配されているということです。 rho/rho_c 37 C 47 C 67 C 107 C 177 C (uPa.s) (uPa.s) (uPa.s) (uPa.s) (uPa.s) \---------------------------------------------------------------------- 0.2 1.029 1.079 1.164 1.292 1.421 0.5 4.594 4.670 4.800 4.986 5.156 1.0 17.182 17.213 17.267 17.350 17.448 1.5 41.018 40.924 40.785 40.669 40.839 2.0 86.379 85.881 85.087 84.153 83.996 密度を固定したときの310-450 Kにおけるd_etaのばらつき: rho/rho_c = 0.2: 平均 = 1.197 uPa.s, ばらつき = 平均の 32.7 % rho/rho_c = 0.5: 平均 = 4.841 uPa.s, ばらつき = 平均の 11.6 % rho/rho_c = 1.0: 平均 = 17.292 uPa.s, ばらつき = 平均の 1.5 % rho/rho_c = 1.5: 平均 = 40.847 uPa.s, ばらつき = 平均の 0.9 % rho/rho_c = 2.0: 平均 = 85.099 uPa.s, ばらつき = 平均の 2.8 % === 臨界点で粘度は発散するのか === Tr = 1.001の等温線上をrho_cを横切って進む: rho/rho_c eta (uPa.s) d(eta)/d(rho) (uPa.s per kg/m3) \------------------------------------------------------------ 0.90 29.0797 0.06808 0.95 30.6715 0.07032 0.99 32.0203 0.07394 1.00 32.3682 0.07487 1.01 32.7204 0.07580 1.05 34.1736 0.07963 1.10 36.0921 0.08206 同じ等温線上で、実際に発散する定圧比熱と比較する: rho/rho_c = 0.90: cp = 295511.7 J/(kg.K), eta = 29.080 uPa.s rho/rho_c = 1.00: cp = 579540.1 J/(kg.K), eta = 32.368 uPa.s rho/rho_c = 1.10: cp = 252983.5 J/(kg.K), eta = 36.092 uPa.s 図を 'co2_residual_viscosity.png' に保存しました

#### 例2が確立したこと

  * **残差粘度は密度の関数である。** 140 Kの温度範囲にわたって、$\rho \geq \rho_c$ であれば密度固定時の $\Delta\eta$ の変動は1-3%です。粘度の制御変数は温度ではなく密度であり、本章全体を圧力ではなく密度で組み立てているのはそのためです。
  * **希薄気体項は小さな補正であって答えではない。** $\rho = 2\rho_c$ では残差項が85 µPa·sに対し、希薄気体項は17 µPa·s程度です。気体分子運動論だけで超臨界粘度を見積もると6倍の過小評価になります。
  * **粘度は臨界点を気にしない。** $T_c$ より0.3 K高い等温線上で、$\eta$ は $\rho_c$ を滑らかに通過し、微分係数は $0.9$-$1.1\,\rho_c$ の全域でわずか20%しか変わりません。同じ等温線上で $c_p$ は $5.8\times10^{5}$ J/(kg·K) に達します。これは第6章で導いた種類の発散です。粘度がもつ臨界増大は、相関式自身の不確かさに埋もれてしまいます。

実務上の帰結は都合の良いものです。超臨界系の流体力学モデルは、臨界性をまったく無視した粘度相関式で組んでも、層の空隙率の不確かさより小さい誤差で収まります。同じ近道を熱伝導率に適用すると重大な誤りになることは、8.4節が示します。

* * *

## 8.3 拡散係数

### 3種類の拡散係数

「拡散係数」という語は多義的で、臨界点近傍ではその曖昧さが問題になります。

係数 | 物理的意味 | $T_c$ 近傍でのふるまい  
---|---|---  
自己拡散係数 $D_s$ | 同種分子中での1分子のランダムウォーク | 滑らか。異常なし  
トレーサー拡散係数 $D_{12}^{\infty}$ | 溶媒中の希薄溶質。抽出で意味をもつのはこれ | 無限希釈で弱い異常  
相互拡散係数 $D_{12}$ | 有限濃度における濃度勾配の緩和 | 混合物の臨界軌跡でゼロになる  
  
溶解度律速のプロセスに典型的な希薄濃度での超臨界抽出では、トレーサー係数が正しい選択であり、これは流体力学的にうまく記述できます。

### Stokes-Einstein式：低粘度が速い拡散を買う

粘度 $\eta$ の連続媒質中にある半径 $r$ の球についての流体力学的結果は次の式です。

$$ D_{12} = \frac{k_B T}{n \pi \eta r}, \qquad n = 6 \ \text{（すべりなし）}, \quad n = 4 \ \text{（完全すべり）} $$ 

この1行に超臨界流体の売り文句がすべて入っています。$D \propto T/\eta$ であり、超臨界流体の $\eta$ はどの液体よりも十分に低い。同時に温度依存性が自明でなくなります。$T$ を上げると分子項が大きくなり、 _かつ_ 密度低下を通じて $\eta$ が下がるため、$D$ は $T$ に対して線形より速く増えます。

古典的な実務上の代替手段は、第2章で引用したWilke-Chang相関式です。

$$ D_{AB} = \frac{7.4\times10^{-8}\,(\phi M_B)^{0.5}\,T}{\eta\,V_A^{0.6}} \quad [\text{cm}^2/\text{s},\ \eta \text{ は cP}] $$ 

これは液体溶媒に対してフィットされた式であり、超臨界流体に適用するのは外挿です。両者を比べることは、安価で正直な不確かさの見積もりになります。

コード例3: 超臨界CO₂中のナフタレン拡散係数とシュミット数
    
    
    """例3: 超臨界CO₂中の溶質拡散係数 - Stokes-Einstein式、Wilke-Chang式、シュミット数。"""
    import numpy as np
    import CoolProp.CoolProp as CP
    
    k_B = 1.380649e-23      # J/K
    N_A = 6.02214076e23     # 1/mol
    
    # ナフタレン：超臨界CO₂中の拡散測定における標準的な溶質
    V_A = 147.6      # Le Basのモル体積 (cm3/mol)
    M_B = 44.01      # CO₂のモル質量 (g/mol)
    
    
    def hydrodynamic_radius(V_le_bas_cm3):
        """Le Basのモル体積から等価な剛体球半径を求める。"""
        V_m3 = V_le_bas_cm3 * 1e-6                       # m3/mol
        return (3.0 * V_m3 / (4.0 * np.pi * N_A)) ** (1.0 / 3.0)
    
    
    def stokes_einstein(T, eta, radius, n=6.0):
        """D = k_B T / (n pi eta r)。n = 6 はすべりなし、n = 4 は完全すべり。"""
        return k_B * T / (n * np.pi * eta * radius)
    
    
    def wilke_chang(T, eta_Pa_s, phi=1.0):
        """Wilke-Chang相関式。戻り値の単位は m2/s。
    
        D_AB = 7.4e-8 (phi M_B)^0.5 T / (eta V_A^0.6)   [cm2/s、etaはcP]
        """
        eta_cP = eta_Pa_s * 1e3
        D_cm2_s = 7.4e-8 * np.sqrt(phi * M_B) * T / (eta_cP * V_A ** 0.6)
        return D_cm2_s * 1e-4
    
    
    r_A = hydrodynamic_radius(V_A)
    print(f"ナフタレン: Le Basのモル体積 {V_A:.1f} cm3/mol "
          f"-> 等価半径 {r_A * 1e10:.2f} A")
    print()
    
    print("=== 40 °Cの超臨界CO₂中におけるナフタレンの拡散係数 ===")
    print(f"{'P':>6s} {'rho':>9s} {'eta':>10s} {'D (S-E, 6pi)':>14s} "
          f"{'D (Wilke-Chang)':>17s} {'Sc':>7s}")
    print(f"{'(MPa)':>6s} {'(kg/m3)':>9s} {'(uPa.s)':>10s} {'(1e-8 m2/s)':>14s} "
          f"{'(1e-8 m2/s)':>17s} {'(-)':>7s}")
    print("-" * 68)
    
    T = 40 + 273.15
    for P_MPa in [8, 10, 15, 20, 25, 30]:
        P = P_MPa * 1e6
        rho = CP.PropsSI('D', 'T', T, 'P', P, 'CO2')
        eta = CP.PropsSI('V', 'T', T, 'P', P, 'CO2')
        D_se = stokes_einstein(T, eta, r_A)
        D_wc = wilke_chang(T, eta)
        Sc = (eta / rho) / D_se
        print(f"{P_MPa:6d} {rho:9.1f} {eta*1e6:10.2f} {D_se*1e8:14.3f} "
              f"{D_wc*1e8:17.3f} {Sc:7.2f}")
    
    print()
    print("=== Dを実際に上げるには：15 MPaでの2つの経路 ===")
    print(f"{'T (C)':>7s} {'rho (kg/m3)':>12s} {'eta (uPa.s)':>12s} "
          f"{'D (1e-8 m2/s)':>15s} {'Sc':>7s}")
    print("-" * 56)
    for T_C in [35, 40, 50, 60, 80, 100]:
        T_k = T_C + 273.15
        rho = CP.PropsSI('D', 'T', T_k, 'P', 15e6, 'CO2')
        eta = CP.PropsSI('V', 'T', T_k, 'P', 15e6, 'CO2')
        D_se = stokes_einstein(T_k, eta, r_A)
        Sc = (eta / rho) / D_se
        print(f"{T_C:7d} {rho:12.1f} {eta*1e6:12.2f} {D_se*1e8:15.3f} {Sc:7.2f}")
    
    print()
    print("=== 同じ溶質について液体溶媒と比較する ===")
    print(f"{'Solvent':22s} {'eta (uPa.s)':>12s} {'D (1e-8 m2/s)':>15s} {'Sc':>9s}")
    print("-" * 61)
    cases = [
        ('scCO2, 40 C, 8 MPa',  'CO2',       313.15,  8.0e6),
        ('scCO2, 40 C, 20 MPa', 'CO2',       313.15, 20.0e6),
        ('n-Hexane, 25 C',      'n-Hexane',  298.15, 101325.0),
        ('Ethanol, 25 C',       'Ethanol',   298.15, 101325.0),
        ('Water, 25 C',         'Water',     298.15, 101325.0),
    ]
    store = {}
    for label, fluid, T_l, P_l in cases:
        rho_l = CP.PropsSI('D', 'T', T_l, 'P', P_l, fluid)
        eta_l = CP.PropsSI('V', 'T', T_l, 'P', P_l, fluid)
        D_l = stokes_einstein(T_l, eta_l, r_A)
        Sc_l = (eta_l / rho_l) / D_l
        store[label] = (eta_l, D_l)
        print(f"{label:22s} {eta_l*1e6:12.1f} {D_l*1e8:15.4f} {Sc_l:9.1f}")
    
    print()
    print("「拡散係数は液体の10-100倍」という主張の検証:")
    eta_hex, D_hex = store['n-Hexane, 25 C']
    for label in ['scCO2, 40 C, 8 MPa', 'scCO2, 40 C, 20 MPa']:
        eta_s, D_s = store[label]
        print(f"  {label:22s}: D / D_hexane = {D_s/D_hex:5.1f} 倍, "
              f"eta_hexane / eta = {eta_hex/eta_s:5.1f} 倍")

ナフタレン: Le Basのモル体積 147.6 cm3/mol -> 等価半径 3.88 A === 40 °Cの超臨界CO₂中におけるナフタレンの拡散係数 === P rho eta D (S-E, 6pi) D (Wilke-Chang) Sc (MPa) (kg/m3) (uPa.s) (1e-8 m2/s) (1e-8 m2/s) (-) \-------------------------------------------------------------------- 8 277.9 21.93 2.694 3.502 2.93 10 628.6 47.65 1.240 1.611 6.11 15 780.2 68.46 0.863 1.122 10.17 20 839.8 79.38 0.744 0.967 12.70 25 879.5 87.86 0.672 0.874 14.85 30 909.9 95.15 0.621 0.807 16.84 === Dを実際に上げるには：15 MPaでの2つの経路 === T (C) rho (kg/m3) eta (uPa.s) D (1e-8 m2/s) Sc \-------------------------------------------------------- 35 815.1 74.49 0.781 11.71 40 780.2 68.46 0.863 10.17 50 699.8 56.77 1.074 7.55 60 604.1 45.88 1.370 5.54 80 427.2 31.98 2.084 3.59 100 332.3 27.52 2.559 3.24 === 同じ溶質について液体溶媒と比較する === Solvent eta (uPa.s) D (1e-8 m2/s) Sc \------------------------------------------------------------- scCO2, 40 C, 8 MPa 21.9 2.6944 2.9 scCO2, 40 C, 20 MPa 79.4 0.7443 12.7 n-Hexane, 25 C 298.0 0.1888 241.0 Ethanol, 25 C 1082.4 0.0520 2652.5 Water, 25 C 890.0 0.0632 1412.4 「拡散係数は液体の10-100倍」という主張の検証: scCO2, 40 C, 8 MPa : D / D_hexane = 14.3 倍, eta_hexane / eta = 13.6 倍 scCO2, 40 C, 20 MPa : D / D_hexane = 3.9 倍, eta_hexane / eta = 3.8 倍

#### 意味をもつのはシュミット数

$Sc = \nu/D_{12}$ は運動量輸送と物質輸送を比べる量であり、あらゆる物質移動相関式に入ってくる無次元数です。超臨界CO₂では**3-17** 。n-ヘキサンでは241、エタノールでは2650です。濃度境界層を支配する無次元数が2-3桁違うというのが、「超臨界流体の物質移動は速い」という主張の背後にある本当の定量的言明であり、拡散係数の比よりもはるかに鋭い言明です。$Sc$ は超臨界流体の密度が低いという事実をすでに織り込んでいるからです。

**「液体の10-100倍」という主張には圧力を添える必要がある。** 例3の最後のブロックでn-ヘキサンと比較しています。40 °C、8 MPaでは超臨界CO₂の拡散係数は14倍で、通常の主張の範囲に十分収まります。40 °C、20 MPaでは3.9倍です。どちらも同じ流体、同じ温度であり、違いは密度の560 kg/m³分だけです。実際の抽出は溶解度のために密度を必要とするからこそ15-30 MPaで運転するので、工業的実務の大半に当てはまるのは控えめな側の値です。優位性は本物ですが、それは数倍であり、圧力容器という代償を払って得られるものです。

### 流体力学的描像が破れる場所

上の数値には2つの但し書きが付きます。

  * **絶対精度。** Le Basの剛体球半径を用いたStokes-Einstein式とWilke-Chang式はここで約30%異なり、いずれも特定の溶質について実測で校正しない限りオーダー見積もりとして扱うべきです。超臨界CO₂に特化してフィットされた相関式（He-Yu、Catchpole-King、船津らの一連の研究）はより良い精度を与え、設計作業に用いるべき道具ですが、流体固有であり公表された係数を必要とします。本章が汎用の2式を使うのはそのためです。
  * **臨界点直近の領域。** 相互拡散は臨界減速を示します。混合物の臨界軌跡に近づくと $D_{12}$ はゼロへ向かい、粘度低下から流体力学的相関式が予測する上昇とはまったく逆になります。したがって $T/\eta$ 型の相関式は臨界軌跡の数%以内では定性的に誤りです。物質移動の設計は異常が減衰した $T_r \gtrsim 1.05$ で行い、臨界点直近は運転する場所ではなく通過する場所と考えるべきです。

* * *

## 8.4 熱伝導率と臨界増大

### 他の2つの係数がもたない項

熱伝導率は、臨界増大が紛れもなく現れる係数です。物理的には、熱は分子衝突だけでなく、臨界点で相関長 $\xi$ が限りなく成長する長寿命の密度ゆらぎの集団的緩和によっても運ばれます。モード結合理論の結果では、増大項はおおよそ次のようにスケールします。

$$ \lambda_c \sim \frac{k_B T \, \rho \, c_p}{6\pi \eta \, \xi} \cdot \xi \propto \frac{k_B T \rho c_p}{6 \pi \eta} $$ 

したがって第6章で導いた $c_p$ の発散を受け継ぎ、$c_p$ が大きいところではどこでも大きくなります。CO₂と水の参照相関式はこの項を明示的に含んでいるので、CoolPropの熱伝導率呼び出しには常に反映されています。

コード例4: CO₂の熱伝導率と臨界増大
    
    
    """例4: CO₂の熱伝導率と臨界増大。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import CoolProp.CoolProp as CP
    
    FLUID = 'CO2'
    Tc = CP.PropsSI('Tcrit', FLUID)
    Pc = CP.PropsSI('pcrit', FLUID)
    rho_c = CP.PropsSI('rhocrit', FLUID)
    
    
    def lam_TP(T, P):
        return CP.PropsSI('L', 'T', T, 'P', P, FLUID)
    
    
    def lam_Trho(T, rho):
        return CP.PropsSI('L', 'T', T, 'D', rho, FLUID)
    
    
    def lam0(T):
        """希薄気体の熱伝導率：相関式を密度ほぼゼロで評価した値。"""
        return CP.PropsSI('L', 'T', T, 'D', 1e-6, FLUID)
    
    
    print(f"CO2: Tc = {Tc:.2f} K, Pc = {Pc / 1e6:.3f} MPa, "
          f"rho_c = {rho_c:.1f} kg/m3")
    print()
    print("=== 等温線に沿った熱伝導率 (mW/(m.K)) ===")
    temps = [32, 35, 40, 50, 80, 150]
    press = [7.5, 8, 9, 10, 15, 20, 30]
    print(f"{'P (MPa)':>8s}" + ''.join(f"{f'{t} C':>10s}" for t in temps))
    print("-" * (8 + 10 * len(temps)))
    for P_MPa in press:
        row = f"{P_MPa:8.1f}"
        for T_C in temps:
            row += f"{lam_TP(T_C + 273.15, P_MPa * 1e6) * 1e3:10.2f}"
        print(row)
    
    print()
    print("=== 各等温線の局所ピーク（7.0-13 MPa、0.002 MPa刻み）===")
    print(f"{'T (C)':>7s} {'Tr':>7s} {'P_peak':>9s} {'rho_peak':>10s} "
          f"{'lam_peak':>10s} {'lam_min':>10s} {'peak/min':>10s}")
    print(f"{'':>7s} {'':>7s} {'(MPa)':>9s} {'(kg/m3)':>10s} "
          f"{'(mW/m/K)':>10s} {'(mW/m/K)':>10s} {'':>10s}")
    print("-" * 66)
    
    P_scan = np.arange(7.0, 13.0 + 1e-9, 0.002)
    for T_C in [31.5, 32, 33, 35, 40, 50, 80]:
        T_k = T_C + 273.15
        lams = np.array([lam_TP(T_k, p * 1e6) for p in P_scan])
        i_pk = int(np.argmax(lams))
        if i_pk == 0 or i_pk == len(P_scan) - 1:
            print(f"{T_C:7.1f} {T_k / Tc:7.4f} {'--':>9s} {'--':>10s} "
                  f"{'--':>10s} {'--':>10s} {'monotonic':>10s}")
            continue
        i_min = i_pk + int(np.argmin(lams[i_pk:]))
        rho_peak = CP.PropsSI('D', 'T', T_k, 'P', P_scan[i_pk] * 1e6, FLUID)
        print(f"{T_C:7.1f} {T_k / Tc:7.4f} {P_scan[i_pk]:9.3f} {rho_peak:10.1f} "
              f"{lams[i_pk] * 1e3:10.2f} {lams[i_min] * 1e3:10.2f} "
              f"{lams[i_pk] / lams[i_min]:10.2f}")
    
    print()
    print("=== rho = rho_c に固定して臨界増大だけを取り出す ===")
    print("臨界密度のままTcに上から近づける:")
    print(f"{'T/Tc - 1':>10s} {'T (K)':>9s} {'P (MPa)':>9s} {'lam':>10s} "
          f"{'lam0':>9s} {'lam/lam0':>9s}")
    print(f"{'':>10s} {'':>9s} {'':>9s} {'(mW/m/K)':>10s} {'(mW/m/K)':>9s} {'':>9s}")
    print("-" * 58)
    for eps in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.5, 1.0]:
        T_k = Tc * (1 + eps)
        l = lam_Trho(T_k, rho_c)
        l0 = lam0(T_k)
        P = CP.PropsSI('P', 'T', T_k, 'D', rho_c, FLUID)
        print(f"{eps:10.4f} {T_k:9.3f} {P / 1e6:9.3f} {l * 1e3:10.2f} "
              f"{l0 * 1e3:9.2f} {l / l0:9.2f}")
    
    lam_water = CP.PropsSI('L', 'T', 298.15, 'P', 101325.0, 'Water')
    lam_near = lam_Trho(Tc * 1.0001, rho_c)
    print()
    print(f"比較のため、25 °Cの液体水の熱伝導率は "
          f"{lam_water * 1e3:.1f} mW/(m.K) です。")
    print(f"Tc + 0.03 K、rho_cのCO2は {lam_near * 1e3:.1f} mW/(m.K) に達し、"
          f"水の {lam_near / lam_water:.2f} 倍になります。")
    print(f"一方、同じ温度で密度ゼロの気体は "
          f"{lam0(Tc * 1.0001) * 1e3:.1f} mW/(m.K) しかありません。")
    
    # 図の作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    P_grid = np.linspace(6.5, 25, 500)
    for T_C in [32, 35, 40, 50, 80]:
        T_k = T_C + 273.15
        ax1.plot(P_grid, [lam_TP(T_k, p * 1e6) * 1e3 for p in P_grid],
                 label=f'{T_C} °C')
    ax1.axvline(Pc / 1e6, color='k', ls=':', lw=1, label=r'$P_c$')
    ax1.set_xlabel('Pressure (MPa)')
    ax1.set_ylabel(r'$\lambda$ (mW/(m·K))')
    ax1.set_title('Isotherms: local spike near $P_c$')
    ax1.legend(); ax1.grid(alpha=0.3)
    
    eps_grid = np.logspace(-4, 0, 60)
    ax2.loglog(eps_grid, [lam_Trho(Tc * (1 + e), rho_c) * 1e3 for e in eps_grid],
               'o-', ms=3)
    ax2.set_xlabel(r'$T/T_c - 1$')
    ax2.set_ylabel(r'$\lambda$ at $\rho_c$ (mW/(m·K))')
    ax2.set_title('Critical enhancement diverges')
    ax2.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('co2_thermal_conductivity.png', dpi=150)
    print("\n図を 'co2_thermal_conductivity.png' に保存しました")

CO2: Tc = 304.13 K, Pc = 7.377 MPa, rho_c = 467.6 kg/m3 === 等温線に沿った熱伝導率 (mW/(m.K)) === P (MPa) 32 C 35 C 40 C 50 C 80 C 150 C \-------------------------------------------------------------------- 7.5 89.03 45.85 36.20 30.64 27.86 30.71 8.0 76.92 84.91 43.87 33.38 28.76 31.07 9.0 78.51 74.95 72.18 41.59 30.84 31.84 10.0 81.64 77.77 71.87 53.98 33.37 32.68 15.0 93.29 90.30 85.33 75.58 51.06 37.79 20.0 101.41 98.79 94.49 86.19 64.91 44.28 30.0 113.73 111.41 107.64 100.45 82.02 57.83 === 各等温線の局所ピーク（7.0-13 MPa、0.002 MPa刻み）=== T (C) Tr P_peak rho_peak lam_peak lam_min peak/min (MPa) (kg/m3) (mW/m/K) (mW/m/K) \------------------------------------------------------------------ 31.5 1.0017 7.464 449.8 188.41 76.89 2.45 32.0 1.0034 7.548 454.6 143.42 76.59 1.87 33.0 1.0066 7.714 453.5 110.98 75.99 1.46 35.0 1.0132 8.052 457.1 88.62 74.77 1.19 40.0 1.0297 -- -- -- -- monotonic 50.0 1.0625 -- -- -- -- monotonic 80.0 1.1612 -- -- -- -- monotonic === rho = rho_c に固定して臨界増大だけを取り出す === 臨界密度のままTcに上から近づける: T/Tc - 1 T (K) P (MPa) lam lam0 lam/lam0 (mW/m/K) (mW/m/K) \---------------------------------------------------------- 0.0001 304.159 7.382 632.40 17.04 37.11 0.0003 304.219 7.393 386.01 17.04 22.65 0.0010 304.432 7.429 229.97 17.06 13.48 0.0030 305.041 7.533 147.33 17.11 8.61 0.0100 307.169 7.897 96.17 17.27 5.57 0.0300 313.252 8.952 71.39 17.74 4.02 0.1000 334.541 12.691 56.71 19.41 2.92 0.5000 456.192 34.070 56.70 29.26 1.94 1.0000 608.256 60.063 68.43 41.58 1.65 比較のため、25 °Cの液体水の熱伝導率は 606.5 mW/(m.K) です。 Tc + 0.03 K、rho_cのCO2は 632.4 mW/(m.K) に達し、水の 1.04 倍になります。 一方、同じ温度で密度ゼロの気体は 17.0 mW/(m.K) しかありません。 図を 'co2_thermal_conductivity.png' に保存しました

#### 数値の読み方

  * **増大は実在し、しかも巨大である。** 臨界密度に保つと、CO₂は $T_c$ より0.03 K高いところで632 mW/(m·K) に達します。希薄気体値の37倍であり、 _液体の水をわずかに上回ります_ 。粘度で見れば99.9%「気体的」な流体が、熱は液体のように伝えるのです。
  * **同時に、きわめて局所的である。** 増大係数は $T_r = 1.01$ で37から5.6へ、$T_r = 1.03$ で4.0へと落ちます。等温線の圧力走査では、局所スパイクはおおよそ $T_r = 1.013$（35 °C）まで見え、40 °Cでは消えて、7-13 MPaの範囲全体で等温線が圧力に対し単調になります。したがって実用的な抽出条件は増大の外側にあり、擬臨界線の近くを意図的に通る超臨界CO₂発電サイクルは内側にあります。
  * **ピークは固定圧力ではなく臨界密度に従う。** 31.5-35 °Cの局所極大は450-457 kg/m³、つまり本質的に $\rho_c = 467.6$ kg/m³ で生じており、その一方でピーク圧力は7.46から8.05 MPaへ移動します。増大の軌跡は密度の軌跡です。

* * *

## 8.5 プラントル数、温度拡散率、臨界減速

### 熱伝導率と熱の浸透速度は同じものではない

$\lambda$ が大きいことは自動的に良い知らせではありません。温度前線の進む速さを決めるのは温度拡散率です。

$$ a = \frac{\lambda}{\rho c_p}, \qquad Pr = \frac{\nu}{a} = \frac{\eta c_p}{\lambda} $$ 

$\lambda$ と $c_p$ はともに臨界点で発散しますが、$c_p$ の方が速く発散します。したがってその比はゼロへ向かいます。伝導が最も良い場所で、まさに熱拡散が止まるのです。これが**臨界減速** であり、臨界点近傍の温度制御を難しくしているのと同じ現象で、そのまま計算できます。

コード例5: プラントル数、温度拡散率、臨界減速
    
    
    """例5: プラントル数、温度拡散率、そして臨界減速。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import CoolProp.CoolProp as CP
    
    FLUID = 'CO2'
    Tc = CP.PropsSI('Tcrit', FLUID)
    Pc = CP.PropsSI('pcrit', FLUID)
    rho_c = CP.PropsSI('rhocrit', FLUID)
    
    
    def transport_group(T, P):
        """(T, P) における rho, eta, lam, cp, nu, a, Pr を返す。"""
        rho = CP.PropsSI('D', 'T', T, 'P', P, FLUID)
        eta = CP.PropsSI('V', 'T', T, 'P', P, FLUID)
        lam = CP.PropsSI('L', 'T', T, 'P', P, FLUID)
        cp = CP.PropsSI('C', 'T', T, 'P', P, FLUID)
        nu = eta / rho                       # 動粘度 (m2/s)
        a = lam / (rho * cp)                 # 温度拡散率 (m2/s)
        Pr = nu / a                          # = eta cp / lam
        return rho, eta, lam, cp, nu, a, Pr
    
    
    print("=== 40 °CのCO₂における輸送関連量 ===")
    print(f"{'P':>6s} {'rho':>8s} {'eta':>9s} {'lam':>9s} {'cp':>10s} "
          f"{'nu':>11s} {'a':>11s} {'Pr':>7s}")
    print(f"{'(MPa)':>6s} {'(kg/m3)':>8s} {'(uPa.s)':>9s} {'(mW/m/K)':>9s} "
          f"{'(J/kg/K)':>10s} {'(1e-8 m2/s)':>11s} {'(1e-8 m2/s)':>11s} {'(-)':>7s}")
    print("-" * 76)
    for P_MPa in [8, 10, 15, 20, 25, 30]:
        rho, eta, lam, cp, nu, a, Pr = transport_group(313.15, P_MPa * 1e6)
        print(f"{P_MPa:6d} {rho:8.1f} {eta*1e6:9.2f} {lam*1e3:9.2f} {cp:10.1f} "
              f"{nu*1e8:11.2f} {a*1e8:11.2f} {Pr:7.3f}")
    
    print()
    print("=== 熱輸送の臨界減速 ===")
    print("rho = rho_c でTcに近づけたときの温度拡散率 a = lam / (rho cp):")
    print(f"{'T/Tc - 1':>10s} {'lam':>10s} {'cp':>12s} {'a':>13s} {'Pr':>10s}")
    print(f"{'':>10s} {'(mW/m/K)':>10s} {'(J/kg/K)':>12s} {'(1e-8 m2/s)':>13s} {'(-)':>10s}")
    print("-" * 58)
    for eps in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.5]:
        T_k = Tc * (1 + eps)
        lam = CP.PropsSI('L', 'T', T_k, 'D', rho_c, FLUID)
        cp = CP.PropsSI('C', 'T', T_k, 'D', rho_c, FLUID)
        eta = CP.PropsSI('V', 'T', T_k, 'D', rho_c, FLUID)
        a = lam / (rho_c * cp)
        Pr = eta * cp / lam
        print(f"{eps:10.4f} {lam*1e3:10.2f} {cp:12.1f} {a*1e8:13.4f} {Pr:10.2f}")
    
    print()
    print("lambdaとcpはともに発散しますが、cpの方が速く発散するのでa -> 0となります。")
    print("流体が最もよく熱を伝えるまさにその点で、熱は動かなくなります。")
    
    print()
    print("=== 熱交換器にとって何を意味するか ===")
    print("1 mmの流路を熱が拡散するのに要する時間 t ~ L^2 / a:")
    L = 1e-3  # m
    for label, T_k, P in [
            ('CO2, 40 C, 20 MPa', 313.15, 20e6),
            ('CO2, 35 C, 8 MPa (near-critical)', 308.15, 8e6),
            ('CO2, 32 C, 7.5 MPa (very near-critical)', 305.15, 7.5e6),
            ('Liquid water, 25 C, 0.1 MPa', 298.15, 101325.0)]:
        fluid = 'Water' if 'water' in label else FLUID
        lam = CP.PropsSI('L', 'T', T_k, 'P', P, fluid)
        rho = CP.PropsSI('D', 'T', T_k, 'P', P, fluid)
        cp = CP.PropsSI('C', 'T', T_k, 'P', P, fluid)
        a = lam / (rho * cp)
        print(f"  {label:42s} a = {a*1e8:8.2f}e-8 m2/s -> t = {L**2/a:7.2f} s")
    
    print()
    print("=== 擬臨界（Widom）線：等圧線上でcpが極大となる位置 ===")
    print(f"{'P (MPa)':>9s} {'T_pc (C)':>10s} {'cp_max (J/kg/K)':>17s} "
          f"{'Pr at T_pc':>12s} {'a (1e-8 m2/s)':>15s}")
    print("-" * 66)
    for P_MPa in [7.5, 8, 9, 10, 12, 15]:
        P = P_MPa * 1e6
        T_scan = np.arange(300.0, 400.0, 0.01)
        cps = np.array([CP.PropsSI('C', 'T', t, 'P', P, FLUID) for t in T_scan])
        i = int(np.argmax(cps))
        T_pc = T_scan[i]
        rho, eta, lam, cp, nu, a, Pr = transport_group(T_pc, P)
        print(f"{P_MPa:9.1f} {T_pc - 273.15:10.2f} {cps[i]:17.1f} {Pr:12.3f} "
              f"{a*1e8:15.3f}")
    
    # 図の作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    P_grid = np.linspace(7.5, 25, 300)
    for T_C in [35, 40, 50, 80]:
        T_k = T_C + 273.15
        ax1.plot(P_grid, [transport_group(T_k, p * 1e6)[6] for p in P_grid],
                 label=f'{T_C} °C')
    ax1.set_xlabel('Pressure (MPa)'); ax1.set_ylabel('Pr (-)')
    ax1.set_title('Prandtl number of CO$_2$'); ax1.legend(); ax1.grid(alpha=0.3)
    
    eps_grid = np.logspace(-4, -0.3, 50)
    ax2.loglog(eps_grid,
               [CP.PropsSI('L', 'T', Tc*(1+e), 'D', rho_c, FLUID)
                / (rho_c * CP.PropsSI('C', 'T', Tc*(1+e), 'D', rho_c, FLUID)) * 1e8
                for e in eps_grid], 'o-', ms=3)
    ax2.set_xlabel(r'$T/T_c - 1$')
    ax2.set_ylabel(r'$a$ at $\rho_c$ ($10^{-8}$ m$^2$/s)')
    ax2.set_title('Critical slowing down of heat diffusion')
    ax2.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('co2_prandtl_thermal_diffusivity.png', dpi=150)
    print("\n図を 'co2_prandtl_thermal_diffusivity.png' に保存しました")

=== 40 °CのCO₂における輸送関連量 === P rho eta lam cp nu a Pr (MPa) (kg/m3) (uPa.s) (mW/m/K) (J/kg/K) (1e-8 m2/s) (1e-8 m2/s) (-) \---------------------------------------------------------------------------- 8 277.9 21.93 43.87 4950.1 7.89 3.19 2.474 10 628.6 47.65 71.87 5657.5 7.58 2.02 3.751 15 780.2 68.46 85.33 2674.7 8.77 4.09 2.146 20 839.8 79.38 94.49 2253.9 9.45 4.99 1.894 25 879.5 87.86 101.60 2066.1 9.99 5.59 1.787 30 909.9 95.15 107.64 1955.6 10.46 6.05 1.729 === 熱輸送の臨界減速 === rho = rho_c でTcに近づけたときの温度拡散率 a = lam / (rho cp): T/Tc - 1 lam cp a Pr (mW/m/K) (J/kg/K) (1e-8 m2/s) (-) \---------------------------------------------------------- 0.0001 632.40 6288566.8 0.0215 321.73 0.0003 386.01 2052206.7 0.0402 172.02 0.0010 229.97 579540.1 0.0849 81.57 0.0030 147.33 171292.8 0.1839 37.67 0.0100 96.17 42910.8 0.4793 14.51 0.0300 71.39 13101.9 1.1653 6.02 0.1000 56.71 4240.5 2.8599 2.53 0.5000 56.70 1619.3 7.4887 1.13 lambdaとcpはともに発散しますが、cpの方が速く発散するのでa -> 0となります。 流体が最もよく熱を伝えるまさにその点で、熱は動かなくなります。 === 熱交換器にとって何を意味するか === 1 mmの流路を熱が拡散するのに要する時間 t ~ L^2 / a: CO2, 40 C, 20 MPa a = 4.99e-8 m2/s -> t = 20.03 s CO2, 35 C, 8 MPa (near-critical) a = 0.68e-8 m2/s -> t = 146.07 s CO2, 32 C, 7.5 MPa (very near-critical) a = 0.65e-8 m2/s -> t = 154.57 s Liquid water, 25 C, 0.1 MPa a = 14.55e-8 m2/s -> t = 6.87 s === 擬臨界（Widom）線：等圧線上でcpが極大となる位置 === P (MPa) T_pc (C) cp_max (J/kg/K) Pr at T_pc a (1e-8 m2/s) \------------------------------------------------------------------ 7.5 31.71 228063.4 43.984 0.157 8.0 34.67 35266.1 12.405 0.560 9.0 40.01 12833.1 6.073 1.159 10.0 45.01 8081.3 4.331 1.642 12.0 53.97 4986.4 3.029 2.392 15.0 64.33 3495.7 2.329 3.199 図を 'co2_prandtl_thermal_diffusivity.png' に保存しました

#### 設計上の3つの帰結

  * **臨界点近傍のプラントル数はオーダー1ではない。** $Pr$ は30 MPaでの1.7から、7.5 MPaの擬臨界線上で44へ、$T_c$ から0.03 K以内では322へと上昇します。$Pr \approx 1$ で検証された $Nu = f(Re, Pr)$ 型の伝熱相関式は、そこでは適用範囲外です。これが超臨界CO₂や超臨界水の伝熱研究で報告される「伝熱劣化」領域の技術的な起源です。
  * **温度拡散率は2桁崩壊する。** $a$ は $T_r = 1.5$ での $7.5\times10^{-8}$ m²/sから $T_r = 1.0001$ での $2.2\times10^{-10}$ m²/sへ落ちます。1 mmの流路を熱が横断する時間は、通常の抽出条件での20秒から臨界点近傍での155秒へ延び、しかも液体の水は6.9秒でその両方を上回ります。
  * **擬臨界（Widom）線こそが舞台である。** 等圧線に沿って $c_p$ の極大を追うと、7.5 MPaの31.7 °Cから15 MPaの64.3 °Cへ至る軌跡が描かれます。これは $Pr$ 最大、$a$ 最小、そして調整可能性が最大となる軌跡です。超臨界CO₂ブレイトンサイクルはこのすぐ上で圧縮するよう設計されており、それによって気体に近い圧縮性のまま液体に近い密度を得る一方、代償として本章のあらゆる輸送異常を引き受けています。

**臨界点近傍では物性に依らない制御チューニングは通用しない。** 擬臨界線上で8から15 MPaの間に $a$ は6倍、$Pr$ は5倍変化します。ある運転点で調整した熱ループは、2 MPa離れた点では大きく外れます。これは第6章が $c_p$ の発散から到達したのと同じ結論を、輸送物性の側から得たものです。

* * *

## 8.6 実際の抽出層における輸送物性

### 運動量：Ergun式

超臨界抽出容器は充填層であり、その圧力損失はErgun式に従います。粘性項（Kozeny-Carman）と慣性項（Burke-Plummer）の和です。

$$ \frac{\Delta P}{L} = \underbrace{\frac{150\,\eta\,(1-\varepsilon)^2 u}{\varepsilon^3 d_p^2}}_{\text{粘性項}} + \underbrace{\frac{1.75\,\rho\,(1-\varepsilon)u^2}{\varepsilon^3 d_p}}_{\text{慣性項}} $$ 

ここで $u$ は空塔速度、$\varepsilon$ は層の空隙率です。密度が相反する2つの効果をもつことに注意してください。 _質量_ 流量を固定すると、密度が高い流体はゆっくり動くので両方の項が下がりますが、密度の高い超臨界流体は粘度も高くなります。

コード例6: 充填抽出層の圧力損失
    
    
    """例6: 充填層抽出器の圧力損失（Ergun式）。"""
    import numpy as np
    import CoolProp.CoolProp as CP
    
    # 層の形状：ラボからパイロット規模の抽出容器
    d_p = 1.0e-3      # 粒子径 (m)。粉砕したコーヒー豆や植物原料を想定
    eps = 0.40        # 空隙率 (-)
    L_bed = 1.0       # 層高 (m)
    D_bed = 0.10      # 層径 (m)
    A_bed = np.pi * D_bed ** 2 / 4.0
    m_dot = 20.0 / 3600.0     # 溶媒の質量流量 (kg/s)、20 kg/h
    
    
    def ergun(rho, eta, u):
        """Ergun式による圧力勾配 (Pa/m)。uは空塔速度 (m/s)。"""
        viscous = 150.0 * eta * (1 - eps) ** 2 * u / (eps ** 3 * d_p ** 2)
        inertial = 1.75 * rho * (1 - eps) * u ** 2 / (eps ** 3 * d_p)
        return viscous, inertial
    
    
    print(f"層: D = {D_bed*1e3:.0f} mm, L = {L_bed:.1f} m, d_p = {d_p*1e3:.1f} mm, "
          f"空隙率 = {eps:.2f}")
    print(f"溶媒質量流量: {m_dot*3600:.0f} kg/h "
          f"（断面積 {A_bed*1e4:.1f} cm2）")
    print()
    print("=== 40 °Cの超臨界CO₂ ===")
    print(f"{'P':>6s} {'rho':>8s} {'eta':>9s} {'u':>10s} {'Re_p':>8s} "
          f"{'dP visc':>10s} {'dP inert':>10s} {'dP total':>10s}")
    print(f"{'(MPa)':>6s} {'(kg/m3)':>8s} {'(uPa.s)':>9s} {'(mm/s)':>10s} {'(-)':>8s} "
          f"{'(kPa)':>10s} {'(kPa)':>10s} {'(kPa)':>10s}")
    print("-" * 76)
    for P_MPa in [8, 10, 15, 20, 30]:
        rho = CP.PropsSI('D', 'T', 313.15, 'P', P_MPa * 1e6, 'CO2')
        eta = CP.PropsSI('V', 'T', 313.15, 'P', P_MPa * 1e6, 'CO2')
        u = m_dot / (rho * A_bed)
        Re_p = rho * u * d_p / eta
        dv, di = ergun(rho, eta, u)
        print(f"{P_MPa:6d} {rho:8.1f} {eta*1e6:9.2f} {u*1e3:10.3f} {Re_p:8.2f} "
              f"{dv*L_bed/1e3:10.4f} {di*L_bed/1e3:10.4f} "
              f"{(dv+di)*L_bed/1e3:10.4f}")
    
    print()
    print("=== 同じ層・同じ質量流量で、25 °Cの従来型液体溶媒を用いた場合 ===")
    print(f"{'Solvent':12s} {'rho':>8s} {'eta':>9s} {'u':>10s} {'Re_p':>8s} "
          f"{'dP total':>10s} {'ratio to':>10s}")
    print(f"{'':12s} {'(kg/m3)':>8s} {'(uPa.s)':>9s} {'(mm/s)':>10s} {'(-)':>8s} "
          f"{'(kPa)':>10s} {'scCO2':>10s}")
    print("-" * 71)
    
    rho_ref = CP.PropsSI('D', 'T', 313.15, 'P', 20e6, 'CO2')
    eta_ref = CP.PropsSI('V', 'T', 313.15, 'P', 20e6, 'CO2')
    u_ref = m_dot / (rho_ref * A_bed)
    dP_ref = sum(ergun(rho_ref, eta_ref, u_ref)) * L_bed
    
    for label, fluid in [('n-Hexane', 'n-Hexane'), ('Ethanol', 'Ethanol'),
                         ('Water', 'Water')]:
        rho = CP.PropsSI('D', 'T', 298.15, 'P', 101325.0, fluid)
        eta = CP.PropsSI('V', 'T', 298.15, 'P', 101325.0, fluid)
        u = m_dot / (rho * A_bed)
        Re_p = rho * u * d_p / eta
        dP = sum(ergun(rho, eta, u)) * L_bed
        print(f"{label:12s} {rho:8.1f} {eta*1e6:9.1f} {u*1e3:10.3f} {Re_p:8.2f} "
              f"{dP/1e3:10.4f} {dP/dP_ref:10.2f}")
    
    print()
    print(f"基準: 40 °C / 20 MPaの超臨界CO2では、層高 {L_bed:.1f} m あたり "
          f"dP = {dP_ref/1e3:.4f} kPa。")
    print()
    print("=== 送液動力の観点 ===")
    print("水力動力 = 体積流量 × 圧力損失。")
    print(f"{'Case':26s} {'Q (L/h)':>10s} {'dP (kPa)':>10s} {'P_hyd (mW)':>12s}")
    print("-" * 60)
    cases = [('scCO2, 40 C, 10 MPa', 'CO2', 313.15, 10e6),
             ('scCO2, 40 C, 20 MPa', 'CO2', 313.15, 20e6),
             ('n-Hexane, 25 C', 'n-Hexane', 298.15, 101325.0),
             ('Water, 25 C', 'Water', 298.15, 101325.0)]
    for label, fluid, T, P in cases:
        rho = CP.PropsSI('D', 'T', T, 'P', P, fluid)
        eta = CP.PropsSI('V', 'T', T, 'P', P, fluid)
        u = m_dot / (rho * A_bed)
        Q = m_dot / rho                    # m3/s
        dP = sum(ergun(rho, eta, u)) * L_bed
        print(f"{label:26s} {Q*3.6e6:10.2f} {dP/1e3:10.4f} {Q*dP*1e3:12.4f}")
    
    print()
    print("=== 層そのものが律速でなくなる領域 ===")
    print("微粉化は物質移動を改善しますが、粘性項は1/d_p^2で増大します:")
    print(f"{'d_p (um)':>10s} {'dP total (kPa)':>16s} {'viscous share':>15s}")
    print("-" * 43)
    for d_p_um in [2000, 1000, 500, 200, 100, 50]:
        d_p = d_p_um * 1e-6
        rho, eta = rho_ref, eta_ref
        u = m_dot / (rho * A_bed)
        dv, di = ergun(rho, eta, u)
        print(f"{d_p_um:10d} {(dv+di)*L_bed/1e3:16.3f} "
              f"{dv/(dv+di)*100:14.1f}%")

層: D = 100 mm, L = 1.0 m, d_p = 1.0 mm, 空隙率 = 0.40 溶媒質量流量: 20 kg/h （断面積 78.5 cm2） === 40 °Cの超臨界CO₂ === P rho eta u Re_p dP visc dP inert dP total (MPa) (kg/m3) (uPa.s) (mm/s) (-) (kPa) (kPa) (kPa) \---------------------------------------------------------------------------- 8 277.9 21.93 2.545 32.26 0.0471 0.0295 0.0766 10 628.6 47.65 1.125 14.84 0.0452 0.0131 0.0583 15 780.2 68.46 0.907 10.33 0.0524 0.0105 0.0629 20 839.8 79.38 0.842 8.91 0.0564 0.0098 0.0662 30 909.9 95.15 0.777 7.43 0.0624 0.0090 0.0714 === 同じ層・同じ質量流量で、25 °Cの従来型液体溶媒を用いた場合 === Solvent rho eta u Re_p dP total ratio to (kg/m3) (uPa.s) (mm/s) (-) (kPa) scCO2 \----------------------------------------------------------------------- n-Hexane 654.9 298.0 1.080 2.37 0.2841 4.29 Ethanol 785.1 1082.4 0.901 0.65 0.8332 12.59 Water 997.0 890.0 0.709 0.79 0.5410 8.17 基準: 40 °C / 20 MPaの超臨界CO2では、層高 1.0 m あたり dP = 0.0662 kPa。 === 送液動力の観点 === 水力動力 = 体積流量 × 圧力損失。 Case Q (L/h) dP (kPa) P_hyd (mW) \------------------------------------------------------------ scCO2, 40 C, 10 MPa 31.82 0.0583 0.5153 scCO2, 40 C, 20 MPa 23.81 0.0662 0.4378 n-Hexane, 25 C 30.54 0.2841 2.4102 Water, 25 C 20.06 0.5410 3.0145 === 層そのものが律速でなくなる領域 === 微粉化は物質移動を改善しますが、粘性項は1/d_p^2で増大します: d_p (um) dP total (kPa) viscous share \------------------------------------------- 2000 0.019 74.3% 1000 0.066 85.2% 500 0.245 92.0% 200 1.459 96.7% 100 5.739 98.3% 50 22.761 99.1%

目を引く点が2つあります。第一に、圧力損失は圧力に対して _平坦_ です。8-30 MPaの全域で0.058-0.077 kPa/mであり、粘度の上昇と速度の低下がほぼ打ち消し合うためです。層の流体力学は運転圧力を選ぶ理由にはなりません。第二に、同じ質量流量で液体溶媒と比べた比はヘキサンで4.3倍、エタノールで12.6倍です。実在する差ですが、粘度だけを素朴に比べた場合の4分の1程度であり、これも液体側の流速が遅くなるためです。そして最後のブロックがトレードオフの効き方を示しています。粒子内拡散を速めるために50 µmまで粉砕すると $\Delta P$ は344倍になり、粘性項が全体の99%を占めるようになります。

### 物質：シャーウッド数とどの抵抗が律速か

充填層における外部（境膜）物質移動はシャーウッド数で相関づけられます。標準的な選択はWakao-Kaguei式です。

$$ Sh = \frac{k_f d_p}{D_{12}} = 2 + 1.1\, Sc^{1/3} Re^{0.6}, \qquad Re = \frac{\rho u d_p}{\eta} $$ 

定数2は孤立した球に対する停滞境膜の極限値です。そこから得られる境膜の時間定数を粒子内拡散の時間定数と比べれば、プラントの運転方針を決める問いに答えられます。

コード例7: 境膜物質移動と律速抵抗の判定
    
    
    """例7: 超臨界CO₂充填層における境膜物質移動。"""
    import numpy as np
    import CoolProp.CoolProp as CP
    
    k_B = 1.380649e-23
    N_A = 6.02214076e23
    
    # 溶質：ナフタレン（Le Bas体積 147.6 cm3/mol）から等価半径を求める
    r_A = (3.0 * 147.6e-6 / (4.0 * np.pi * N_A)) ** (1.0 / 3.0)
    
    # 層と粒子の形状
    d_p = 1.0e-3
    eps = 0.40
    D_bed = 0.10
    A_bed = np.pi * D_bed ** 2 / 4.0
    m_dot = 20.0 / 3600.0        # kg/s
    tortuosity = 3.0             # 粉砕した植物組織の代表値
    porosity_p = 0.5             # 粒子内空隙率
    
    
    def sherwood_wakao_kaguei(Re, Sc):
        """Wakao-Kaguei充填層相関式: Sh = 2 + 1.1 Sc^(1/3) Re^0.6。
    
        適用範囲はおおよそ 3 < Re < 10000。定数2は停滞境膜の極限値。
        """
        return 2.0 + 1.1 * Sc ** (1.0 / 3.0) * Re ** 0.6
    
    
    print("溶質: ナフタレン、等価半径 "
          f"{r_A*1e10:.2f} A。層は d_p = {d_p*1e3:.1f} mm、空隙率 {eps:.2f}")
    print("拡散係数はStokes-Einstein式 D = kT / (6 pi eta r) で求める")
    print()
    print("=== 40 °C、20 kg/h、内径100 mmの層を通す超臨界CO₂ ===")
    print(f"{'P':>6s} {'D':>13s} {'Sc':>7s} {'Re_p':>7s} {'Sh':>7s} "
          f"{'k_f':>12s} {'k_f a_v':>11s} {'tau_film':>10s}")
    print(f"{'(MPa)':>6s} {'(1e-8 m2/s)':>13s} {'(-)':>7s} {'(-)':>7s} {'(-)':>7s} "
          f"{'(1e-5 m/s)':>12s} {'(1/s)':>11s} {'(s)':>10s}")
    print("-" * 78)
    
    a_v = 6.0 * (1 - eps) / d_p          # 層単位体積あたりの界面積 (1/m)
    rows = []
    for P_MPa in [8, 10, 15, 20, 30]:
        P = P_MPa * 1e6
        rho = CP.PropsSI('D', 'T', 313.15, 'P', P, 'CO2')
        eta = CP.PropsSI('V', 'T', 313.15, 'P', P, 'CO2')
        D12 = k_B * 313.15 / (6 * np.pi * eta * r_A)
        nu = eta / rho
        Sc = nu / D12
        u = m_dot / (rho * A_bed)
        Re = rho * u * d_p / eta
        Sh = sherwood_wakao_kaguei(Re, Sc)
        k_f = Sh * D12 / d_p
        tau_film = 1.0 / (k_f * a_v)
        rows.append((P_MPa, D12, Sc, Re, Sh, k_f, tau_film))
        print(f"{P_MPa:6d} {D12*1e8:13.3f} {Sc:7.2f} {Re:7.2f} {Sh:7.2f} "
              f"{k_f*1e5:12.3f} {k_f*a_v:11.4f} {tau_film:10.1f}")
    
    print()
    print(f"界面積 a_v = 6(1-eps)/d_p = {a_v:.0f} m2（層1 m3あたり）")
    print()
    print("=== 境膜と粒子内、どちらが律速か ===")
    print("粒子内有効拡散係数 D_eff = eps_p D / tau、")
    print("球の内部時間定数は tau_int ~ (d_p/2)^2 / (15 D_eff)。")
    print(f"{'P (MPa)':>8s} {'D_eff':>13s} {'tau_film':>10s} {'tau_int':>10s} "
          f"{'Bi_m':>8s} {'controlling':>14s}")
    print(f"{'':>8s} {'(1e-9 m2/s)':>13s} {'(s)':>10s} {'(s)':>10s} {'(-)':>8s}")
    print("-" * 68)
    for P_MPa, D12, Sc, Re, Sh, k_f, tau_film in rows:
        D_eff = porosity_p * D12 / tortuosity
        tau_int = (d_p / 2) ** 2 / (15 * D_eff)
        Bi_m = k_f * (d_p / 2) / D_eff
        which = 'intraparticle' if tau_int > tau_film else 'film'
        print(f"{P_MPa:8d} {D_eff*1e9:13.3f} {tau_film:10.1f} {tau_int:10.1f} "
              f"{Bi_m:8.1f} {which:>14s}")
    
    print()
    print("=== 比較のため、同じ層に液体溶媒を流した場合 ===")
    print(f"{'Solvent':14s} {'D (1e-8 m2/s)':>15s} {'Sc':>8s} {'Re_p':>7s} "
          f"{'Sh':>7s} {'k_f (1e-5 m/s)':>16s}")
    print("-" * 70)
    for label, fluid, T, P in [('scCO2 (20 MPa)', 'CO2', 313.15, 20e6),
                               ('n-Hexane', 'n-Hexane', 298.15, 101325.0),
                               ('Ethanol', 'Ethanol', 298.15, 101325.0)]:
        rho = CP.PropsSI('D', 'T', T, 'P', P, fluid)
        eta = CP.PropsSI('V', 'T', T, 'P', P, fluid)
        D12 = k_B * T / (6 * np.pi * eta * r_A)
        Sc = (eta / rho) / D12
        u = m_dot / (rho * A_bed)
        Re = rho * u * d_p / eta
        Sh = sherwood_wakao_kaguei(Re, Sc)
        k_f = Sh * D12 / d_p
        print(f"{label:14s} {D12*1e8:15.4f} {Sc:8.1f} {Re:7.2f} {Sh:7.2f} "
              f"{k_f*1e5:16.3f}")
    
    print()
    print("注意: ここでのRe_pは液体で0.6-2.4、超臨界CO2で8.9-32であり、")
    print("液体の2行はWakao-Kaguei式の適用範囲（Re > 3）をわずかに下回る。")
    print("参考値として扱うこと。")

溶質: ナフタレン、等価半径 3.88 A。層は d_p = 1.0 mm、空隙率 0.40 拡散係数はStokes-Einstein式 D = kT / (6 pi eta r) で求める === 40 °C、20 kg/h、内径100 mmの層を通す超臨界CO₂ === P D Sc Re_p Sh k_f k_f a_v tau_film (MPa) (1e-8 m2/s) (-) (-) (-) (1e-5 m/s) (1/s) (s) \------------------------------------------------------------------------------ 8 2.694 2.93 32.26 14.65 39.477 1.4212 0.7 10 1.240 6.11 14.84 12.15 15.063 0.5423 1.8 15 0.863 10.17 10.33 11.67 10.075 0.3627 2.8 20 0.744 12.70 8.91 11.53 8.585 0.3090 3.2 30 0.621 16.84 7.43 11.40 7.076 0.2547 3.9 界面積 a_v = 6(1-eps)/d_p = 3600 m2（層1 m3あたり） === 境膜と粒子内、どちらが律速か === 粒子内有効拡散係数 D_eff = eps_p D / tau、 球の内部時間定数は tau_int ~ (d_p/2)^2 / (15 D_eff)。 P (MPa) D_eff tau_film tau_int Bi_m controlling (1e-9 m2/s) (s) (s) (-) \-------------------------------------------------------------------- 8 4.491 0.7 3.7 44.0 intraparticle 10 2.066 1.8 8.1 36.4 intraparticle 15 1.438 2.8 11.6 35.0 intraparticle 20 1.240 3.2 13.4 34.6 intraparticle 30 1.035 3.9 16.1 34.2 intraparticle === 比較のため、同じ層に液体溶媒を流した場合 === Solvent D (1e-8 m2/s) Sc Re_p Sh k_f (1e-5 m/s) \---------------------------------------------------------------------- scCO2 (20 MPa) 0.7443 12.7 8.91 11.53 8.585 n-Hexane 0.1888 241.0 2.37 13.50 2.549 Ethanol 0.0520 2652.5 0.65 13.80 0.717 注意: ここでのRe_pは液体で0.6-2.4、超臨界CO2で8.9-32であり、 液体の2行はWakao-Kaguei式の適用範囲（Re > 3）をわずかに下回る。 参考値として扱うこと。

#### 律速は粒子内拡散であり、それが最適化の対象を変える

物質移動のビオ数は運転範囲全体で34-44であり、粒子内の時間定数はどこでも境膜の時間定数の4-5倍です。この層では溶媒境膜は _ボトルネックではありません_ 。帰結は具体的です。CO₂流量を上げてもおおむね溶媒を浪費するだけであり、粒径を半分にすれば内部時間定数は4分の1になります。だからこそ工業的な超臨界抽出は、流量よりも粒径と原料の前処理をはるかに丁寧に規定するのです。そして例6が示した微粉化の圧力損失代償が、実際に粒径を決める制約になります。

**これらの数値が信用できなくなる境目。** 粒子内空隙率0.5、屈曲度3を仮定した $D_{\mathrm{eff}} = \varepsilon_p D_{12}/\tau$ は、実測値の代用にすぎません。実際のプラント原料は1桁の幅でばらつき、しばしば異方性をもちます。またWakao-Kaguei式は、液体との比較の行では適用範囲 $Re > 3$ をわずかに下回った状態で使っています。 _結論_ 、すなわち粒子内律速であるという判定は十分な余裕をもって成立しており、これを覆すには $D_{\mathrm{eff}}$ に5倍の誤差が必要です。個々の時間定数の値についてはそうではありません。

* * *

## 8.7 CO₂と水以外の超臨界溶媒

第2章と第3章では超臨界CO₂と超臨界水を詳しく扱いました。残りの候補は比較表で名前が挙がるだけで終わっていますが、それでは有用な論点を取りこぼします。同じ換算条件で見ると、これらの輸送物性は驚くほど似ており、それらが使われない理由は輸送物性の理由ではほとんどないのです。

### 超臨界エタノール

$T_c = 241.6$ °C、$P_c = 6.27$ MPa、$\rho_c = 273$ kg/m³。エタノールはCO₂と水のあいだの極性領域を占め、水素結合能を部分的に保ち、再生可能で食品グレードでもあります。臨界圧力は極性溶媒の選択肢の中で最も低い。

  * **極性天然物** ：ポリフェノール、配糖体、カロテノイドなど、純CO₂ではモディファイアなしに到達できない成分
  * **バイオディーゼル** ：250-350 °C、10-20 MPaでの超臨界エステル交換は、アルカリ触媒なし、単一相で進行し、遊離脂肪酸の多い原料にも耐えます。触媒を用いる液相反応が1時間かかるところを数分で完了します
  * **バイオマスの脱リグニン** や医薬品の粒子形成

制約は267 °Cにおける可燃性です。容器の不活性化が必要で、安全対策の費用はCO₂より実質的に高くなります。

### 超臨界プロパン

$T_c = 96.7$ °C、$P_c = 4.25$ MPa。実用的な候補の中で臨界圧力が最も低い。プロパンは同程度の条件でCO₂よりも明確に優れた脂質溶媒であり、可燃性・爆発性であるにもかかわらず植物油抽出と石油の脱アスファルトで生き残っているのはそのためです。例8が示すように、同じ換算条件では実用的な流体群の中で粘度が最も低く、溶質拡散係数が最も高くなります。

### 超臨界窒素とキセノン

窒素（$T_c = -147$ °C、$P_c = 3.40$ MPa）は化学的には理想的で、完全に不活性、安価、無毒ですが、超臨界状態に到達するには極低温設備が必要なため、抽出には実質的に使えません。溶媒としてではなく不活性雰囲気プロセスに登場します。

キセノン（$T_c = 16.6$ °C、$P_c = 5.84$ MPa）は本節の中で最も魅力的な臨界温度をもち、不活性・無毒で、特殊な医薬品や放射性医薬品の分野で使われます。1 kgあたり数千ドルというコストが、他に手段のない用途に限定させています。さらに例8が示すように、CoolPropにはキセノンの参照輸送物性モデルがまったくありません。これ自体が1つの教訓です。参照品質の物性データが利用できるかどうかは、後付けの話ではなく溶媒選択に対する実在の工学的制約なのです。

### フッ素化合物

R-134a（$T_c = 101$ °C、$P_c = 4.06$ MPa）とSF₆（$T_c = 45.6$ °C、$P_c = 3.76$ MPa）は臨界点が扱いやすく、オゾン層破壊係数はゼロで、参照状態方程式も整備され、冷媒産業から受け継いだ成熟した取り扱い技術があります。同時に地球温暖化係数はそれぞれおよそ1430と24 300であり、現行の削減規制の下では代替手段が存在するあらゆる新規プロセスから排除されます。同じ換算条件での密度は本節の流体群で最も高いので、輸送物性だけで見れば魅力的なはずです。溶媒選択が輸送物性の問題ではないことを、これは明瞭に示しています。

### すべてを同じ換算状態で並べる

公平な比較は換算条件を揃えたものです。対応状態原理が違いを縮めると主張しているのは、まさにこの比較についてです。

コード例8: 超臨界溶媒群の輸送物性比較
    
    
    """例8: 超臨界溶媒群の輸送物性比較。"""
    import numpy as np
    import CoolProp.CoolProp as CP
    
    k_B = 1.380649e-23
    N_A = 6.02214076e23
    r_A = (3.0 * 147.6e-6 / (4.0 * np.pi * N_A)) ** (1.0 / 3.0)   # ナフタレン
    
    FLUIDS = ['CO2', 'Ethanol', 'Water', 'Propane', 'Nitrogen', 'Xenon',
              'R134a', 'SF6']
    
    Tr_TARGET = 1.05
    Pr_TARGET = 2.00
    
    
    def matched_state(fluid, Tr=Tr_TARGET, Pr=Pr_TARGET):
        Tc = CP.PropsSI('Tcrit', fluid)
        Pc = CP.PropsSI('pcrit', fluid)
        return Tr * Tc, Pr * Pc, Tc, Pc
    
    
    print(f"=== すべての流体を同じ換算状態で比較する: Tr = {Tr_TARGET}, "
          f"Pr = {Pr_TARGET} ===")
    print(f"{'Fluid':10s} {'Tc':>8s} {'Pc':>7s} {'T':>8s} {'P':>7s} {'rho':>8s} "
          f"{'eta':>9s} {'lam':>9s} {'nu':>11s}")
    print(f"{'':10s} {'(C)':>8s} {'(MPa)':>7s} {'(C)':>8s} {'(MPa)':>7s} "
          f"{'(kg/m3)':>8s} {'(uPa.s)':>9s} {'(mW/m/K)':>9s} {'(1e-8 m2/s)':>11s}")
    print("-" * 82)
    
    data = {}
    for fluid in FLUIDS:
        T, P, Tc, Pc = matched_state(fluid)
        rho = CP.PropsSI('D', 'T', T, 'P', P, fluid)
        try:
            eta = CP.PropsSI('V', 'T', T, 'P', P, fluid)
            lam = CP.PropsSI('L', 'T', T, 'P', P, fluid)
        except ValueError:
            print(f"{fluid:10s} {Tc-273.15:8.2f} {Pc/1e6:7.3f} {T-273.15:8.2f} "
                  f"{P/1e6:7.3f} {rho:8.1f} "
                  f"{'no model':>9s} {'no model':>9s} {'--':>11s}")
            data[fluid] = None
            continue
        nu = eta / rho
        data[fluid] = (T, P, rho, eta, lam, nu)
        print(f"{fluid:10s} {Tc-273.15:8.2f} {Pc/1e6:7.3f} {T-273.15:8.2f} "
              f"{P/1e6:7.3f} {rho:8.1f} {eta*1e6:9.2f} {lam*1e3:9.2f} {nu*1e8:11.2f}")
    
    print()
    print("=== 同じ換算状態における派生量 ===")
    print(f"{'Fluid':10s} {'cp':>10s} {'a':>12s} {'Pr':>8s} {'D (S-E)':>12s} {'Sc':>8s}")
    print(f"{'':10s} {'(J/kg/K)':>10s} {'(1e-8 m2/s)':>12s} {'(-)':>8s} "
          f"{'(1e-8 m2/s)':>12s} {'(-)':>8s}")
    print("-" * 64)
    for fluid in FLUIDS:
        if data[fluid] is None:
            print(f"{fluid:10s} {'--':>10s} {'--':>12s} {'--':>8s} {'--':>12s} {'--':>8s}")
            continue
        T, P, rho, eta, lam, nu = data[fluid]
        cp = CP.PropsSI('C', 'T', T, 'P', P, fluid)
        a = lam / (rho * cp)
        Pr = nu / a
        D12 = k_B * T / (6 * np.pi * eta * r_A)
        Sc = nu / D12
        print(f"{fluid:10s} {cp:10.1f} {a*1e8:12.3f} {Pr:8.3f} {D12*1e8:12.3f} "
              f"{Sc:8.2f}")
    
    print()
    print("=== Tr = 1.05, Pr = 2.00 における順位（上位から）===")
    avail = [f for f in FLUIDS if data[f] is not None]
    by_visc = sorted(avail, key=lambda f: data[f][3])
    by_lam = sorted(avail, key=lambda f: -data[f][4])
    by_rho = sorted(avail, key=lambda f: -data[f][2])
    print("粘度が低い（送液が容易）順:       " + ", ".join(by_visc))
    print("熱伝導率が高い（除熱に有利）順:   " + ", ".join(by_lam))
    print("密度が高い（溶解力が大きい）順:   " + ", ".join(by_rho))
    
    print()
    print("=== 落とし穴：換算状態が同じでも運転条件は同じではない ===")
    print(f"{'Fluid':10s} {'T (C)':>8s} {'P (MPa)':>9s} {'comment'}")
    print("-" * 60)
    comments = {
        'CO2': '常温付近の容器で運転できる',
        'Ethanol': '可燃性、不活性化が必要',
        'Water': '腐食と耐食合金のコストが支配的',
        'Propane': '可燃性だが圧力は最も低い',
        'Nitrogen': '極低温設備が必要',
        'Xenon': '参照輸送物性モデルがなく、しかも高価',
        'R134a': '高GWP冷媒であり規制で削減対象',
        'SF6': 'GWP 24 300、使用が制限される',
    }
    for fluid in FLUIDS:
        T, P, Tc, Pc = matched_state(fluid)
        print(f"{fluid:10s} {T-273.15:8.1f} {P/1e6:9.3f} {comments[fluid]}")

=== すべての流体を同じ換算状態で比較する: Tr = 1.05, Pr = 2.0 === Fluid Tc Pc T P rho eta lam nu (C) (MPa) (C) (MPa) (kg/m3) (uPa.s) (mW/m/K) (1e-8 m2/s) \---------------------------------------------------------------------------------- CO2 30.98 7.377 46.18 14.755 727.2 60.43 78.66 8.31 Ethanol 241.56 6.268 267.29 12.536 422.6 53.92 130.84 12.76 Water 373.95 22.064 406.30 44.128 527.2 62.19 413.13 11.80 Propane 96.74 4.251 115.23 8.502 343.7 42.74 68.55 12.44 Nitrogen -146.96 3.396 -140.65 6.792 494.9 35.12 56.64 7.10 Xenon 16.58 5.842 31.07 11.684 1776.1 no model no model -- R134a 101.06 4.059 119.77 8.119 786.6 64.92 50.67 8.25 SF6 45.57 3.755 61.51 7.510 1140.9 82.30 48.71 7.21 === 同じ換算状態における派生量 === Fluid cp a Pr D (S-E) Sc (J/kg/K) (1e-8 m2/s) (-) (1e-8 m2/s) (-) \---------------------------------------------------------------- CO2 2945.4 3.672 2.263 0.997 8.33 Ethanol 6266.1 4.941 2.582 1.891 6.75 Water 8177.0 9.584 1.231 2.061 5.72 Propane 3905.8 5.107 2.435 1.715 7.25 Nitrogen 3277.1 3.492 2.032 0.712 9.97 Xenon -- -- -- -- -- R134a 2064.8 3.120 2.645 1.142 7.23 SF6 1357.3 3.145 2.293 0.767 9.40 === Tr = 1.05, Pr = 2.00 における順位（上位から）=== 粘度が低い（送液が容易）順: Nitrogen, Propane, Ethanol, CO2, Water, R134a, SF6 熱伝導率が高い（除熱に有利）順: Water, Ethanol, CO2, Propane, Nitrogen, R134a, SF6 密度が高い（溶解力が大きい）順: SF6, R134a, CO2, Water, Nitrogen, Ethanol, Propane === 落とし穴：換算状態が同じでも運転条件は同じではない === Fluid T (C) P (MPa) comment \------------------------------------------------------------ CO2 46.2 14.755 常温付近の容器で運転できる Ethanol 267.3 12.536 可燃性、不活性化が必要 Water 406.3 44.128 腐食と耐食合金のコストが支配的 Propane 115.2 8.502 可燃性だが圧力は最も低い Nitrogen -140.6 6.792 極低温設備が必要 Xenon 31.1 11.684 参照輸送物性モデルがなく、しかも高価 R134a 119.8 8.119 高GWP冷媒であり規制で削減対象 SF6 61.5 7.510 GWP 24 300、使用が制限される

#### 対応状態原理は働いており、それこそが要点

  * **化学的に無関係な7流体で、粘度の幅は2.3倍にすぎない** （35-82 µPa·s）。動粘度は1.8倍（7.1-12.8 $\times 10^{-8}$ m²/s）、プラントル数は1.2-2.6、シュミット数は5.7-10.0です。$T_r$ と $P_r$ を揃えれば、超臨界流体はどれもほぼ同じように運動量、熱、物質を運びます。
  * **例外は水の熱伝導率** で、413 mW/(m·K)、CO₂の5倍です。水素結合は超臨界状態でも十分に生き残り、対応状態原理が他をすべて平坦にしたところでも、熱輸送では水を外れ値のままにします。
  * **したがって輸送物性は識別要因にならない。** 同じ換算状態に到達するために必要な絶対温度・絶対圧力と、そこに到達する際の安全・規制・コスト上の帰結は、大きく異なります。CO₂は46 °C、14.8 MPaに対し、水は406 °C、44.1 MPaです。これが本当の選択問題であり、8.8節で定式化します。

* * *

## 8.8 超臨界溶媒の選択

### 決定木

まず溶質の極性、次に熱安定性、そして安全性。輸送物性は最後に、同点決着の判定として入ってきます。
    
    
    ```mermaid
    graph TD
        A[分離または反応の目的] --> B{溶質の極性}
        B -->|非極性| C{熱安定性}
        B -->|中極性| D{バイオマス由来か}
        B -->|極性・イオン性| E{水溶性か}
    
        C -->|低温が必須| F[超臨界CO2]
        C -->|高温可| G[超臨界プロパン]
    
        D -->|Yes| H[超臨界エタノール]
        D -->|No| I[超臨界CO2 + 共溶媒]
    
        E -->|Yes| J[超臨界水]
        E -->|No| K[超臨界エタノール]
    
        F --> L[抽出・洗浄・粒子製造]
        G --> M[油脂抽出]
        H --> N[バイオディーゼル・脱リグニン]
        I --> O[極性天然物の抽出]
        J --> P[酸化・水熱合成]
        K --> Q[精製・反応媒体]
    
        style A fill:#e0f7fa
        style F fill:#e8f5e9
        style J fill:#fff3e0
    ```

### 比較表

流体 | $T_c$ (°C) | $P_c$ (MPa) | 極性 | 実用密度域 (kg/m³) | 安全性 | コスト | 主な用途  
---|---|---|---|---|---|---|---  
**CO₂**|  31.0| 7.38| 非〜弱極性（モディファイアで調整可） | 200-900| 優：無毒、不燃、GRAS| 低 | 抽出、洗浄、粒子製造、発電サイクル  
**水**|  374.0| 22.06| 極性 → $T_c$ 以上で非極性 | 50-600| 良だが腐食と熱傷のリスクが大| 極低 | 酸化（SCWO）、水熱合成、ガス化  
**エタノール**|  241.6| 6.27| 中極性 | 100-500| 良だが運転温度で可燃| 中 | バイオディーゼル、極性天然物、脱リグニン  
**プロパン**|  96.7| 4.25| 非極性 | 150-500| 可燃・爆発性| 低 | 植物油抽出、脱アスファルト  
**窒素**|  -147.0| 3.40| 非極性 | 300-800| 優（不活性）。窒息リスクあり| 低 | 不活性雰囲気プロセス。抽出には不向き  
**キセノン**|  16.6| 5.84| 非極性 | 1000-2000| 優（不活性）| 極高 | 特殊な医薬品用途。参照輸送物性モデルなし  
**R-134a**|  101.1| 4.06| 弱極性 | 300-900| 良、不燃| 高 | 特殊洗浄。GWP 1430により規制対象  
  
### 共溶媒：極性を買う

数モル%の極性モディファイアを加えるのは、CO₂単独では到達できない極性域へ移すための標準的な手段です。溶解度の側面は第2章が扱っています。輸送物性の側面も重要で、モディファイアは粘度を上げ、したがって拡散係数を下げます。

モディファイア | 代表的な添加量 | 効果 | 対象溶質  
---|---|---|---  
エタノール| 1-10 vol%（5-15 mol%）| 極性を高め水素結合を付与| ポリフェノール、アルカロイド  
メタノール| 1-5 vol%| エタノールより強い極性化| 糖類、アミノ酸  
水| 1-10 vol%| 親水性を大きく向上| ペプチド、タンパク質  
酢酸| 0.1-3 vol%| 媒体を酸性化| 塩基性化合物  
  
40 °C、20 MPaにおける純CO₂のHildebrandパラメータは12 MPa1/2程度で、ヘキサンに近い値です。10 mol%のエタノールを加えると15-16 MPa1/2程度まで上がり、トルエンに近くなります。代償は、モディファイアを製品から分離しなければならないこと、不要成分の共抽出により選択性がふつう低下すること、そして参照混合物データが乏しいことです。第7章が例外の発生で示すとおり、CoolPropはCO₂ + エタノール系の二成分相互作用データをもっていません。

### 安全性と規制上の制約

流体 | 主なハザード | 必要な対策 | 規制上の位置づけ  
---|---|---|---  
CO₂| 高圧。換気不良時の窒息| 安全弁、圧力インターロック、換気とCO₂濃度監視| GRAS。食品・医薬用途で広く認可  
水| 超高圧、熱傷、腐食と塩の析出閉塞| ニッケル合金またはライニング容器、断熱、腐食監視| 認可済。排水の放流は規制対象  
エタノール| 運転温度における可燃性| 不活性ガスシール、防爆電気設備| 食品グレード品あり。危険物規制の対象  
プロパン| 可燃性と爆発| 危険場所区分、ガス検知、自動遮断| 制限あり。可燃性ガス規制の対象  
R-134a / SF₆| 窒息。環境への放出| 漏洩検知と回収、大気放出の禁止| GWPを理由に制限または削減対象  
  
### 説明できるスクリーニング手順

適切な構成は、ほとんどが計算不能で可否判定である「厳しい制約」と、計算物性のみを使うべき「順位づけ」を分離することです。温度、極性、安全性、コストに任意の点数を足し上げるスコアリング方式は、客観的に見える1つの数字を生み出しますが、監査できません。絞り込みの後に透明な輸送物性の順位づけを置く方式は監査できます。

コード例9: 制約による絞り込みと順位づけによる溶媒スクリーニング
    
    
    """例9: 輸送物性を考慮した超臨界溶媒のスクリーニング。"""
    import numpy as np
    import CoolProp.CoolProp as CP
    
    k_B = 1.380649e-23
    N_A = 6.02214076e23
    r_A = (3.0 * 147.6e-6 / (4.0 * np.pi * N_A)) ** (1.0 / 3.0)
    
    # 計算できない属性は表として与えるしかない。CoolPropが供給できる量とは
    # 分けて管理し、どの数値が判断であり、どれが参照品質の物性値なのかを
    # 一目でわかるようにしておく。
    CANDIDATES = {
        'CO2':      dict(polarity='non-polar', flammable=False, cost='low',
                         cosolvent=True,
                         notes='GRAS、不燃、減圧で容易に分離'),
        'Ethanol':  dict(polarity='polar',     flammable=True,  cost='medium',
                         cosolvent=False,
                         notes='再生可能、食品グレード、不活性化が必要'),
        'Water':    dict(polarity='polar',     flammable=False, cost='very low',
                         cosolvent=False,
                         notes='Tc以上で腐食性、耐食合金製の容器が必要'),
        'Propane':  dict(polarity='non-polar', flammable=True,  cost='low',
                         cosolvent=False,
                         notes='CO2より脂質をよく溶かすが爆発リスクあり'),
        'Nitrogen': dict(polarity='non-polar', flammable=False, cost='low',
                         cosolvent=False,
                         notes='不活性だがTc = -147 °C'),
        'R134a':    dict(polarity='weak',      flammable=False, cost='high',
                         cosolvent=False,
                         notes='GWP 1430、規制で削減対象'),
    }
    
    POLARITY_OK = {
        'non-polar': {'non-polar', 'weak'},
        'polar':     {'polar'},
    }
    
    
    def screen(max_T_C, required_polarity, allow_flammable,
               allow_cosolvent=True, Tr=1.05, Pr=2.0):
        """候補を可否判定で絞り込み、残った候補を輸送物性で順位づけする。
    
        制約は通過／不通過の二値、順位づけは計算物性のみを使う。
        """
        passed, rejected = [], []
        for fluid, meta in CANDIDATES.items():
            Tc = CP.PropsSI('Tcrit', fluid)
            Pc = CP.PropsSI('pcrit', fluid)
            T_op, P_op = Tr * Tc, Pr * Pc
            reasons, flags = [], []
    
            if Tc < 273.15:
                reasons.append('cryogenic critical temperature')
            if T_op - 273.15 > max_T_C:
                reasons.append(f'operating T {T_op-273.15:.0f} C > limit {max_T_C} C')
            if meta['flammable'] and not allow_flammable:
                reasons.append('flammable')
            if meta['polarity'] not in POLARITY_OK[required_polarity]:
                if required_polarity == 'polar' and meta['cosolvent'] and allow_cosolvent:
                    flags.append('needs a polar co-solvent (5-10 mol% ethanol)')
                else:
                    reasons.append(f"{meta['polarity']} solvent for a "
                                   f"{required_polarity} solute")
            if reasons:
                rejected.append((fluid, reasons))
                continue
            try:
                rho = CP.PropsSI('D', 'T', T_op, 'P', P_op, fluid)
                eta = CP.PropsSI('V', 'T', T_op, 'P', P_op, fluid)
            except ValueError:
                rejected.append((fluid, ['no reference transport model available']))
                continue
            D12 = k_B * T_op / (6 * np.pi * eta * r_A)
            passed.append(dict(fluid=fluid, T=T_op - 273.15, P=P_op / 1e6, rho=rho,
                               eta=eta, D12=D12, Sc=(eta / rho) / D12,
                               cost=meta['cost'], flags=flags))
    
        passed.sort(key=lambda r: -r['D12'])
    
        print(f"  制約: T <= {max_T_C} C, 溶質は{required_polarity}, "
              f"可燃性溶媒は{'許容' if allow_flammable else '不可'}, "
              f"共溶媒は{'許容' if allow_cosolvent else '不可'}")
        print(f"  評価状態: Tr = {Tr}, Pr = {Pr}")
        if passed:
            print(f"  {'rank':>4s} {'fluid':10s} {'T (C)':>7s} {'P (MPa)':>8s} "
                  f"{'rho':>8s} {'eta':>8s} {'D (1e-8)':>9s} {'Sc':>6s} {'cost':>9s}")
            for i, r in enumerate(passed, 1):
                print(f"  {i:4d} {r['fluid']:10s} {r['T']:7.1f} {r['P']:8.2f} "
                      f"{r['rho']:8.1f} {r['eta']*1e6:8.2f} {r['D12']*1e8:9.3f} "
                      f"{r['Sc']:6.2f} {r['cost']:>9s}")
                for f in r['flags']:
                    print(f"       -> {f}")
        else:
            print("  制約を満たす候補はありません。")
        for fluid, reasons in rejected:
            print(f"  rejected  {fluid:10s} -- {'; '.join(reasons)}")
        return passed, rejected
    
    
    print("=== ケース1: 熱に弱い天然物、無極性溶質、可燃性溶媒は不可 ===")
    screen(max_T_C=80, required_polarity='non-polar', allow_flammable=False)
    
    print()
    print("=== ケース2: ポリフェノール抽出、極性溶質、可燃性溶媒は不可 ===")
    screen(max_T_C=120, required_polarity='polar', allow_flammable=False)
    
    print()
    print("=== ケース3: 脂質抽出、可燃性溶媒を許容 ===")
    screen(max_T_C=150, required_polarity='non-polar', allow_flammable=True)
    
    print()
    print("=== ケース4: 水熱酸化、極性溶質、高温を許容 ===")
    screen(max_T_C=500, required_polarity='polar', allow_flammable=False,
           allow_cosolvent=False)
    
    print()
    print("ケース1で残るのは1つだけであり、これが正直な答えです。超臨界CO2が")
    print("工業的に支配的なのは、商業的に効いてくる制約が、輸送物性を比較する")
    print("以前に他の候補をすべて排除してしまうからです。輸送物性が結論を決める")
    print("のは、ケース3のように2つ以上の候補が制約を通過した場合だけです。")

=== ケース1: 熱に弱い天然物、無極性溶質、可燃性溶媒は不可 === 制約: T <= 80 C, 溶質はnon-polar, 可燃性溶媒は不可, 共溶媒は許容 評価状態: Tr = 1.05, Pr = 2.0 rank fluid T (C) P (MPa) rho eta D (1e-8) Sc cost 1 CO2 46.2 14.75 727.2 60.43 0.997 8.33 low rejected Ethanol -- operating T 267 C > limit 80 C; flammable; polar solvent for a non-polar solute rejected Water -- operating T 406 C > limit 80 C; polar solvent for a non-polar solute rejected Propane -- operating T 115 C > limit 80 C; flammable rejected Nitrogen -- cryogenic critical temperature rejected R134a -- operating T 120 C > limit 80 C === ケース2: ポリフェノール抽出、極性溶質、可燃性溶媒は不可 === 制約: T <= 120 C, 溶質はpolar, 可燃性溶媒は不可, 共溶媒は許容 評価状態: Tr = 1.05, Pr = 2.0 rank fluid T (C) P (MPa) rho eta D (1e-8) Sc cost 1 CO2 46.2 14.75 727.2 60.43 0.997 8.33 low -> needs a polar co-solvent (5-10 mol% ethanol) rejected Ethanol -- operating T 267 C > limit 120 C; flammable rejected Water -- operating T 406 C > limit 120 C rejected Propane -- flammable; non-polar solvent for a polar solute rejected Nitrogen -- cryogenic critical temperature; non-polar solvent for a polar solute rejected R134a -- weak solvent for a polar solute === ケース3: 脂質抽出、可燃性溶媒を許容 === 制約: T <= 150 C, 溶質はnon-polar, 可燃性溶媒は許容, 共溶媒は許容 評価状態: Tr = 1.05, Pr = 2.0 rank fluid T (C) P (MPa) rho eta D (1e-8) Sc cost 1 Propane 115.2 8.50 343.7 42.74 1.715 7.25 low 2 R134a 119.8 8.12 786.6 64.92 1.142 7.23 high 3 CO2 46.2 14.75 727.2 60.43 0.997 8.33 low rejected Ethanol -- operating T 267 C > limit 150 C; polar solvent for a non-polar solute rejected Water -- operating T 406 C > limit 150 C; polar solvent for a non-polar solute rejected Nitrogen -- cryogenic critical temperature === ケース4: 水熱酸化、極性溶質、高温を許容 === 制約: T <= 500 C, 溶質はpolar, 可燃性溶媒は不可, 共溶媒は不可 評価状態: Tr = 1.05, Pr = 2.0 rank fluid T (C) P (MPa) rho eta D (1e-8) Sc cost 1 Water 406.3 44.13 527.2 62.19 2.061 5.72 very low rejected CO2 -- non-polar solvent for a polar solute rejected Ethanol -- flammable rejected Propane -- flammable; non-polar solvent for a polar solute rejected Nitrogen -- cryogenic critical temperature; non-polar solvent for a polar solute rejected R134a -- weak solvent for a polar solute ケース1で残るのは1つだけであり、これが正直な答えです。超臨界CO2が 工業的に支配的なのは、商業的に効いてくる制約が、輸送物性を比較する 以前に他の候補をすべて排除してしまうからです。輸送物性が結論を決める のは、ケース3のように2つ以上の候補が制約を通過した場合だけです。

#### スクリーニング手順が実際に示していること

  * **ケース1では残るのがちょうど1つ。** 熱に弱い非極性の対象で可燃性溶媒が使えない場合、制約を通過するのはCO₂だけです。これは道具の欠陥ではなく、工業的な超臨界応用の圧倒的多数を超臨界CO₂が占めている理由そのものです。
  * **ケース2は共溶媒という道を浮かび上がらせる。** 120 °Cの上限の下では、極性溶質に対してエタノールと水が温度で排除されるため、答えは別の流体ではなくCO₂と極性モディファイアの組み合わせになります。商業的なポリフェノール抽出がまさにこれを行っています。
  * **ケース3は輸送物性が決める場面。** 可燃性溶媒を許容すると3つの候補が制約を通過し、その順位づけは真に輸送物性による順位づけになります。拡散係数でプロパンが1位、しかもCO₂の14.8 MPaに対し8.5 MPaで済みます。
  * **支配しているのは制約。** 4つのケースを通じて、輸送物性によって排除された候補は1つもありません。排除の理由は温度、可燃性、極性、データ欠如です。輸送物性は実現可能なプロセスがどれだけうまく動くかを教えますが、どのプロセスが実現可能かは教えてくれません。

* * *

## まとめ

### 重要なポイント

**1\. 輸送係数の構造**

  * $\eta = \eta_0(T) + \Delta\eta(\rho) + \eta_c$。$\lambda$ にも同じ分解が成り立つ。
  * Lucas式は臨界定数と双極子モーメントから $\eta_0$ を1-4%で与える。
  * 残差項は140 Kの範囲で1-3%の精度で密度の関数であり、運転条件では支配的。

**2\. 臨界異常は3つで同じではない**

  * 粘度：使えるほどの異常はない。臨界点を滑らかに通過する。
  * 熱伝導率：$T_c$ より0.03 K高い $\rho_c$ で37倍。$T_r = 1.03$ で4倍まで減衰し、圧力走査での局所スパイクは40 °Cで消える。
  * 温度拡散率：$2\times10^{-10}$ m²/sまで崩壊する（臨界減速）。$Pr$ は322に達する。
  * 相互拡散：混合物の臨界軌跡でゼロになるため、$T/\eta$ 型の相関式はそこでは定性的に誤り。

**3\. 拡散係数と物質移動**

  * $D \propto T/\eta$。圧力一定で温度を上げると両方の項を通じて $D$ が上がる。
  * 超臨界CO₂のシュミット数は3-17。ヘキサンは241、エタノールは2650。この比こそが、拡散係数の比ではなく、超臨界物質移動の定量的な論拠。
  * 「液体の10-100倍」は低密度側（8 MPaで14倍）で成立し、20 MPaでは3.9倍まで縮む。

**4\. 実際の層では**

  * Ergun圧力損失は圧力に対しほぼ平坦（8-30 MPaで0.058-0.077 kPa/m）で、液体溶媒より4-13倍低い。
  * 律速は粒子内拡散（$Bi_m$ = 34-44）であり、流量よりも粒径がはるかに効く。
  * 微粉化の代償は $1/d_p^2$ の圧力損失。粒径を決めるのは物質移動ではなくこのトレードオフ。

**5\. 溶媒選択**

  * $T_r$ と $P_r$ を揃えると、無関係な7流体の粘度は2.3倍、$Pr$ は2.2倍の幅に収まる。対応状態原理は働く。
  * 熱伝導率で水は外れ値（CO₂の5倍）。$T_c$ 以上でも水素結合が生き残るため。
  * 選択を決めるのは絶対温度・絶対圧力、安全性、規制、コスト、データの入手可能性であり、輸送物性ではない。輸送物性は、実現可能な選択がどれだけうまく機能するかを決める。

第6章が平衡状態を、第7章が計算の道具を、そして本章が速度を与えたことで、本シリーズの定量的な半分が完成しました。超臨界プロセスがどこに位置し、どれだけ速く動き、どの流体で運転すべきかを計算できるようになりました。

* * *

**演習問題**

#### 演習1: Lucas式を高圧域へ拡張する

Lucas法には、希薄気体の結果に $T_r$ と $P_r$ から作った補正係数を掛ける高圧拡張があります。Polingらの文献で調べて実装し、40 °CのCO₂について8-30 MPaでCoolPropと比較してください。どこで破綻し、その破綻は $\rho/\rho_c$ と相関するのか、それとも $P_r$ と相関するのかを論じてください。

#### 演習2: 極性流体の残差粘度

例2を水とエタノールについて繰り返してください。残差粘度はCO₂と同じくらいきれいに密度の曲線に重なりますか。$\rho = \rho_c$ と $\rho = 2\rho_c$ でのばらつきを定量化し、差があれば水素結合の観点から説明してください。

#### 演習3: 共溶媒のコスト

CO₂に10 mol%のエタノールを加えると混合物の粘度が上がります。CoolPropのHEOSバックエンドが動く範囲で（動かない場合は第7章を参照して、その事実を記録して）粘度の増加分と、それによるStokes-Einstein拡散係数の低下を見積もってください。粒子内律速を仮定し、それを補うために必要な抽出時間の増加率として結果を表してください。

#### 演習4: 増大が消える境界を求める

例4では35 °Cまで熱伝導率の局所ピークが見つかり、40 °Cでは見つかりませんでした。温度について二分探索し、局所極大が消える等温線を0.1 °Cの精度で求めてください。その温度を $T_r$ に換算し、$\rho_c$ における増大係数が4を下回る $T_r$ と比較してください。

#### 演習5: 伝熱劣化

Dittus-Boelter式 $Nu = 0.023 Re^{0.8} Pr^{0.4}$ を用いて、8 MPa、内径2 mmの管内のCO₂について、質量流束を500 kg/(m²·s)に固定したまま主流温度を擬臨界点を横切って変化させ、$Nu$ を計算してください。熱伝達係数を温度に対してプロットし、相関式が極大を予測する位置と、なぜそこでこの相関式を信用してはいけないのかを説明してください。

#### 演習6: 最適粒径

例6と例7を1つの最適化にまとめてください。送液動力の上限を制約として、粒径を決定変数に総抽出時間を最小化します。目的関数には粒子内時間定数を、制約にはErgunの圧力損失を用いてください。最適解は仮定した屈曲度にどれだけ敏感ですか。

#### 演習7: 調査対象に流体を追加する

例8にエタン、アンモニア、メタノールを追加してください。元の7流体で見つかった2.3倍の粘度帯の内側に収まりますか。アンモニアは強く水素結合します。水と並ぶ熱伝導率の外れ値になりますか。なるとすれば、どの程度でしょうか。

* * *

## 参考文献とさらなる読み物

### 輸送物性の相関式

  * Poling, Prausnitz & O'Connell, _The Properties of Gases and Liquids_ , 5th ed. - 第9-11章が粘度、熱伝導率、拡散を扱い、本章で用いたLucas法とWilke-Chang法を含む
  * Bird, Stewart & Lightfoot, _Transport Phenomena_ , 2nd ed. - 構成方程式と無次元数の枠組み
  * Wakao & Kaguei, _Heat and Mass Transfer in Packed Beds_ \- 充填層のシャーウッド相関式とその適用範囲

### 超臨界流体の輸送物性に特化した文献

  * Sengers & Sengers, _Thermodynamic Behavior of Fluids near the Critical Point_ \- $\lambda_c$ の背後にある臨界増大の定式化
  * Vesovicらおよび Huberら - CoolPropが実装しているCO₂の粘度・熱伝導率の参照相関式
  * Funazukuri, Kong & Kagei - 超臨界CO₂中の二成分拡散係数の実測
  * Catchpole & King、He & Yu - 超臨界CO₂に特化してフィットされた拡散係数の相関式

### ツールとデータ

  * [CoolProp](<http://www.coolprop.org/>) \- `V` と `L` 出力による参照輸送物性
  * [NIST Chemistry WebBook, Thermophysical Properties of Fluid Systems](<https://webbook.nist.gov/chemistry/fluid/>)
  * [NIST REFPROP](<https://www.nist.gov/srd/refprop>) \- これらの相関式の参照実装

* * *
