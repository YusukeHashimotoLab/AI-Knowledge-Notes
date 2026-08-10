---
title: "第6章: 熱力学と状態方程式"
chapter_title: "第6章: 熱力学と状態方程式"
subtitle: 立方型状態方程式、臨界現象、相平衡、フガシティ
---

第1章から第5章では超臨界流体を定性的に説明してきました。本章ではそれを定量化する道具立てを提供します。密度と圧縮率を予測する状態方程式、臨界点近傍の異常を支配するスケーリング則、何が何に溶けるかを決める相平衡の条件、そしてこれらすべてを具体的な数値に変換するフガシティの関係式です。8つのPythonコード例により、本章のすべての結果を自分の手で再現できます。

## 学習目標

本章を修了すると、以下のことができるようになります：

  * 理想気体の法則が臨界点近傍で破綻する理由を説明し、圧縮因子 $Z$ でそのずれを定量化できる
  * van der Waalsパラメータ $a$、$b$ の物理的意味を理解し、そこから臨界定数と $Z_c = 3/8$ を導出できる
  * Peng-Robinson状態方程式で密度と圧縮因子を計算し、偏心因子 $\omega$ の役割を説明できる
  * 臨界乳光と $\kappa_T$、$C_P$ の発散を普遍的な臨界指数で記述できる
  * 音速が臨界点で極小値を取る理由と、それがプロセス制御に及ぼす影響を説明できる
  * 二成分系相図を分類（Type I、II、III）し、Chrastil式で固体溶解度をモデル化できる
  * 立方型状態方程式からフガシティ係数を計算し、相平衡条件 $\phi_i^L x_i = \phi_i^V y_i$ を適用できる
  * 二成分相互作用パラメータ $k_{ij}$ を含む混合則を多成分系に適用できる

* * *

## 6.1 超臨界流体の状態方程式

### 理想気体の法則の限界

理想気体の状態方程式

$$ PV = nRT $$ 

は、分子を体積の無視できる点粒子とみなし、分子間力が存在せず、衝突はすべて弾性的であると仮定しています。しかし超臨界流体が実際に使われる条件では、これらの仮定はいずれも成り立ちません。

  1. **高圧下** ：分子間距離が縮まり、van der Waals力による引力が無視できなくなる。
  2. **高密度** ：分子自身が占める排除体積が全体積に対して無視できない割合になる。
  3. **臨界点近傍** ：気液の区別が消失し、密度ゆらぎが極端に大きくなる。

物性 | 理想気体の予測 | 超臨界流体の実際  
---|---|---  
圧縮率 | 一定 | $T_c$ で発散  
密度の圧力依存性 | $P$ に比例（線形） | 強い非線形性  
相転移 | 記述できない | 明確な気液平衡境界が存在  
溶媒力 | $P$ に比例 | 分子クラスター形成により増強  
  
理想性からのずれは**圧縮因子** で定量化されます。

$$ Z = \frac{PV}{nRT} = \frac{PM}{\rho RT} $$ 

ここで $M$ はモル質量です。理想気体では $Z = 1$ ですが、臨界点近傍では $Z$ は概ね 0.2 から 1.2 の範囲で変化します。

### van der Waals方程式

van der Waals（1873年）は、分子の大きさと分子間引力の2つの補正を理想気体の式に加えました。

$$ \left(P + \frac{a}{V_m^2}\right)(V_m - b) = RT $$ 

ここで $V_m$ はモル体積（m³/mol）、$a$ は引力パラメータ（Pa·m⁶·mol⁻²）、$b$ は排除体積パラメータ（m³/mol）です。

#### 2つの補正項の物理的意味

  * **圧力の補正 $a/V_m^2$** ：分子間引力が内部圧力を生じさせるため、実測される外部圧力は理想値より小さくなります。$1/V_m^2$ の依存性は2体相互作用に由来します（分子対の数は $N^2$ に比例）。$a$ が大きいほど引力が強く、極性分子や大きな分子で顕著です。
  * **体積の補正 $-b$** ：各分子は自身の周囲に他の分子が入れない体積を作り、利用可能な空間を減らします。半径 $r$ の剛体球では $b \approx 4 \times \frac{4}{3}\pi r^3$、つまり分子体積の約4倍になります。

圧力について解くと、等温線を描くのに便利な形になります。

$$ P = \frac{RT}{V_m - b} - \frac{a}{V_m^2} $$ 

等温線の形状は温度によって次のように変わります。

  * **$T > T_c$（超臨界領域）**：単調減少し、相転移は現れない
  * **$T = T_c$（臨界等温線）** ：水平な変曲点をもつ
  * **$T < T_c$（二相領域）**：S字型のループが現れ、その非物理的な部分はMaxwellの等面積則により水平な共存線に置き換えられる

**臨界点の条件。** 臨界点では等温線が水平な変曲点をもちます。

$$ \left(\frac{\partial P}{\partial V_m}\right)_{T_c} = 0, \quad \left(\frac{\partial^2 P}{\partial V_m^2}\right)_{T_c} = 0 $$ 

この2つの条件を連立して解くと、臨界定数が得られます。

$$ T_c = \frac{8a}{27Rb}, \quad P_c = \frac{a}{27b^2}, \quad V_{m,c} = 3b $$ 

これを逆に解けば、文献値の臨界定数から $a$ と $b$ を求められます。

$$ a = \frac{27R^2T_c^2}{64P_c}, \qquad b = \frac{RT_c}{8P_c} $$ 

**van der Waals方程式の普遍的な帰結。** 臨界定数を $Z$ の定義に代入すると、物質に依存しない値が得られます。

$$ Z_c = \frac{P_c V_{m,c}}{RT_c} = \frac{3}{8} = 0.375 $$ 

実在流体の $Z_c$ は約0.23〜0.31（CO₂では0.274）であり、van der Waals式は $Z_c$ を系統的に過大評価します。この1つの数値が、この方程式の限界を最も明確に示しています。教育用途には優れていますが、設計計算には信頼できません。

### Peng-Robinson方程式

Peng-Robinson方程式（1976年）は、炭化水素、CO₂、超臨界プロセス設計における実用上の標準的な立方型状態方程式です。

$$ P = \frac{RT}{V_m - b} - \frac{a\alpha(T)}{V_m^2 + 2bV_m - b^2} $$ 

ここで各パラメータは次のように与えられます。

$$ a = 0.45724\frac{R^2T_c^2}{P_c} $$ $$ b = 0.07780\frac{RT_c}{P_c} $$ $$ \alpha(T) = \left[1 + \kappa\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2 $$ $$ \kappa = 0.37464 + 1.54226\omega - 0.26992\omega^2 $$ 

**偏心因子** $\omega$ は、分子が球対称な単純流体からどれだけ離れているかを表す指標です。

$$ \omega = -\log_{10}\left(\frac{P^{sat}(T_r = 0.7)}{P_c}\right) - 1 $$  物質 | $\omega$ | 分子の特徴  
---|---|---  
Ar, Kr, Xe | 0.00 〜 0.01 | 球形の希ガス（単純流体）  
CH₄ | 0.011 | ほぼ球形  
CO₂ | 0.225 | 四重極モーメントをもつ直線分子  
n-ヘキサン | 0.301 | 柔軟な鎖状分子  
H₂O | 0.344 | 極性、水素結合  
長鎖アルカン | 0.5 以上 | 強い非球形性  
  
**立方形式。** 数値計算では $Z$ についての3次方程式に書き換えます。

$$ Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0 $$ 

ここで

$$ A = \frac{a\alpha P}{R^2T^2}, \quad B = \frac{bP}{RT} $$ 

二相領域の内部では、この3次方程式は3つの実正根をもちます。最大の根が気相、最小の根が液相に対応し、中間の根は熱力学的に不安定で物理的意味をもちません。単相の超臨界領域では実正根は1つだけになります。

**重い分子への適用。** 上記の $\kappa$ の相関式は $\omega \le 0.49$ に対してフィッティングされたものです。より重い化合物に対しては、Peng とRobinsonが後に拡張形を提示しています。

$$ \kappa = 0.379642 + 1.48503\omega - 0.164423\omega^2 + 0.016666\omega^3 \quad (\omega > 0.49) $$ 

重い溶質に軽分子用の相関式を使ってしまうのは、溶解度計算で頻繁に起こり、しかも気づきにくい誤差の原因です。

### 状態方程式の比較

状態方程式 | 精度 | 計算コスト | 適した用途  
---|---|---|---  
理想気体 | $T_c$ 近傍では不可 | 最小 | $T \gg T_c$、$P < 1$ MPa  
van der Waals | 定性的な理解のみ | 低 | 教育、概念理解  
Peng-Robinson | 良好（密度で5〜10%） | 中 | 炭化水素、CO₂、プロセス設計  
Soave-Redlich-Kwong | 良好（PRと同程度） | 中 | 石油産業  
SAFT系（PC-SAFT等） | 優秀（1〜3%） | 高 | 複雑分子、高分子、会合性流体  
  
**$T = 1.05T_c$、$P = 1.5P_c$ における典型的な密度予測誤差** （超臨界CO₂抽出の多くがこの領域で運転されます）：

  * van der Waals：15〜25%
  * Peng-Robinson：5〜10%
  * PC-SAFT：1〜3%

溶解度は密度のおよそ4〜10乗に比例するため（6.4節のChrastil式を参照）、20%の密度誤差は溶解度で一桁の誤差になります。状態方程式の選択は見かけ上の問題ではありません。

* * *

## 6.2 臨界現象

### 臨界乳光

臨界点に近づくと、透明だった流体が乳白色に濁ります。これが**臨界乳光** であり、臨界点で何か特異なことが起きていることを最も直接的に示す視覚的証拠です。

  1. **密度ゆらぎ** ：$T_c$ 近傍ではゆらぎを生じるための自由エネルギー的コストがゼロに近づくため、局所密度が大きな振幅で長距離にわたってゆらぐ。
  2. **相関長** ：相関したゆらぎの空間的広がり $\xi$ が次のように発散する。 $$ \xi \sim |T - T_c|^{-\nu} $$ ここで $\nu \approx 0.63$ は相関長の臨界指数。
  3. **光散乱** ：$\xi$ が可視光の波長（$\lambda \sim 400$〜$700$ nm）と同程度になると、Rayleigh散乱が劇的に強まる。

散乱強度は次のように増大します。

$$ I \sim \frac{\xi^6}{\lambda^4} \sim |T - T_c|^{-6\nu} \approx |T - T_c|^{-3.8} $$ 

これが、$T_c$ から数ケルビン離れれば超臨界流体が完全に透明に見え、$T_c$ の1ケルビン以内では不透明になる理由です。

### 応答関数の発散

**等温圧縮率：**

$$ \kappa_T = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T = \frac{1}{\rho}\left(\frac{\partial \rho}{\partial P}\right)_T \sim |T - T_c|^{-\gamma} $$ 

ここで $\gamma \approx 1.24$ です。**定圧熱容量：**

$$ C_P = T\left(\frac{\partial S}{\partial T}\right)_P \sim |T - T_c|^{-\alpha} $$ 

ここで $\alpha \approx 0.11$ です。

#### 発散が実験室で意味すること

  * $\kappa_T$ の発散：わずかな圧力変化で体積（密度）が大きく変わります。これが「ピストン効果」や流動ループにおける密度振動の起源です。
  * $C_P$ の発散：わずかな温度変化で大量の熱を吸収するため、熱応答が鈍く、温度制御はオーバーシュートしやすくなります。

いずれも、超臨界プロセスを $T_c$ 上ではなく意図的に $T_c$ から離した条件で運転する理由です。

### 普遍性と臨界指数

驚くべきことに、CO₂、水、キセノンという構造的に何の共通点もない分子が、同じ臨界指数を共有します。これが**普遍性** です。

指数 | 記述する量 | 3次元Ising模型の値  
---|---|---  
$\alpha$ | 熱容量 | 0.110  
$\beta$ | 秩序変数（密度差） | 0.326  
$\gamma$ | 感受率（圧縮率） | 1.237  
$\delta$ | 臨界等温線 | 4.789  
$\nu$ | 相関長 | 0.630  
  
流体における**秩序変数** は、共存する2相の密度差です。

$$ \Delta\rho = \rho_L - \rho_V \sim |T - T_c|^{\beta} $$ 

また**臨界等温線** （$T = T_c$）上では次が成り立ちます。

$$ |P - P_c| \sim |\rho - \rho_c|^{\delta} $$ 

これらの指数は独立ではなく、**スケーリング関係** で結ばれています。

$$ \alpha + 2\beta + \gamma = 2 \quad \text{(Rushbrooke)} $$ $$ \gamma = \beta(\delta - 1) \quad \text{(Widom)} $$ 

### くりこみ群理論

**くりこみ群** 理論（Wilson, 1971年）は普遍性を説明します。その中心的な考え方は、臨界点では系が長さスケールの変換に対して不変になり、短距離の詳細が次々と洗い流されていくというものです。最終的に残るのは次の3つだけです。

  * 空間次元（流体では $d = 3$）
  * 秩序変数の次元（スカラー量である密度では $n = 1$）
  * 相互作用の対称性

分子量や結合角、化学種の違いは、専門用語の意味で「無関係（irrelevant）」になります。実用上の見返りは大きく、1つの流体の臨界挙動を丁寧に測定すれば、すべての流体について知ったことになります。これは対応状態原理の理論的基礎でもあり、単一の換算物性チャートが多くの物質に使える理由でもあります。

* * *

## 6.3 臨界点近傍の熱力学的性質

### エンタルピーとエントロピー

理想的な挙動からのずれは残差項の積分としてまとめられます。

$$ H(T, P) = H^{ideal}(T) + \int_0^P \left[V - T\left(\frac{\partial V}{\partial T}\right)_P\right] dP $$ $$ S(T, P) = S^{ideal}(T, P) - \int_0^P \left(\frac{\partial V}{\partial T}\right)_P dP $$ 

$T_c$ 近傍ではこれらの積分項が大きく、かつ強く非線形になります。熱膨張係数 $(\partial V/\partial T)_P$ 自体が発散するためです。

臨界点では共存する2相が同一になるため、蒸発潜熱と蒸発エントロピーはいずれもゼロに収束します。

$$ \Delta H_{vap} = H_V - H_L \sim |T - T_c|^{\beta} \to 0, \qquad \Delta S_{vap} = S_V - S_L \to 0 $$ 

その結果、Clausius-Clapeyronの式は不定形になります。

$$ \frac{dP}{dT} = \frac{\Delta S_{vap}}{\Delta V_{vap}} \to \frac{0}{0} $$ 

これは、蒸気圧曲線が超臨界領域へ延びていくのではなく、臨界点で文字どおり _終端する_ ことの熱力学的な表現です。

**実用上の帰結** ：超臨界プロセスの熱交換器は、狭い温度範囲で生じる非常に大きなエンタルピー変化を前提に設計しなければなりません。

### 熱容量の異常

定圧熱容量

$$ C_P = \left(\frac{\partial H}{\partial T}\right)_P $$ 

は臨界点で鋭いピークを示します。$P \approx P_c$ におけるCO₂の場合：

温度 | 条件 | $C_P$ [J/(mol·K)]  
---|---|---  
295 K | $T_c$ より約10 K低い（液相） | 約 180  
304.1 K | $T_c$ | → ∞（理論上は発散）  
313 K | $T_c$ より約10 K高い | 約 70  
  
物理的には、$T_c$ 近傍で加えた熱は温度上昇ではなく分子構造の再編成（クラスターの生成と解裂）に使われます。定積熱容量 $C_V$ も異常を示しますが、$C_P$ よりはるかに緩やかです。

熱容量比は標準的な熱力学関係から得られます。

$$ \frac{C_P}{C_V} = \frac{\kappa_T}{\kappa_S} = 1 + \frac{TV\alpha_P^2}{\kappa_T C_V} $$ 

ここで $\alpha_P = (1/V)(\partial V/\partial T)_P$ です。臨界点では両方の熱容量が発散しますが、その比は有限に留まり1に近づきます。

### 音速

音速は断熱微分で与えられます。

$$ c = \sqrt{\left(\frac{\partial P}{\partial \rho}\right)_S} = \sqrt{\frac{1}{\rho\kappa_S}} $$ 

実在流体では次のように書けます。

$$ c = \sqrt{\frac{C_P}{C_V}\left(\frac{\partial P}{\partial \rho}\right)_T} $$ 

臨界点近傍では $\kappa_T$ の発散により $(\partial P/\partial \rho)_T \to 0$ となり、一方で $C_P/C_V \to 1$、断熱圧縮率 $\kappa_S$ は有限に留まります。その結果、音速は $T_c$ で**極小値** を取ります。$P_c$ におけるCO₂では：

  * $T = 290$ K（液相）：$c \approx 900$ m/s
  * $T = 304$ K（$T_c$）：$c \approx 180$ m/s — 極小値
  * $T = 320$ K（超臨界）：$c \approx 250$ m/s

**プロセス設計上の帰結** ：伝播時間から距離を換算する超音波式の液面計や流量計は、仮定した音速が成り立たなくなるため、臨界点近傍では信頼できません。

### プロセス設計への影響
    
    
    ```mermaid
    graph TD
        A[臨界点の熱力学]
        A --> B[プロセス上の課題]
        A --> C[プロセス上の利点]
    
        B --> B1[温度制御の困難さ]
        B --> B2[圧力損失の問題]
        B --> B3[伝熱の制約]
        B --> B4[密度振動]
    
        C --> C1[調整可能な溶媒力]
        C --> C2[高い物質移動速度]
        C --> C3[迅速な相分離]
        C --> C4[選択的抽出]
    
        style A fill:#e0f7fa
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

#### 設計指針

  * 臨界点から意図的に距離を取って運転する：$T_r = 1.05$〜$1.20$、$P_r = 1.1$〜$2.0$ であれば、調整可能性の大部分を保ちながら発散の悪影響を回避できます。
  * 温度と圧力の両方に十分な制御マージンを確保します。
  * 過渡応答は動的シミュレーションで評価します。定常計算では、起動・停止時に支配的となる密度振動が見えません。

* * *

## 6.4 超臨界流体系における相平衡

### 気液平衡

亜臨界領域では蒸気圧曲線に沿って2相が共存します。平衡条件は化学ポテンシャルの一致、すなわちフガシティの一致です。

$$ \mu_L(T, P) = \mu_V(T, P) \quad \Longleftrightarrow \quad f_L(T, P) = f_V(T, P) $$ 

蒸気圧曲線の傾きは**Clausius-Clapeyronの式** で与えられます。

$$ \frac{dP^{sat}}{dT} = \frac{\Delta H_{vap}}{T \Delta V} = \frac{\Delta H_{vap}}{T(V_V - V_L)} $$ 

蒸気を理想気体とみなし、蒸発潜熱が温度に依らないと仮定すると、よく知られた積分形 $\ln P = -\Delta H_{vap}/(RT) + C$ が得られます。実務では経験式である**Antoine式** が用いられます。

$$ \log_{10} P^{sat} = A - \frac{B}{C + T} $$ 

ここで $A$、$B$、$C$ は物質固有の定数で、適用温度範囲と単位系を明記して表にまとめられています。$T \to T_c$ では $\Delta H_{vap} \to 0$、$V_V - V_L \to 0$ となり、曲線は臨界点で終端します。

### 超臨界流体を含む二成分系相図

二成分系（超臨界溶媒＋溶質）の相挙動は $P$-$T$-$x$ 曲面上に表されます。全体像を整理する鍵は次の2つです。

  * **臨界曲線（critical locus）** ：純成分1の臨界点と純成分2の臨界点を結ぶ曲線。単調である必要はありません。
  * **三相線** ：固体・液体・気体が共存する線で、純成分の三重点から延びています。

古典的なvan Konynenburg-Scott分類はいくつかの型を区別しますが、超臨界プロセスで重要なのは次の3つです。

型 | 代表的な系 | 特徴的な挙動  
---|---|---  
Type I | CO₂ + 軽質アルカン | 2つの純成分臨界点を連続的に結ぶ臨界曲線。共沸も液液不溶性もない  
Type II | CO₂ + 重質アルカン | 臨界曲線が温度の極大と圧力の極小をもつ。液液気の三相領域が現れうる  
Type III | CO₂ + 重質炭化水素、高分子 | 臨界曲線が高圧側へ発散する。中程度の温度で液液不溶性を示す  
  
Type IIIの挙動は厄介な現象ではなく道具です。そこに現れる強い圧力感受性こそが、選択性の高い超臨界分別を可能にしています。

### 溶解度のモデル化：Chrastil式

Chrastil（1982年）は、超臨界流体中の固体溶解度に対する半経験的な相関式を提案しました。

$$ \ln S = k \ln \rho + \frac{a}{T} + b $$ 

ここで $S$ は溶解度、$\rho$ は超臨界流体の密度（kg/m³）、$k$ は会合数、$a$ は溶媒化熱と蒸発熱に関係する定数（K）、$b$ は定数です。$S$ は文献によって単位体積あたり（kg/m³）または単位質量あたり（kg/kg）で報告されており、その違いはフィッティングされた $b$ が吸収してしまいます。公表パラメータを再利用する前に必ず単位を確認してください。

#### 会合数 $k$ の解釈

  * 小さな分子（カフェイン、ニコチン）では $k \approx 2$〜6
  * 大きな分子（トリグリセリド、脂肪酸）では $k \approx 10$〜20

物理的には、$k$ は1個の溶質分子を溶媒化する超臨界流体分子の数、すなわち溶媒化クラスターの大きさです。同時に、$k$ はプロセスの圧力感受性そのものを決めます。

$$ \frac{\partial \ln S}{\partial \ln \rho}\bigg|_T = k, \qquad \frac{\partial \ln S}{\partial (1/T)}\bigg|_\rho = a $$ 

一定温度では、溶解度は密度（したがって圧力）とともに急激に増加します。一定密度では、溶媒化が発熱的であるため（$a < 0$）、通常は温度上昇とともに減少します。

**公表されたChrastilパラメータを次元チェックなしに再利用してはいけません。** $b$ は対数の内側にある単なる加法定数であるため、相関式におけるあらゆる単位系の約束事を暗黙のうちに吸収します。$\rho$ が kg/m³ か g/L か、$S$ が kg/kg か kg/m³ か g/L か。単位が明記されていないパラメータセットは使えませんし、誤った約束事に代入すれば何桁もずれることがあります。フィッティング済みパラメータを信用する前に、必ず論文中の実測点を1つ再現してみてください。可能な限り、生データから自分で再フィッティングするのが最善です。

### 逆行凝縮

**逆行凝縮** とは、一定圧力下で温度を _上げる_ と液相が現れるという直観に反する現象です（ガスコンデンセート貯留層では等温減圧でも同様のことが起こります）。

  1. 臨界圧力より高い単相の超臨界領域から出発する。
  2. 等圧のまま温度を上げる。
  3. 相境界を越え、液滴が現れる。
  4. さらに加熱すると液滴は再溶解して消える。

**分子論的な説明** ：低温側では超臨界流体の密度、したがって溶媒力が高く、すべてが溶解したままです。温度が上がると、溶質の蒸気圧の増加よりも密度の低下のほうが速く進むため、溶質が析出します。この現象は $T > T_c$、$P \approx P_c$ で、混合物の臨界曲線が負の勾配をもつ領域で起こります。

**重要となる場面** ：天然ガス処理（メタンからのコンデンセート回収）、超臨界逆溶媒析出法（SAS）、CO₂圧入による石油増進回収。

* * *

## 6.5 熱力学計算

### フガシティとフガシティ係数

**フガシティ** $f$ は、化学ポテンシャルの理想気体的表現を厳密に成り立たせる「実効圧力」です。

$$ d\mu = RT\, d\ln f, \qquad \mu = \mu^\circ(T) + RT\ln\frac{f}{f^\circ} $$ 

**フガシティ係数** は理想性からのずれを表します。

$$ \phi = \frac{f}{P} $$ 

理想気体では $\phi = 1$ です。任意の状態方程式から積分により求められます。

$$ \ln \phi = \int_{\infty}^{V_m} \left[\frac{P}{RT} - \frac{1}{V_m}\right] dV_m - \ln Z $$ 

Peng-Robinson方程式についてこの積分を実行すると閉じた表式が得られます。立方型状態方程式が今なお使われ続ける主要な理由の1つです。

$$ \ln \phi = (Z - 1) - \ln(Z - B) - \frac{A}{2\sqrt{2}B} \ln\left(\frac{Z + (1+\sqrt{2})B}{Z + (1-\sqrt{2})B}\right) $$ 

$A$ と $B$ は6.1節で定義したものです。液相と気相の間の平衡条件は次のように書けます。

$$ \phi_i^L x_i = \phi_i^V y_i $$ 

ここで $x_i$、$y_i$ はそれぞれ液相と気相のモル分率です。

### 超臨界流体相における化学ポテンシャル

混合物中の成分 $i$ について、

$$ \mu_i = \left(\frac{\partial G}{\partial n_i}\right)_{T,P,n_{j\neq i}} = \mu_i^0(T) + RT \ln\left(\frac{f_i}{f_i^0}\right), \qquad f_i = \phi_i(T, P, \\{x\\})\, x_i P $$ 

液体的な密度領域では、活量係数を用いるほうが便利なことが多いです。

$$ f_i = \gamma_i(T, P, \\{x\\})\, x_i f_i^{pure}(T, P) $$ 

固相と超臨界相における溶質の化学ポテンシャルを等置する条件 $\mu_i^{solid} = \mu_i^{SCF}$ から溶解度が直接得られ、無限希釈の極限では

$$ \ln \gamma_i^\infty = \ln \phi_i^\infty - \ln \phi_i^{sat} + \frac{v_i^L(P - P_i^{sat})}{RT} $$ 

となります。これにより純成分の物性のみから溶解度を予測できます。最後の項がPoynting補正です。

### 多成分系の混合則

混合物に対する立方型状態方程式のパラメータは、**van der Waals型の一流体混合則** を用いて純成分パラメータから構成します。

$$ a_m = \sum_i \sum_j x_i x_j a_{ij}, \qquad b_m = \sum_i x_i b_i $$ $$ a_{ij} = \sqrt{a_i a_j}\,(1 - k_{ij}) $$ 

**二成分相互作用パラメータ** $k_{ij}$ は、異種分子対に対する幾何平均近似を補正します。実験データからフィッティングされ、$k_{ii} = 0$ を満たし、値は小さいものの決して無視できません。

成分ペア | $k_{ij}$  
---|---  
CO₂ - エタノール | 0.10  
CO₂ - n-ヘキサン | 0.13  
CO₂ - 水 | 0.19  
  
$k_{ij}$ が正であることは、異種分子間の引力が幾何平均より弱いことを意味します（CO₂と炭化水素の組み合わせでは通常 $k_{ij} \approx 0.1$〜$0.15$）。負の値はまれです。より高度な混合則（Wong-Sandler、MHV2など）は活量係数モデルを混合則に組み込み、強い極性をもつ混合物に対してかなり高い精度を与えます。

#### 二相フラッシュ計算の手順

  1. $T$、$P$、および全体組成 $\\{z_i\\}$ を指定する。
  2. 液相組成 $\\{x_i\\}$ と気相組成 $\\{y_i\\}$ を仮定する。
  3. 各相について立方型状態方程式を解き、$\phi_i^L$ と $\phi_i^V$ を求める。
  4. 平衡関係から組成を更新する。
  5. すべての成分について $|\phi_i^L x_i - \phi_i^V y_i| < \epsilon$ となるまで反復する。

* * *

## 6.6 Pythonコード例

**実行環境。** 本節の8つの例はすべてNumPy、SciPy、Matplotlibのみで動作します。

`pip install numpy scipy matplotlib`

熱物性ライブラリは不要なので、上で示した式から結果をそのまま再現できます。CoolPropや`thermo`による参照品質の物性データは第7章で導入します。

グラフの軸ラベルには日本語を用いています。文字化けを避けるため、実行前に日本語フォントを指定してください（例: `plt.rcParams['font.family'] = 'Hiragino Sans'`、Linuxでは `'IPAexGothic'`）。

### 例1: van der Waals等温線

van der Waals方程式からCO₂の $P$-$V$ 等温線を、亜臨界・臨界・超臨界の各温度について描き、$Z_c = 3/8$ を数値的に確認します。

コード例1: CO₂のvan der Waals等温線
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # CO2のvan der Waalsパラメータ
    R = 8.314  # J/(mol·K)
    Tc = 304.1  # K
    Pc = 7.38e6  # Pa
    
    # 臨界定数からaとbを算出
    a = 27 * R**2 * Tc**2 / (64 * Pc)  # Pa·m^6/mol^2
    b = R * Tc / (8 * Pc)  # m^3/mol
    
    # モル体積の範囲（Vm = b の特異点を避ける）
    Vm = np.linspace(1.5*b, 50*b, 1000)
    
    # 温度リスト: 亜臨界、臨界、超臨界
    temperatures = [280, 304.1, 320, 350]
    colors = ['blue', 'red', 'green', 'purple']
    
    plt.figure(figsize=(10, 6))
    
    for T, color in zip(temperatures, colors):
        # van der Waals式による圧力
        P = R * T / (Vm - b) - a / Vm**2
    
        # プロット用にMPaへ変換
        P_MPa = P / 1e6
    
        label = f'T = {T} K'
        if T == Tc:
            label += '（臨界）'
    
        plt.plot(Vm * 1e6, P_MPa, color=color, linewidth=2, label=label)
    
    # 臨界点をマーク
    Vm_c = 3 * b
    P_c_vdw = R * Tc / (Vm_c - b) - a / Vm_c**2
    plt.plot(Vm_c * 1e6, P_c_vdw / 1e6, 'ro', markersize=10, label='臨界点')
    
    plt.xlabel('モル体積 (cm³/mol)', fontsize=12)
    plt.ylabel('圧力 (MPa)', fontsize=12)
    plt.title('CO₂のvan der Waals等温線', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(0, 500)
    plt.ylim(0, 15)
    plt.tight_layout()
    plt.show()
    
    print("CO2のvan der Waalsパラメータ:")
    print(f"a = {a:.4e} Pa·m^6/mol^2")
    print(f"b = {b:.4e} m^3/mol")
    print(f"臨界圧縮因子: Zc = {Pc * Vm_c / (R * Tc):.3f}")
    

CO2のvan der Waalsパラメータ: a = 3.6541e-01 Pa·m^6/mol^2 b = 4.2823e-05 m^3/mol 臨界圧縮因子: Zc = 0.375

**着目点：** 亜臨界の等温線（$T < T_c$）にはMaxwellの等面積則で水平な共存線に置き換えられるべき非物理的なS字ループが現れます。臨界等温線は水平な変曲点をもち、超臨界の等温線は単調です。出力された $Z_c = 0.375$ はちょうど $3/8$ です。

### 例2: 臨界点の決定

等温線の1階微分と2階微分が同時にゼロとなる点として臨界点を求め、解析解を数値的に検証します。

コード例2: 変曲条件からの臨界点決定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def van_der_waals_P(Vm, T, a, b, R=8.314):
        """van der Waals式による圧力。"""
        return R * T / (Vm - b) - a / Vm**2
    
    def dP_dVm(Vm, T, a, b, R=8.314):
        """圧力のモル体積に関する1階微分。"""
        return -R * T / (Vm - b)**2 + 2 * a / Vm**3
    
    def d2P_dVm2(Vm, T, a, b, R=8.314):
        """圧力のモル体積に関する2階微分。"""
        return 2 * R * T / (Vm - b)**3 - 6 * a / Vm**4
    
    def find_critical_point(a, b, R=8.314):
        """
        van der Waalsパラメータから臨界点を求める。
    
        Returns:
        --------
        Tc, Pc, Vm_c : float
            臨界温度、臨界圧力、臨界モル体積
        """
        # van der Waals式の解析解
        Tc = 8 * a / (27 * R * b)
        Vm_c = 3 * b
        Pc = a / (27 * b**2)
    
        return Tc, Pc, Vm_c
    
    def verify_critical_conditions(Vm_c, Tc, a, b):
        """
        臨界点で1階微分と2階微分が消えることを検証する。
    
        SI単位ではこれらの微分値が巨大なスケール因子を含むため
        （dP/dVm は Pc/Vm_c ~ 1e11 Pa per m^3/mol のオーダー）、
        絶対値による許容判定は意味をもたない。無次元化した残差で比較する。
        """
        dP = dP_dVm(Vm_c, Tc, a, b)
        d2P = d2P_dVm2(Vm_c, Tc, a, b)
    
        Pc = a / (27 * b**2)
        res1 = dP * Vm_c / Pc
        res2 = d2P * Vm_c**2 / Pc
    
        print("臨界点における検証:")
        print(f"  (dP/dVm)   * Vm_c/Pc   = {res1:.2e} （ほぼ0であるべき）")
        print(f"  (d2P/dVm2) * Vm_c^2/Pc = {res2:.2e} （ほぼ0であるべき）")
    
        return np.abs(res1) < 1e-9 and np.abs(res2) < 1e-9
    
    # 例: CO2の臨界点を求める
    R = 8.314
    a = 0.3658  # Pa·m^6/mol^2（実験データにフィッティングした値）
    b = 4.267e-5  # m^3/mol
    
    Tc, Pc, Vm_c = find_critical_point(a, b, R)
    
    print("CO2の臨界点の決定")
    print("=" * 50)
    print("van der Waalsパラメータ:")
    print(f"  a = {a:.4f} Pa·m^6/mol^2")
    print(f"  b = {b:.2e} m^3/mol")
    print("\n計算された臨界点:")
    print(f"  Tc = {Tc:.2f} K")
    print(f"  Pc = {Pc/1e6:.2f} MPa")
    print(f"  Vm,c = {Vm_c*1e6:.2f} cm^3/mol")
    print(f"  Zc = {Pc * Vm_c / (R * Tc):.4f}")
    print("\n実験値:")
    print("  Tc = 304.1 K")
    print("  Pc = 7.38 MPa")
    print("  Zc = 0.274")
    
    # 条件の検証
    is_critical = verify_critical_conditions(Vm_c, Tc, a, b)
    print(f"\n臨界点の条件を満たすか: {is_critical}")
    
    # 臨界温度近傍の等温線をプロット
    Vm_range = np.linspace(1.5*b, 10*b, 500)
    temps = [Tc - 5, Tc, Tc + 5]
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(10, 6))
    
    for T, color in zip(temps, colors):
        P_values = [van_der_waals_P(Vm, T, a, b, R) for Vm in Vm_range]
        label = f'T = {T:.1f} K'
        if T == Tc:
            label += '（臨界）'
        plt.plot(Vm_range*1e6, np.array(P_values)/1e6, color=color,
                 linewidth=2, label=label)
    
    # 臨界点をマーク
    Pc_calc = van_der_waals_P(Vm_c, Tc, a, b, R)
    plt.plot(Vm_c*1e6, Pc_calc/1e6, 'ko', markersize=10, label='臨界点')
    
    # 臨界点における接線（水平になるはず）
    tangent_Vm = np.linspace(0.9*Vm_c, 1.1*Vm_c, 50)
    tangent_P = np.ones_like(tangent_Vm) * Pc_calc / 1e6
    plt.plot(tangent_Vm*1e6, tangent_P, 'k--', linewidth=1.5, alpha=0.7,
             label='接線（水平）')
    
    plt.xlabel('モル体積 (cm³/mol)', fontsize=12)
    plt.ylabel('圧力 (MPa)', fontsize=12)
    plt.title('臨界点: van der Waals等温線の変曲点',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(0, 400)
    plt.ylim(0, 12)
    plt.tight_layout()
    plt.show()
    

**要点。** 臨界点は $(\partial P/\partial V)_T = 0$ _かつ_ $(\partial^2 P/\partial V^2)_T = 0$ で定義されます。van der Waals式はフィッティングした $a$、$b$ から $T_c$ と $P_c$ をまずまず再現しますが、$Z_c$ は0.375となり実測値0.274と比べて臨界モル体積で37%の誤差になります。定量的な計算にはより良い状態方程式が必要です。

### 例3: Peng-Robinson式による密度計算

再利用可能なPeng-Robinsonクラスです。3次方程式を解いて $Z$ を求め、気相根または液相根を選び、質量密度に変換します。

コード例3: Peng-Robinson状態方程式ソルバー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 純成分に対するPeng-Robinson状態方程式
    class PengRobinson:
        def __init__(self, Tc, Pc, omega):
            """
            PR状態方程式のパラメータを初期化する。
    
            Parameters:
            -----------
            Tc : float
                臨界温度 (K)
            Pc : float
                臨界圧力 (Pa)
            omega : float
                偏心因子
            """
            self.R = 8.314  # J/(mol·K)
            self.Tc = Tc
            self.Pc = Pc
            self.omega = omega
    
            # aとbの算出
            self.a = 0.45724 * self.R**2 * Tc**2 / Pc
            self.b = 0.07780 * self.R * Tc / Pc
    
            # κパラメータ
            self.kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    
        def alpha(self, T):
            """引力項の温度補正。"""
            Tr = T / self.Tc
            return (1 + self.kappa * (1 - np.sqrt(Tr)))**2
    
        def solve_Z(self, T, P):
            """
            圧縮因子Zについて3次方程式を解く。
    
            Returns:
            --------
            Z_positive : array
                3次方程式の実正根
            """
            A = self.a * self.alpha(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
    
            # 3次方程式: Z^3 + p*Z^2 + q*Z + r = 0
            p = -(1 - B)
            q = A - 3*B**2 - 2*B
            r = -(A*B - B**2 - B**3)
    
            # 3次方程式を解く
            coeffs = [1, p, q, r]
            Z_roots = np.roots(coeffs)
    
            # 実正根のみを返す
            Z_real = Z_roots[np.isreal(Z_roots)].real
            Z_positive = Z_real[Z_real > 0]
    
            return Z_positive
    
        def density(self, T, P, phase='vapor'):
            """
            モル密度を計算する。
    
            Parameters:
            -----------
            phase : str
                'vapor'（最大のZ）または 'liquid'（最小のZ）
    
            Returns:
            --------
            rho : float
                モル密度 (mol/m^3)
            Z : float
                選択された圧縮因子
            """
            Z_values = self.solve_Z(T, P)
    
            if len(Z_values) == 0:
                raise ValueError("実正根が見つかりません")
    
            if phase == 'vapor':
                Z = np.max(Z_values)
            elif phase == 'liquid':
                Z = np.min(Z_values)
            else:
                raise ValueError("phase は 'vapor' または 'liquid' を指定してください")
    
            Vm = Z * self.R * T / P  # モル体積 (m^3/mol)
            rho = 1 / Vm  # モル密度 (mol/m^3)
    
            return rho, Z
    
    # CO2の物性
    co2 = PengRobinson(Tc=304.1, Pc=7.38e6, omega=0.225)
    
    # 各条件での試算
    test_conditions = [
        (310, 8e6, '超臨界'),
        (350, 15e6, '高圧・高温'),
        (280, 5e6, '亜臨界（液相）'),
        (280, 5e6, '亜臨界（気相）')
    ]
    
    print("CO2に対するPeng-Robinson式の計算結果:")
    print("=" * 70)
    
    for T, P, description in test_conditions:
        try:
            if '液相' in description:
                rho, Z = co2.density(T, P, phase='liquid')
            else:
                rho, Z = co2.density(T, P, phase='vapor')
    
            # kg/m^3へ変換（CO2の分子量 = 44.01 g/mol）
            rho_kg = rho * 44.01 / 1000
    
            print(f"\n{description}:")
            print(f"  T = {T} K, P = {P/1e6:.1f} MPa")
            print(f"  密度 = {rho_kg:.1f} kg/m^3")
            print(f"  圧縮因子 Z = {Z:.3f}")
    
        except Exception as e:
            print(f"\n{description}: エラー - {e}")
    
    # 一定温度における密度の圧力依存性をプロット
    T_iso = 313  # K（超臨界）
    P_range = np.linspace(7.5e6, 30e6, 50)
    densities = []
    
    for P in P_range:
        rho, _ = co2.density(T_iso, P, phase='vapor')
        densities.append(rho * 44.01 / 1000)  # kg/m^3へ変換
    
    plt.figure(figsize=(8, 5))
    plt.plot(P_range / 1e6, densities, 'b-', linewidth=2)
    plt.xlabel('圧力 (MPa)', fontsize=12)
    plt.ylabel('密度 (kg/m³)', fontsize=12)
    plt.title(f'CO₂の密度の圧力依存性 T = {T_iso} K（Peng-Robinson）',
              fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

CO2に対するPeng-Robinson式の計算結果: ====================================================================== 超臨界: T = 310 K, P = 8.0 MPa 密度 = 330.0 kg/m^3 圧縮因子 Z = 0.414 高圧・高温: T = 350 K, P = 15.0 MPa 密度 = 428.6 kg/m^3 圧縮因子 Z = 0.529 亜臨界（液相）: T = 280 K, P = 5.0 MPa 密度 = 868.7 kg/m^3 圧縮因子 Z = 0.109 亜臨界（気相）: T = 280 K, P = 5.0 MPa 密度 = 203.3 kg/m^3 圧縮因子 Z = 0.465

**着目点：** 280 K・5 MPaでは3次方程式が3つの実根をもち、同じ呼び出しから液相根（869 kg/m³）と気相根（203 kg/m³）の両方が得られます。二相領域の本質がここに凝縮されています。立方型状態方程式における気液の区別は、最小根か最大根のどちらを選ぶかという問題に帰着します。$T_c$ より上では実根は1つだけになり、密度-圧力曲線は $P_c$ のすぐ上で超臨界CO₂の調整可能性を生む強い非線形性を示します。参照値からのずれは5〜10%程度、$T_c$ の数ケルビン以内では10〜20%まで拡大します。第7章でCoolPropと比較して定量化します。

### 例4: 一般化圧縮因子チャート

換算座標（$T_r$、$P_r$）を用いると多くの流体の挙動が1枚のチャートに集約されます。この例では古典的な圧縮因子チャートを図から読み取る代わりに、Peng-Robinson式から再構成します。

コード例4: 換算座標における圧縮因子
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def PR_compressibility(Tr, Pr, omega):
        """
        換算座標におけるPeng-Robinson式から圧縮因子を計算する。
    
        Parameters:
        -----------
        Tr : float
            換算温度 T/Tc
        Pr : float
            換算圧力 P/Pc
        omega : float
            偏心因子
    
        Returns:
        --------
        Z : float
            圧縮因子（最大の実根）
        """
        # PRパラメータ
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
    
        # 換算パラメータ
        a_r = 0.45724 * alpha / Tr**2
        b_r = 0.07780 / Tr
    
        A = a_r * Pr
        B = b_r * Pr
    
        # 3次方程式を解く
        coeffs = [1, -(1-B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
        roots = np.roots(coeffs)
    
        # 最大の実根（気相的な相）を採る
        real_roots = roots[np.isreal(roots)].real
        Z = np.max(real_roots) if len(real_roots) > 0 else 1.0
    
        return Z
    
    # 換算物性のメッシュを作成
    Tr_range = np.linspace(0.7, 2.0, 100)
    Pr_range = np.linspace(0.1, 5.0, 100)
    Tr_grid, Pr_grid = np.meshgrid(Tr_range, Pr_range)
    
    # CO2（omega = 0.225）についてZを計算
    Z_grid = np.zeros_like(Tr_grid)
    
    for i in range(len(Pr_range)):
        for j in range(len(Tr_range)):
            Z_grid[i, j] = PR_compressibility(Tr_grid[i, j], Pr_grid[i, j],
                                              omega=0.225)
    
    # 等高線マップ
    plt.figure(figsize=(12, 8))
    
    contour = plt.contourf(Tr_grid, Pr_grid, Z_grid, levels=20, cmap='viridis')
    plt.colorbar(contour, label='圧縮因子 Z')
    
    # 等高線
    contour_lines = plt.contour(Tr_grid, Pr_grid, Z_grid, levels=10,
                                colors='white', linewidths=0.5, alpha=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # 臨界点をマーク
    plt.plot(1.0, 1.0, 'r*', markersize=20, label='臨界点 (Tr=1, Pr=1)')
    
    plt.xlabel('換算温度 Tr = T/Tc', fontsize=13)
    plt.ylabel('換算圧力 Pr = P/Pc', fontsize=13)
    plt.title('一般化圧縮因子チャート（Peng-Robinson, ω=0.225）',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.2, color='white')
    plt.xlim(0.7, 2.0)
    plt.ylim(0.1, 5.0)
    plt.tight_layout()
    plt.show()
    
    # 各等温線についてZ対Prをプロット
    plt.figure(figsize=(10, 6))
    
    Tr_values = [0.9, 1.0, 1.05, 1.1, 1.2, 1.5]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(Tr_values)))
    
    for Tr, color in zip(Tr_values, colors):
        Z_values = [PR_compressibility(Tr, Pr, 0.225) for Pr in Pr_range]
        label = f'Tr = {Tr:.2f}'
        if Tr == 1.0:
            label += '（臨界）'
        plt.plot(Pr_range, Z_values, color=color, linewidth=2.5, label=label)
    
    # 理想気体の線
    plt.plot(Pr_range, np.ones_like(Pr_range), 'k--', linewidth=1.5,
             label='理想気体 (Z=1)')
    
    plt.xlabel('換算圧力 Pr = P/Pc', fontsize=12)
    plt.ylabel('圧縮因子 Z', fontsize=12)
    plt.title('CO₂の圧縮因子の換算圧力依存性',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(alpha=0.3)
    plt.xlim(0, 5)
    plt.ylim(0, 1.5)
    plt.tight_layout()
    plt.show()
    
    # 代表的な条件での値を出力
    print("\n代表的な条件における圧縮因子:")
    print("=" * 60)
    test_cases = [
        (1.0, 1.0, "臨界点"),
        (1.05, 1.5, "超臨界（典型的な抽出条件）"),
        (1.2, 2.0, "超臨界（高密度）"),
        (0.9, 0.5, "亜臨界の気相"),
    ]
    
    for Tr, Pr, description in test_cases:
        Z = PR_compressibility(Tr, Pr, 0.225)
        print(f"{description:28s}: Tr={Tr:.2f}, Pr={Pr:.2f} -> Z={Z:.3f}")
    

代表的な条件における圧縮因子: ============================================================ 臨界点 : Tr=1.00, Pr=1.00 -> Z=0.321 超臨界（典型的な抽出条件） : Tr=1.05, Pr=1.50 -> Z=0.345 超臨界（高密度） : Tr=1.20, Pr=2.00 -> Z=0.615 亜臨界の気相 : Tr=0.90, Pr=0.50 -> Z=0.664

**着目点：** $T_r$ 一定のもとで $P_r$ が増すと $Z$ は低下し（引力が支配的）、$P_r$ が小さい領域では1に戻ります（理想気体極限）。そして $T_r = 1$、$P_r \sim 1$〜$2$ の付近で最小値に近づきます。これはまさに抽出に用いられる領域です。$T_r = P_r = 1$ における値はPeng-Robinson式の臨界圧縮因子で、解析的には0.307ですが、その付近で3次方程式が極端に平坦になるため数値解は0.321を返します。

### 例5: 臨界指数と秩序変数

べき乗則 $\rho_L - \rho_V \sim |T - T_c|^{\beta}$ は両対数プロット上で傾き $\beta$ の直線になるため、そこで検証するのが最も確実です。

コード例5: 密度差と臨界指数β
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def density_difference(T, Tc, rho_c, beta=0.326):
        """
        臨界点近傍の気液密度差。
    
        rho_L - rho_V ~ |T - Tc|^beta
        """
        epsilon = np.abs((T - Tc) / Tc)
        # 振幅B0は物質固有。ここでは 2*rho_c を妥当なスケールとして用いる
        B0 = 2.0 * rho_c
        return B0 * epsilon**beta
    
    # CO2のパラメータ
    Tc = 304.1  # K
    rho_c = 467.6  # kg/m^3
    beta = 0.326  # 臨界指数
    
    # 臨界温度以下の温度範囲
    T = np.linspace(280, Tc - 0.01, 100)
    
    # 密度差
    delta_rho = density_difference(T, Tc, rho_c, beta)
    
    # 換算温度差
    epsilon = (Tc - T) / Tc
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 線形プロット
    ax1.plot(T, delta_rho, 'b-', linewidth=2)
    ax1.set_xlabel('温度 (K)', fontsize=12)
    ax1.set_ylabel('密度差 ρ_L − ρ_V (kg/m³)', fontsize=12)
    ax1.set_title('臨界点近傍の気液密度差', fontsize=13)
    ax1.axvline(Tc, color='red', linestyle='--', label=f'Tc = {Tc} K')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 両対数プロット: べき乗則は傾きβの直線になる
    ax2.loglog(epsilon, delta_rho, 'bo-', linewidth=2, markersize=3, label='データ')
    fit_line = delta_rho[0] * (epsilon / epsilon[0])**beta
    ax2.loglog(epsilon, fit_line, 'r--', linewidth=2,
               label=f'べき乗則 (β={beta})')
    ax2.set_xlabel('換算温度差 ε = (Tc − T) / Tc', fontsize=12)
    ax2.set_ylabel('密度差 (kg/m³)', fontsize=12)
    ax2.set_title('べき乗則の検証（両対数）', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # 生成データから線形回帰で指数を復元する
    slope, intercept = np.polyfit(np.log(epsilon), np.log(delta_rho), 1)
    print(f"仮定した臨界指数 β = {beta}")
    print(f"両対数フィットから復元した傾き = {slope:.3f}")
    

仮定した臨界指数 β = 0.326 両対数フィットから復元した傾き = 0.326

**着目点：** 線形プロットでは $T \to T_c$ に向かって共存曲線が平坦化していく様子が見え、両対数プロットではそれが直線になり、その傾きがそのまま臨界指数になります。これは実験の共存曲線データを解析する標準的な方法であり、同じ手順を実際のCO₂のデータに適用すると0.32〜0.33が得られます。van der Waals式が予測する平均場的な値0.5ではありません。

### 例6: Peng-Robinson式によるフガシティ係数

フガシティ係数は、状態方程式を相平衡計算へと変換する要です。ここではCO₂について広い圧力範囲で閉じた表式から評価します。

コード例6: CO₂のフガシティ係数
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    def fugacity_coefficient_PR(T, P, Tc, Pc, omega):
        """
        Peng-Robinson状態方程式からフガシティ係数を計算する。
    
        Returns
        -------
        phi : float
            フガシティ係数 f/P
        Z : float
            圧縮因子
        """
        R = 8.314  # J/(mol·K)
    
        # PRパラメータ
        a = 0.45724 * R**2 * Tc**2 / Pc
        b = 0.07780 * R * Tc / Pc
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
        alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2
    
        A = a * alpha * P / (R**2 * T**2)
        B = b * P / (R * T)
    
        # 立方形式から圧縮因子を求める
        def equation(Z):
            return Z**3 - (1-B)*Z**2 + (A - 3*B**2 - 2*B)*Z - (A*B - B**2 - B**3)
    
        Z = fsolve(equation, 1.0)[0]
    
        # フガシティ係数の閉じた表式
        sqrt2 = np.sqrt(2)
        ln_phi = ((Z - 1) - np.log(Z - B)
                  - A/(2*sqrt2*B) * np.log((Z + (1+sqrt2)*B) / (Z + (1-sqrt2)*B)))
    
        return np.exp(ln_phi), Z
    
    # CO2のパラメータ
    Tc = 304.1  # K
    Pc = 7.38e6  # Pa
    omega = 0.225
    
    temperatures = [300, 320, 350]  # K
    pressures = np.linspace(0.1e6, 20e6, 100)  # Pa
    
    plt.figure(figsize=(10, 6))
    for T in temperatures:
        phi_values = []
        for P in pressures:
            phi, Z = fugacity_coefficient_PR(T, P, Tc, Pc, omega)
            phi_values.append(phi)
    
        label = f'{T} K'
        label += '（亜臨界）' if T < Tc else '（超臨界）'
        plt.plot(pressures/1e6, phi_values, linewidth=2, label=label)
    
    plt.axhline(1.0, color='gray', linestyle='--', label='理想気体 (φ=1)')
    plt.axvline(Pc/1e6, color='red', linestyle='--', alpha=0.5, label='臨界圧力')
    plt.xlabel('圧力 (MPa)', fontsize=12)
    plt.ylabel('フガシティ係数 φ = f/P', fontsize=12)
    plt.title('CO₂のフガシティ係数（Peng-Robinson）', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 特定条件での値
    T_target = 320  # K
    P_target = 10e6  # Pa
    phi, Z = fugacity_coefficient_PR(T_target, P_target, Tc, Pc, omega)
    f = phi * P_target
    print(f"\nT = {T_target} K, P = {P_target/1e6:.1f} MPa:")
    print(f"圧縮因子 Z = {Z:.3f}")
    print(f"フガシティ係数 φ = {phi:.3f}")
    print(f"フガシティ f = {f/1e6:.2f} MPa")
    print(f"理想性からのずれ: {abs(1-phi)*100:.1f}%")
    

T = 320 K, P = 10.0 MPa: 圧縮因子 Z = 0.392 フガシティ係数 φ = 0.604 フガシティ f = 6.04 MPa 理想性からのずれ: 39.6%

**着目点：** $P \to 0$ では $\phi \to 1$ となり、圧力が上がるにつれて引力の効果で1を大きく下回ります。10 MPa・320 Kではフガシティはわずか6.0 MPa、圧力より40%低い値です。プロセス条件下で超臨界CO₂を理想気体として扱うのは小さな近似ではなく、理想気体を仮定した溶解度計算は同程度の倍率で誤ります。

### 例7: Chrastil式による溶解度モデルのフィッティング

Chrastil式は $\ln\rho$ と $1/T$ について線形なので、対数変換したデータに対する通常最小二乗法で3つのパラメータを一度に求められます。非線形ソルバーも初期値も不要です。

コード例7: Chrastilモデルの回帰
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # この例ではCO2密度の算出にコード例3のPengRobinsonクラスを再利用する。
    # 先にそのブロックを実行するか、同じスクリプト内に配置すること。
    
    def chrastil_solubility(rho, T, k, a, b):
        """
        Chrastilの溶解度モデル。
    
        ln(S) = k * ln(rho) + a/T + b
    
        Parameters:
        -----------
        rho : float or array
            超臨界流体の密度 (kg/m^3)
        T : float or array
            温度 (K)
        k, a, b : float
            Chrastilパラメータ
    
        Returns:
        --------
        S : float or array
            溶解度 (kg溶質 / m^3 超臨界流体)
        """
        ln_S = k * np.log(rho) + a / T + b
        return np.exp(ln_S)
    
    # 実験データ: 超臨界CO2中のβ-カロテン（文献の代表的な値）
    data = {
        'T': np.array([313, 313, 313, 333, 333, 333, 353, 353, 353]),  # K
        'P': np.array([15, 20, 25, 15, 20, 25, 15, 20, 25]) * 1e6,  # Pa
        'S_exp': np.array([0.08, 0.15, 0.22, 0.12, 0.20, 0.30,
                           0.18, 0.28, 0.40])  # kg/m^3
    }
    
    co2 = PengRobinson(Tc=304.1, Pc=7.38e6, omega=0.225)
    M_CO2 = 44.01e-3  # kg/mol
    
    def co2_density(T, P):
        """Peng-Robinson式によるCO2の質量密度 (kg/m^3)。"""
        rho_molar, _ = co2.density(T, P, phase='vapor')
        return rho_molar * M_CO2
    
    data['rho'] = np.array([co2_density(T, P)
                            for T, P in zip(data['T'], data['P'])])
    print("Peng-Robinson式によるCO2密度 (kg/m^3):")
    print(np.round(data['rho'], 1))
    
    # 対数変換したモデルに対する線形最小二乗法:
    # ln(S) = k * ln(rho) + a * (1/T) + b
    ln_S_exp = np.log(data['S_exp'])
    ln_rho = np.log(data['rho'])
    T_inv = 1 / data['T']
    
    X = np.column_stack([ln_rho, T_inv, np.ones(len(ln_rho))])
    y = ln_S_exp
    
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    k, a, b = params
    
    print("超臨界CO2中のβ-カロテンに対するChrastilモデルのフィッティング")
    print("=" * 60)
    print("フィッティングされたパラメータ:")
    print(f"  k（会合数） = {k:.3f}")
    print(f"  a（熱項, K） = {a:.1f}")
    print(f"  b（定数） = {b:.3f}")
    
    # 予測値と適合度
    S_pred = chrastil_solubility(data['rho'], data['T'], k, a, b)
    
    SS_res = np.sum((S_pred - data['S_exp'])**2)
    SS_tot = np.sum((data['S_exp'] - np.mean(data['S_exp']))**2)
    R2 = 1 - SS_res / SS_tot
    RMSE = np.sqrt(np.mean((S_pred - data['S_exp'])**2))
    
    print("\n適合度:")
    print(f"  R^2 = {R2:.4f}")
    print(f"  RMSE = {RMSE:.4f} kg/m^3")
    
    # フィッティング結果の可視化
    plt.figure(figsize=(12, 5))
    
    # パリティプロット
    plt.subplot(1, 2, 1)
    plt.scatter(data['S_exp'], S_pred, c=data['T'], cmap='coolwarm', s=100,
                edgecolor='black')
    plt.plot([0, max(data['S_exp'])], [0, max(data['S_exp'])], 'k--',
             label='完全一致')
    plt.xlabel('実験溶解度 (kg/m³)', fontsize=12)
    plt.ylabel('予測溶解度 (kg/m³)', fontsize=12)
    plt.title('パリティプロット: Chrastilモデル', fontsize=13, fontweight='bold')
    plt.colorbar(label='温度 (K)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 各温度における溶解度の圧力依存性
    plt.subplot(1, 2, 2)
    
    for T_iso in [313, 333, 353]:
        P_smooth = np.linspace(15e6, 25e6, 50)
        rho_smooth = np.array([co2_density(T_iso, P) for P in P_smooth])
        S_smooth = chrastil_solubility(rho_smooth, T_iso, k, a, b)
    
        plt.plot(P_smooth / 1e6, S_smooth, linewidth=2, label=f'T = {T_iso} K（モデル）')
    
        mask = data['T'] == T_iso
        plt.scatter(data['P'][mask] / 1e6, data['S_exp'][mask], s=100,
                    edgecolor='black')
    
    plt.xlabel('圧力 (MPa)', fontsize=12)
    plt.ylabel('溶解度 (kg/m³)', fontsize=12)
    plt.title('超臨界CO₂中のβ-カロテンの溶解度', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 新しい条件での予測
    T_new, P_new = 323, 22e6  # K, Pa
    rho_new = co2_density(T_new, P_new)
    S_new = chrastil_solubility(rho_new, T_new, k, a, b)
    
    print("\n新しい条件における予測:")
    print(f"  T = {T_new} K, P = {P_new/1e6:.1f} MPa")
    print(f"  推定CO2密度 = {rho_new:.1f} kg/m^3")
    print(f"  予測溶解度 = {S_new:.3f} kg/m^3")
    

Peng-Robinson式によるCO2密度 (kg/m^3): [748.7 830.6 886.2 562.2 694.9 774.6 411. 563.4 663.5] 超臨界CO2中のβ-カロテンに対するChrastilモデルのフィッティング ============================================================ フィッティングされたパラメータ: k（会合数） = 2.281 a（熱項, K） = -4544.9 b（定数） = -2.787 適合度: R^2 = 0.8986 RMSE = 0.0298 kg/m^3 新しい条件における予測: T = 323 K, P = 22.0 MPa 推定CO2密度 = 793.6 kg/m^3 予測溶解度 = 0.196 kg/m^3

**フィッティングの質は密度モデルの質で決まります。** 溶解度は $\rho^k$ に比例するため、密度の相対誤差 $\varepsilon$ は溶解度でおよそ $k\varepsilon$ になり、そのままフィッティングパラメータに伝播します。Peng-Robinson式の密度（精度5〜10%）は回帰手法を示すには十分ですが、パラメータを公表するのであればCoolPropのような参照状態方程式（第7章）を使うべきです。ここで用いた例示的データセットでは $k \approx 2.3$ が得られますが、これはβ-カロテンほど大きな分子に期待される範囲の下限であり、まさに密度モデルの誤差が生む典型的な症状です。

$a = -4545$ K という負の値は溶媒化が発熱的であることを裏付けています。密度一定のもとでは温度上昇とともに溶解度は低下します。

### 例8: 二成分系の混合則

最後に、純成分の状態方程式を混合物へ拡張する混合則と、混合物パラメータが二成分相互作用パラメータ $k_{12}$ にどれだけ敏感かを見ます。

コード例8: CO₂＋エタノール系のvan der Waals型混合則
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def mixing_rule(x1, a1, a2, b1, b2, k12=0):
        """
        van der Waals型の一流体混合則。
    
        Parameters:
        -----------
        x1 : float
            成分1のモル分率
        a1, a2 : float
            各純成分のaパラメータ
        b1, b2 : float
            各純成分のbパラメータ
        k12 : float
            二成分相互作用パラメータ
    
        Returns:
        --------
        a_mix, b_mix : float
            混合系のパラメータ
        """
        x2 = 1 - x1
    
        # クロスパラメータ
        a12 = np.sqrt(a1 * a2) * (1 - k12)
    
        # 混合則
        a_mix = x1**2 * a1 + 2*x1*x2*a12 + x2**2 * a2
        b_mix = x1 * b1 + x2 * b2
    
        return a_mix, b_mix
    
    R = 8.314  # J/(mol·K)
    
    # CO2
    Tc1, Pc1 = 304.1, 7.38e6  # K, Pa
    a1 = 27 * R**2 * Tc1**2 / (64 * Pc1)
    b1 = R * Tc1 / (8 * Pc1)
    
    # エタノール
    Tc2, Pc2 = 513.9, 6.14e6  # K, Pa
    a2 = 27 * R**2 * Tc2**2 / (64 * Pc2)
    b2 = R * Tc2 / (8 * Pc2)
    
    k12_values = [0.0, 0.05, 0.10, 0.15]
    x_CO2 = np.linspace(0, 1, 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for k12 in k12_values:
        a_mix_values = []
        b_mix_values = []
    
        for x1 in x_CO2:
            a_mix, b_mix = mixing_rule(x1, a1, a2, b1, b2, k12)
            a_mix_values.append(a_mix)
            b_mix_values.append(b_mix)
    
        ax1.plot(x_CO2, a_mix_values, linewidth=2, label=f'k₁₂ = {k12}')
        ax2.plot(x_CO2, np.array(b_mix_values)*1e6, linewidth=2,
                 label=f'k₁₂ = {k12}')
    
    ax1.set_xlabel('CO₂モル分率', fontsize=12)
    ax1.set_ylabel('a_mix (Pa·m⁶/mol²)', fontsize=12)
    ax1.set_title('混合系の引力パラメータ', fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('CO₂モル分率', fontsize=12)
    ax2.set_ylabel('b_mix (cm³/mol)', fontsize=12)
    ax2.set_title('混合系の排除体積パラメータ', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 等モル組成における相互作用パラメータの影響
    print("x_CO2 = 0.5 における k₁₂ の影響:")
    for k12 in k12_values:
        a_mix, b_mix = mixing_rule(0.5, a1, a2, b1, b2, k12)
        print(f"  k12 = {k12:.2f}: a_mix = {a_mix:.4f} Pa·m^6/mol^2, "
              f"b_mix = {b_mix*1e6:.2f} cm^3/mol")
    

x_CO2 = 0.5 における k₁₂ の影響: k12 = 0.00: a_mix = 0.7434 Pa·m^6/mol^2, b_mix = 64.90 cm^3/mol k12 = 0.05: a_mix = 0.7265 Pa·m^6/mol^2, b_mix = 64.90 cm^3/mol k12 = 0.10: a_mix = 0.7096 Pa·m^6/mol^2, b_mix = 64.90 cm^3/mol k12 = 0.15: a_mix = 0.6926 Pa·m^6/mol^2, b_mix = 64.90 cm^3/mol

**着目点：** $b_m$ は組成に対して厳密に線形で、$k_{12}$ には全く影響されません。一方 $a_m$ は $k_{12}$ が大きくなるにつれて下に凸に垂れ下がります。補正を担うのは引力項だけです。だからこそペアごとに $k_{12}$ を1つフィッティングすれば二成分の気液平衡データを再現できることが多く、またエタノールのような共溶媒がモル分率に不釣り合いなほど溶媒力を変える理由にもなっています。

* * *

## まとめ

### 要点

**1\. 状態方程式**

  * 理想気体の法則は臨界点近傍で完全に破綻する。
  * van der Waals式は引力（$a$）と分子サイズ（$b$）に物理的に意味のある補正を導入し、普遍的だが不正確な $Z_c = 3/8$ を予測する。
  * Peng-Robinson式は妥当な計算コストで密度精度5〜10%を与え、実用上の標準となる。
  * 偏心因子 $\omega$ が分子の非球形性を担う。

**2\. 臨界現象**

  * 臨界乳光は相関長の発散から生じる。
  * $\kappa_T$ と $C_P$ は普遍的な臨界指数で発散し、次元と対称性のみが本質的である。
  * くりこみ群理論が、化学的に無関係な流体が同じ挙動を示す理由を説明する。

**3\. 臨界点近傍の物性**

  * エンタルピー、エントロピー、熱容量が狭い温度範囲で急激に変化する。
  * $\Delta H_{vap} \to 0$：蒸気圧曲線は臨界点で終端する。
  * 音速は極小値を通り、超音波計測が成立しなくなる。

**4\. 相平衡**

  * 二成分系は臨界曲線の形状によって分類される（Type I、II、III）。
  * Chrastil式 $\ln S = k \ln \rho + a/T + b$ は3つのパラメータで溶解度を記述する。
  * 逆行凝縮は直観に反するが産業的に活用されている。

**5\. 熱力学計算**

  * フガシティ係数 $\phi$ は非理想性を定量化し、立方型状態方程式では閉じた表式をもつ。
  * 相平衡は $\phi_i^L x_i = \phi_i^V y_i$ に帰着する。
  * フィッティングした $k_{ij}$ を含む混合則により、純成分の状態方程式を混合物へ拡張できる。

**実務への示唆**

  * 流体の種類と、その判断に本当に必要な精度に応じて状態方程式を選ぶ。
  * 臨界点近傍での急激な物性変化に対してマージンを確保して設計する。
  * 一般化相関と妥当性の確認には換算物性を用いる。
  * 多成分の超臨界系で理想混合を仮定してはならない。

* * *

**確認問題**

#### 問題1: 理想気体の法則の破綻

理想気体の法則が前提とする3つの仮定を挙げ、320 KでCO₂を1 MPaから20 MPaまで等温圧縮するとき、どの仮定が最初に破綻するかを説明してください。

#### 問題2: van der Waals式の臨界定数

$P = RT/(V_m - b) - a/V_m^2$ から出発して $V_{m,c} = 3b$ および $T_c = 8a/(27Rb)$ を導出し、$Z_c = 3/8$ となることを示してください。CO₂の実測値が0.274にとどまるのはなぜですか。

#### 問題3: 偏心因子

水は $\omega = 0.344$、メタンは $\omega = 0.011$ です。それぞれについて $\kappa$ を計算し、その差が $\alpha(T)$ の温度依存性について何を意味するかを説明してください。

#### 問題4: 臨界指数

$\alpha = 0.110$、$\beta = 0.326$、$\gamma = 1.237$ を用いてRushbrookeの関係 $\alpha + 2\beta + \gamma = 2$ を確認してください。平均場理論（van der Waals）は $\beta$ をいくつと予測し、実験値とどう比較されますか。

#### 問題5: プロセス制御

密度の調整可能性を最大化するため、ある抽出装置が $T_r = 1.005$ で設計されました。経験のある技術者が $T_r = 1.05$ へ変更する理由を3つ挙げてください。

#### 問題6: Chrastil式の解釈

超臨界CO₂中のトリグリセリドについてフィッティングした結果 $k = 14$ が得られました。この値は溶媒化クラスターについて何を示し、抽出収率の圧力感受性について何を意味しますか。

#### 問題7: フガシティ

320 K・10 MPaにおいてCO₂の $\phi$ は約0.60です。これが物理的に何を意味するかを述べ、$f$ の代わりに $P$ を用いる溶解度モデルが系統的な偏りをもつ理由を説明してください。

#### 問題8: 混合則

超臨界CO₂に共溶媒としてエタノールを5 mol%添加します。$k_{12} = 0.10$ として $a_m$ と $b_m$ がどう変化するかを定性的に議論し、少量の極性共溶媒が極性溶質の溶解度に不釣り合いに大きな効果をもつ理由を説明してください。

* * *
