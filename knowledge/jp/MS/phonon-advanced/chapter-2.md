---
title: "第2章: 非調和フォノンと相転移"
subtitle: "Advanced Phonon Physics シリーズ | 推定読了時間: 45分"
series: "フォノン物理学(上級)"
chapter: 2
language: "ja"
---

[🌐 EN](../../../en/MS/phonon-advanced/chapter-2.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン物理学(上級)](index.md) > 第2章

# 第2章: 非調和フォノンと相転移

**Advanced Phonon Physics シリーズ | 推定読了時間: 45分**

## 学習目標

この章を学ぶことで、以下ができるようになります：

  * 調和近似の限界を理解し、非調和効果の重要性を説明できる
  * 自己無撞着フォノン理論の基礎を理解し、実装できる
  * SCAILD、SSCHAなどの先端的手法の原理を理解できる
  * ソフトモードと相転移の関係を説明できる
  * ランダウ理論を用いて相転移を解析できる
  * ペロブスカイトなどの実際の材料系で相転移を理解できる
  * ALAMODEやSSCHAなどのソフトウェアを使用できる


## 1\. 導入

第1章では調和近似に基づくフォノン理論を学びました。調和近似は低温では優れた近似ですが、 有限温度では格子振動の振幅が大きくなり、非調和効果が無視できなくなります。 

非調和効果は以下のような重要な物理現象を引き起こします： 

  * **熱膨張** ：調和近似では熱膨張はゼロ
  * **フォノン振動数の温度依存性** ：ソフト化やハード化
  * **フォノン寿命の有限性** ：熱伝導率への影響
  * **構造相転移** ：ソフトモードによる不安定性
  * **負熱膨張** ：特殊な非調和相互作用


**重要概念：**

非調和性は格子振動の振幅が大きくなると重要になります。温度が上昇すると、 原子はより大きく振動し、ポテンシャルエネルギー面の調和的でない部分を探索します。 これにより、調和近似では説明できない現象が現れます。 

## 2\. 調和近似の限界

### 2.1 調和近似の仮定

調和近似では、ポテンシャルエネルギーを平衡位置の周りで2次まで展開します：

\[ V(\\{u_i\\}) = V_0 + \frac{1}{2}\sum_{ij\alpha\beta} \Phi_{ij}^{\alpha\beta} u_i^\alpha u_j^\beta \] 

ここで、\\(u_i^\alpha\\)は原子\\(i\\)の\\(\alpha\\)方向の変位、\\(\Phi_{ij}^{\alpha\beta}\\)は力定数行列です。

### 2.2 調和近似で無視される効果

#### 熱膨張の不在

調和近似では、ポテンシャルが対称的なため、熱平均した変位は常にゼロです： 

\[ \langle u_i^\alpha \rangle = 0 \] 

実際の材料では、非調和性により平衡位置が温度とともに変化し、熱膨張が生じます。 

#### フォノンの相互作用

調和近似では、フォノンは独立な粒子として振る舞い、互いに散乱しません。 これは熱伝導率が無限大になることを意味します（非物理的）。 

#### 相転移の不在

調和近似では、力定数が温度に依存しないため、構造相転移を記述できません。 

### 2.3 非調和ポテンシャル

完全なポテンシャルエネルギーは高次項を含みます：

\[ V(\\{u_i\\}) = V_0 + \frac{1}{2}\sum_{ij\alpha\beta} \Phi_{ij}^{\alpha\beta} u_i^\alpha u_j^\beta + \frac{1}{6}\sum_{ijk\alpha\beta\gamma} \Phi_{ijk}^{\alpha\beta\gamma} u_i^\alpha u_j^\beta u_k^\gamma + \cdots \] 

ここで：

  * \\(\Phi_{ijk}^{\alpha\beta\gamma}\\)：3次の力定数（3フォノン相互作用）
  * 4次以上の項：より高次のフォノン相互作用


graph TD A[調和ポテンシャル] --> B[対称的な放物線] A --> C[熱膨張なし] A --> D[フォノン独立] E[非調和ポテンシャル] --> F[非対称性] E --> G[熱膨張あり] E --> H[フォノン相互作用] E --> I[相転移可能] 

## 3\. 自己無撞着フォノン（SCP）理論

### 3.1 基本的なアイデア

自己無撞着フォノン理論は、非調和性を効果的な調和ポテンシャルに繰り込む方法です。 基本的な考え方は： 

  1. 非調和ポテンシャルを熱平均する
  2. 効果的な調和ポテンシャルを得る
  3. このポテンシャルでフォノンを再計算
  4. 収束するまで繰り返す


### 3.2 理論的導出

#### Helmholtz自由エネルギー

系の自由エネルギーは：

\[ F = -k_B T \ln Z = -k_B T \ln \text{Tr} e^{-\beta H} \] 

ここで、\\(\beta = 1/(k_B T)\\)、\\(H\\)はハミルトニアンです。

#### 変分原理

試行的な調和ハミルトニアン\\(H_0\\)を導入し、Gibbs-Bogoliubov不等式を用います： 

\[ F \leq F_0 + \langle H - H_0 \rangle_0 = \Phi[H_0] \] 

ここで、\\(\langle \cdots \rangle_0\\)は\\(H_0\\)に関する熱平均です。 \\(\Phi[H_0]\\)を最小化することで、最適な調和ハミルトニアンを得ます。 

#### 自己無撞着条件

最小化条件\\(\delta \Phi / \delta H_0 = 0\\)から：

\[ \Phi_{ij}^{\alpha\beta}(T) = \left. \frac{\partial^2 V}{\partial u_i^\alpha \partial u_j^\beta} \right|_{\langle u \rangle} + \text{熱揺らぎ項} \] 

この効果的な力定数は温度に依存し、熱平均した変位\\(\langle u \rangle\\)の周りで評価されます。 

### 3.3 SCP方程式

実際のSCP方程式は以下の形になります：

\[ \Phi_{ij}^{\alpha\beta}(T) = \Phi_{ij}^{\alpha\beta}(0) + \sum_{kl\gamma\delta} \Phi_{ijkl}^{\alpha\beta\gamma\delta}(0) \langle u_k^\gamma u_l^\delta \rangle_T \] 

ここで：

  * \\(\Phi_{ij}^{\alpha\beta}(0)\\)：\\(T=0\\)での調和力定数
  * \\(\Phi_{ijkl}^{\alpha\beta\gamma\delta}(0)\\)：4次の力定数
  * \\(\langle u_k^\gamma u_l^\delta \rangle_T\\)：温度\\(T\\)での変位相関


### 3.4 反復解法

SCP方程式は自己無撞着に解く必要があります：

  1. 初期推定：\\(\Phi^{(0)}_{ij}(T) = \Phi_{ij}(0)\\)
  2. \\(\Phi^{(n)}_{ij}(T)\\)を用いてフォノンを計算
  3. 変位相関\\(\langle u_k^\gamma u_l^\delta \rangle_T\\)を計算
  4. 新しい力定数\\(\Phi^{(n+1)}_{ij}(T)\\)を計算
  5. 収束するまで2-4を繰り返す


flowchart LR A[初期力定数 Φ⁰] --> B[フォノン計算] B --> C[変位相関計算] C --> D[新しい力定数 Φⁿ⁺¹] D --> E{収束?} E -->|No| B E -->|Yes| F[自己無撞着解] 

**注意：**

SCP理論は準調和近似（QHA）よりも正確ですが、強い非調和性や相転移の近くでは 破綻する可能性があります。そのような場合は、SSCHAなどのより高度な手法が必要です。 

## 4\. 自己無撞着第一原理格子力学（SCAILD）

### 4.1 SCAILD法の概要

SCAILD（Self-Consistent Ab Initio Lattice Dynamics）は、第一原理計算とSCP理論を 組み合わせた手法です。主な特徴： 

  * 第一原理計算から力定数を直接計算
  * 温度効果を自己無撞着に取り入れる
  * 熱膨張と非調和効果を同時に扱う


### 4.2 計算手順

  1. **初期構造の最適化** ：\\(T=0\\)で構造を最適化
  2. **力定数の計算** ：有限変位法やDFPTで2次、3次力定数を計算
  3. **フォノン計算** ：調和フォノン振動数を計算
  4. **熱平均の計算** ：Bose-Einstein統計で変位相関を計算
  5. **格子定数の更新** ：熱膨張により格子定数を更新
  6. **力定数の更新** ：新しい格子定数で力定数を再計算
  7. **収束判定** ：格子定数とフォノン振動数が収束するまで4-6を繰り返す


### 4.3 熱膨張の計算

格子定数の温度依存性は以下のように計算されます：

\[ a(T) = a_0 + \Delta a(T) \] 

ここで、\\(\Delta a(T)\\)は熱膨張による変化です。線形熱膨張係数は：

\[ \alpha(T) = \frac{1}{a} \frac{da}{dT} \] 

### 4.4 応用例

SCAILDは以下のような材料で成功を収めています： 

  * 金属（Cu, Al, Pt）の熱膨張
  * 半導体（Si, GaAs）のフォノン振動数の温度依存性
  * 酸化物（MgO, Al₂O₃）の高温物性


graph TD A[T=0構造最適化] --> B[力定数計算 DFT] B --> C[フォノン計算] C --> D[熱平均計算] D --> E[格子定数更新] E --> F{収束?} F -->|No| B F -->|Yes| G[温度依存物性] 

## 5\. 確率的自己無撞着調和近似（SSCHA）

### 5.1 SSCHAの動機

従来のSCP理論には限界があります： 

  * 強い非調和性では収束しない
  * 相転移の臨界点近傍で破綻
  * 量子揺らぎの扱いが不十分


SSCHA（Stochastic Self-Consistent Harmonic Approximation）は、 これらの問題を確率的サンプリングで解決します。 

### 5.2 理論的基礎

#### 変分密度行列

SSCHAでは、試行的な密度行列を導入します：

\[ \rho_0 = \frac{e^{-\beta H_0}}{Z_0} \] 

ここで、\\(H_0\\)は試行的な調和ハミルトニアンです。

#### 自由エネルギー汎関数

変分自由エネルギーは：

\[ \mathcal{F}[\rho_0] = \langle H \rangle_0 - TS_0 = F_0 + \langle V - V_0 \rangle_0 \] 

ここで、\\(S_0 = -k_B \text{Tr}(\rho_0 \ln \rho_0)\\)は試行系のエントロピーです。

### 5.3 確率的サンプリング

SSCHAの核心は、調和分布からサンプリングした構造で非調和ポテンシャルを評価することです： 

  1. 調和分布\\(\rho_0\\)から構造\\(\\{R^{(n)}\\}\\)をサンプリング
  2. 各構造で第一原理計算を実行し、エネルギーと力を計算
  3. サンプル平均から効果的な力定数を計算
  4. 新しい調和ハミルトニアンを構築
  5. 収束するまで繰り返す


### 5.4 力定数の計算

効果的な調和力定数は以下のように計算されます：

\[ \Phi_{ij}^{\alpha\beta} = \left\langle \frac{\partial^2 V}{\partial R_i^\alpha \partial R_j^\beta} \right\rangle_0 \] 

実際には、有限差分で近似します：

\[ \Phi_{ij}^{\alpha\beta} \approx \frac{F_i^\alpha(R_j^\beta + \delta) - F_i^\alpha(R_j^\beta - \delta)}{2\delta} \] 

### 5.5 自由エネルギーハイサーフェス

SSCHAは自由エネルギーハイサーフェス上での最小化と見なせます： 

\[ F(\mathcal{R}, \Phi; T) = \min_{\rho_0} \mathcal{F}[\rho_0] \] 

ここで、\\(\mathcal{R}\\)は平衡位置、\\(\Phi\\)は効果的な力定数です。 この自由エネルギー面で相転移を解析できます。 

### 5.6 SSCHAの利点

  * **強い非調和性** ：摂動論を使わないため、強い非調和性も扱える
  * **相転移** ：自由エネルギー面の不安定性から相転移を予測
  * **量子効果** ：量子揺らぎを正しく取り入れる
  * **一般性** ：任意の第一原理計算コードと組み合わせ可能


**SSCHAの成功例：**

  * SnTe、PbTeの構造相転移
  * 水素化物超伝導体の安定性
  * 熱電材料の格子熱伝導率


## 6\. 温度依存フォノン振動数

### 6.1 実験的観測

フォノン振動数は温度とともに変化します。主な測定手法： 

  * **ラマン分光** ：光学フォノンの振動数と線幅
  * **中性子散乱** ：全ブリルアンゾーンのフォノン分散
  * **赤外分光** ：極性光学フォノン


### 6.2 温度シフトのメカニズム

#### 体積効果（準調和）

熱膨張により格子定数が変化し、フォノン振動数が変化します： 

\[ \omega(T) = \omega_0 \left(1 + \gamma \frac{\Delta V(T)}{V_0}\right) \] 

ここで、\\(\gamma\\)はGrüneisenパラメータです。

#### 真の非調和効果

3フォノン、4フォノン相互作用により、フォノン自己エネルギーが生じます： 

\[ \Pi(\mathbf{q}, \omega; T) = \Delta(\mathbf{q}, T) - i\Gamma(\mathbf{q}, T) \] 

  * \\(\Delta(\mathbf{q}, T)\\)：振動数シフト（実部）
  * \\(\Gamma(\mathbf{q}, T)\\)：線幅（虚部、逆寿命）


### 6.3 繰り込み振動数

観測される振動数は繰り込まれた値です：

\[ \omega_{\text{obs}}(\mathbf{q}, T) = \omega_0(\mathbf{q}) + \Delta(\mathbf{q}, T) \] 

温度依存性は以下のようになります：

\[ \Delta(\mathbf{q}, T) \propto \left(1 + 2n_B(\omega/T)\right) \] 

ここで、\\(n_B\\)はBose-Einstein分布関数です。

### 6.4 ソフト化とハード化

材料により、温度とともにフォノンがソフト化またはハード化します：

  * **ソフト化** （\\(\omega\\)減少）：多くの強誘電体、相転移物質
  * **ハード化** （\\(\omega\\)増加）：一部の金属、半導体


graph LR A[温度上昇] --> B[熱膨張] A --> C[フォノン占有増加] B --> D[体積効果] C --> E[非調和散乱] D --> F[振動数シフト] E --> F E --> G[線幅増加] 

## 7\. ソフトモードと構造不安定性

### 7.1 ソフトモードの定義

ソフトモードとは、温度や圧力などのパラメータを変化させると、 振動数がゼロに近づくフォノンモードです： 

\[ \omega_s(\mathbf{q}, T) \to 0 \quad \text{as} \quad T \to T_c \] 

\\(T_c\\)は相転移温度です。振動数がゼロになると、系は新しい構造に対して不安定になります。 

### 7.2 不安定性の物理

フォノン振動数の二乗は、力定数（復元力）に比例します： 

\[ \omega^2 \propto \Phi \] 

\\(\omega^2 < 0\\)（虚の振動数）は、現在の構造が不安定であることを意味します。 系はソフトモードの方向に変位し、より低いエネルギーの構造を見つけます。 

### 7.3 ソフトモードと秩序変数

ソフトモードの振幅は、相転移の秩序変数と関連します： 

\[ \eta \propto \sqrt{\langle |u_{\mathbf{q}_s}|^2 \rangle} \] 

ここで、\\(u_{\mathbf{q}_s}\\)はソフトモードの変位、\\(\eta\\)は秩序変数です。 

### 7.4 ソフトモードの種類

#### Γ点ソフトモード（\\(\mathbf{q}=0\\)）

  * ゾーン中心のモード
  * 単位胞内での歪み
  * 例：BaTiO₃の強誘電転移


#### ゾーン境界ソフトモード

  * \\(\mathbf{q} \neq 0\\)のモード
  * 単位胞の倍化を伴う
  * 例：SrTiO₃の\\(R\\)点ソフトモード


### 7.5 凍結（Freezing-in）

相転移点以下では、ソフトモードが「凍結」し、有限振幅を持ちます： 

\[ u_{\mathbf{q}_s} \neq 0 \quad \text{for} \quad T < T_c \] 

これにより、対称性が破れ、新しい結晶構造が形成されます。 

**古典的な例：SrTiO₃**

SrTiO₃は105 Kで立方晶から正方晶への構造相転移を示します。 この転移は、\\(R\\)点（ブリルアンゾーン境界）のソフトモードの凍結によるものです。 ソフトモードの振動数は、\\(T_c\\)に近づくと\\(\omega \propto (T - T_c)^{1/2}\\)のように減少します。 

## 8\. 相転移のランダウ理論

### 8.1 ランダウ理論の基本原理

ランダウ理論は、相転移を秩序変数\\(\eta\\)の関数としての自由エネルギー展開で記述します： 

\[ F(\eta, T) = F_0 + \frac{1}{2}a(T)\eta^2 + \frac{1}{4}b\eta^4 + \frac{1}{6}c\eta^6 + \cdots \] 

ここで：

  * \\(\eta\\)：秩序変数（例：自発分極、変位振幅）
  * \\(a(T) = a_0(T - T_c)\\)：温度依存係数
  * \\(b, c, \ldots\\)：高次係数


### 8.2 2次相転移

\\(b > 0\\)の場合、2次（連続）相転移が起こります： 

\[ F(\eta, T) = F_0 + \frac{1}{2}a_0(T - T_c)\eta^2 + \frac{1}{4}b\eta^4 \] 

#### 平衡条件

自由エネルギーを最小化します：

\[ \frac{\partial F}{\partial \eta} = 0 \Rightarrow a(T)\eta + b\eta^3 = 0 \] 

解は：

  * \\(T > T_c\\)：\\(\eta = 0\\)（対称相）
  * \\(T < T_c\\)：\\(\eta = \pm\sqrt{\frac{a_0(T_c - T)}{b}}\\)（対称性の破れた相）


### 8.3 1次相転移

\\(b < 0\\)の場合、1次（不連続）相転移が起こります。安定化のため\\(c > 0\\)が必要： 

\[ F(\eta, T) = F_0 + \frac{1}{2}a(T)\eta^2 + \frac{1}{4}b\eta^4 + \frac{1}{6}c\eta^6 \] 

この場合、\\(\eta\\)は転移点で不連続に変化し、潜熱が発生します。 

### 8.4 対称性と許容項

ランダウ展開の許容項は、系の対称性によって決まります： 

  * 高対称性：奇数次項が禁止される（例：\\(\eta^3\\)項なし）
  * 低対称性：奇数次項が許される → 常に1次転移


### 8.5 臨界指数

相転移近傍での物理量のスケーリング則： 

\[ \begin{align} \eta &\propto (T_c - T)^\beta \quad &(\text{秩序変数}) \\\ C &\propto |T - T_c|^{-\alpha} \quad &(\text{比熱}) \\\ \chi &\propto |T - T_c|^{-\gamma} \quad &(\text{感受率}) \end{align} \] 

平均場理論（ランダウ理論）では：

  * \\(\beta = 1/2\\)
  * \\(\alpha = 0\\)（対数発散）
  * \\(\gamma = 1\\)


graph TD A[自由エネルギー F(η,T)] --> B{b > 0?} B -->|Yes| C[2次相転移] B -->|No| D[1次相転移] C --> E[連続的変化] C --> F[潜熱なし] D --> G[不連続変化] D --> H[潜熱あり] 

## 9\. 転移のタイプ：変位型 vs 秩序-無秩序型

### 9.1 変位型転移（Displacive）

#### 特徴

  * 高温相ですでにソフトモードが存在
  * 転移により原子位置が連続的に変化
  * 短距離秩序がほとんどない


#### メカニズム

温度低下に伴い、ソフトモードの振動数が減少し、臨界温度でゼロになります。 その後、変位が凍結して新しい構造が形成されます。 

#### 典型例

  * **BaTiO₃** ：常誘電-強誘電転移
  * **SrTiO₃** ：立方晶-正方晶転移
  * **水晶** ：α-β転移


### 9.2 秩序-無秩序型転移（Order-Disorder）

#### 特徴

  * 高温相で原子が複数のサイトを占有
  * 転移により特定サイトへの占有秩序化
  * 強い短距離秩序が存在


#### メカニズム

高温では、原子が熱的に複数の等価サイト間を移動します（無秩序）。 温度低下により、特定のサイト占有が優位になり、秩序化します。 

#### 典型例

  * **KH₂PO₄（KDP）** ：強誘電転移
  * **氷** ：プロトン秩序化
  * **合金** ：化学的秩序化


### 9.3 両者の比較

特性 | 変位型 | 秩序-無秩序型  
---|---|---  
高温相 | 単一の平衡位置 | 複数の等価サイト  
ソフトモード | 明瞭に観測される | 観測困難（強い減衰）  
短距離秩序 | 弱い | 強い  
転移の性質 | 多くは2次 | 1次も多い  
理論的扱い | ランダウ理論、SCPが有効 | Isingモデルなど統計力学  
  
### 9.4 中間的挙動

実際の材料では、純粋な変位型や秩序-無秩序型は稀で、多くは中間的挙動を示します： 

  * 部分的な秩序-無秩序特性を持つ変位型転移
  * 温度により転移の性質が変化
  * 異なるフォノンモードが異なる性質を持つ


**実験的区別法：**

  * **中性子散乱** ：ソフトモードの観測可否
  * **NMR** ：局所的な無秩序の検出
  * **X線散乱** ：短距離秩序の有無
  * **比熱測定** ：転移のエントロピー変化


## 10\. ペロブスカイト相転移

### 10.1 ペロブスカイト構造

ペロブスカイトは化学式ABO₃を持つ結晶構造です： 

  * A：大きな陽イオン（12配位）
  * B：小さな陽イオン（6配位、酸素八面体の中心）
  * O：酸素イオン（八面体を形成）


### 10.2 BaTiO₃の強誘電転移

#### 相転移系列

BaTiO₃は温度により複数の相転移を示します：

\[ \begin{array}{ccccc} \text{立方晶} & \xrightarrow{393\text{ K}} & \text{正方晶} & \xrightarrow{278\text{ K}} & \text{斜方晶} & \xrightarrow{183\text{ K}} & \text{菱面体} \\\ Pm\bar{3}m & & P4mm & & Amm2 & & R3m \\\ \text{常誘電} & & \multicolumn{5}{c}{\text{強誘電}} \end{array} \] 

#### ソフトモードメカニズム

立方晶→正方晶転移は、Γ点の横波光学（TO）フォノンのソフト化により起こります： 

  * Ti⁴⁺イオンが酸素八面体の中心から変位
  * 自発分極が生じる（強誘電性）
  * 変位型転移の典型例


### 10.3 SrTiO₃の構造相転移

#### 105 K転移

SrTiO₃は105 Kで立方晶から正方晶へ転移します： 

  * \\(R\\)点（\\(\mathbf{q} = (\pi/a, \pi/a, \pi/a)\\)）のソフトモード
  * 酸素八面体の回転による
  * 単位胞が2倍になる
  * 強誘電性は現れない


#### 量子臨界性

SrTiO₃は「量子常誘電体」として知られています： 

  * ソフトモードが0 Kでも有限の振動数を持つ
  * 量子揺らぎが強誘電転移を抑制
  * 低温で異常に大きな誘電率


### 10.4 PbTiO₃とSnTe

これらの材料は、SSCHAの成功例です： 

  * **PbTiO₃** ：高温で強誘電転移（763 K）
  * **SnTe** ：低温で強誘電転移（変位型、弱い）


両者とも、準調和近似では転移温度を正確に予測できませんが、 SSCHAでは実験値とよく一致します。 

### 10.5 トレランス因子

ペロブスカイトの構造安定性は、トレランス因子\\(t\\)で予測されます： 

\[ t = \frac{r_A + r_O}{\sqrt{2}(r_B + r_O)} \] 

ここで、\\(r_A, r_B, r_O\\)はイオン半径です。

  * \\(t \approx 1\\)：理想的な立方晶
  * \\(t < 1\\)：八面体の回転による歪み
  * \\(t > 1\\)：六方晶ペロブスカイトなど


graph TD A[ペロブスカイト ABO₃] --> B[t ≈ 1: 立方晶] A --> C[t < 1: 歪んだ構造] A --> D[t > 1: 他の構造] B --> E[BaTiO₃ 高温相] C --> F[SrTiO₃ 低温相] C --> G[八面体回転] E --> H[強誘電転移] H --> I[Ti変位] 

## 11\. 負熱膨張材料

### 11.1 負熱膨張の定義

ほとんどの材料は加熱により膨張しますが、一部の材料は収縮します（負熱膨張、NTE）： 

\[ \alpha = \frac{1}{V}\frac{dV}{dT} < 0 \] 

ここで、\\(\alpha\\)は体積熱膨張係数です。

### 11.2 NTEのメカニズム

#### 1\. 横振動メカニズム（RUM: Rigid Unit Mode）

剛体ユニット（例：八面体、四面体）が回転または振動することで、 全体の体積が減少します。 

  * 例：ZrW₂O₈、ScF₃
  * メカニズム：M-O-M結合角の曲げ振動
  * 広い温度範囲でNTE


#### 2\. 磁気体積効果

磁気秩序の消失により体積が減少： 

  * 例：反強磁性体Mn₃AN（A=Ga, Zn）
  * Néel温度付近でNTE


#### 3\. 電荷移動

価数変化により原子サイズが変化： 

  * 例：BiNiO₃
  * 温度により電荷分布が変化


### 11.3 ZrW₂O₈の例

ZrW₂O₈は、0-1050 Kの広い温度範囲で等方的なNTEを示します： 

  * 線形熱膨張係数：\\(\alpha_L = -9 \times 10^{-6}\\) K⁻¹
  * 立方晶構造（\\(P2_13\\)）
  * ZrO₆八面体とWO₄四面体が頂点共有


#### RUMモード

ZrW₂O₈のNTEは、低エネルギーの横振動フォノンに起因します： 

  * 剛体ユニット（八面体、四面体）は変形しない
  * ユニット間の結合角が変化
  * 平均二乗変位が増加すると、平均結合角が減少
  * 結果として体積が減少


### 11.4 理論的記述

NTEは、負のGrüneisenパラメータを持つフォノンモードで説明されます： 

\[ \gamma_i = -\frac{V}{\omega_i}\frac{\partial \omega_i}{\partial V} \] 

全体の熱膨張係数は：

\[ \alpha = \frac{1}{BV}\sum_i \gamma_i C_{V,i} \] 

ここで、\\(B\\)は体積弾性率、\\(C_{V,i}\\)はモード\\(i\\)の熱容量です。 低周波のRUMモードが負の\\(\gamma_i\\)を持つと、全体でNTEになります。 

### 11.5 応用

  * **ゼロ熱膨張材料** ：正と負のNTE材料を複合化
  * **精密機器** ：光学部品、センサー
  * **熱応力緩和** ：接合部、コーティング


**第一原理計算の役割：**

NTE材料の設計には、第一原理フォノン計算が不可欠です： 

  * フォノン分散からRUMモードを同定
  * Grüneisenパラメータの計算
  * 新材料のスクリーニング
  * SCPやSSCHAで温度依存性を予測


## 12\. 計算実装

### 12.1 簡易SCP計算

以下は、3次の力定数を用いた簡易的なSCP計算の実装例です：
    
    
    import numpy as np
    from scipy.linalg import eigh
    
    class SimpleSCP:
        """
        簡易的な自己無撞着フォノン計算
    
        1次元モデルでSCP理論を実装
        """
    
        def __init__(self, phi2, phi3, phi4, mass, n_atoms=1):
            """
            Parameters:
            -----------
            phi2 : float
                2次の力定数 (調和項)
            phi3 : float
                3次の力定数
            phi4 : float
                4次の力定数
            mass : float
                原子質量
            n_atoms : int
                単位胞内の原子数
            """
            self.phi2_0 = phi2  # 元の2次力定数
            self.phi3 = phi3
            self.phi4 = phi4
            self.mass = mass
            self.n_atoms = n_atoms
            self.kB = 1.380649e-23  # Boltzmann constant (J/K)
    
        def bose_einstein(self, omega, T):
            """Bose-Einstein分布関数"""
            if T == 0:
                return 0.0
            hbar = 1.054571817e-34
            x = hbar * omega / (self.kB * T)
            if x > 100:  # オーバーフロー回避
                return 0.0
            return 1.0 / (np.exp(x) - 1.0)
    
        def mean_square_displacement(self, omega, T):
            """平均二乗変位 <u²> の計算"""
            if omega <= 0:
                return np.inf
            hbar = 1.054571817e-34
            n_BE = self.bose_einstein(omega, T)
            # <u²> = (ℏ/2Mω)(2n + 1)
            return (hbar / (2 * self.mass * omega)) * (2 * n_BE + 1)
    
        def effective_force_constant(self, T, omega_current):
            """
            温度依存の効果的力定数を計算
    
            Φ(T) = Φ₂ + Φ₄·<u²>
    
            (3次項は1次元では平均がゼロになるため省略)
            """
            u2_mean = self.mean_square_displacement(omega_current, T)
            phi_eff = self.phi2_0 + self.phi4 * u2_mean
            return phi_eff
    
        def calculate_frequency(self, phi_eff):
            """力定数から振動数を計算"""
            if phi_eff <= 0:
                return 0.0  # 不安定
            omega2 = phi_eff / self.mass
            return np.sqrt(omega2)
    
        def self_consistent_solve(self, T, max_iter=100, tol=1e-6):
            """
            自己無撞着方程式を解く
    
            Parameters:
            -----------
            T : float
                温度 (K)
            max_iter : int
                最大反復回数
            tol : float
                収束判定の閾値
    
            Returns:
            --------
            omega : float
                自己無撞着なフォノン振動数
            converged : bool
                収束したかどうか
            """
            # 初期推定：T=0の振動数
            omega_old = self.calculate_frequency(self.phi2_0)
    
            for iteration in range(max_iter):
                # 効果的力定数を計算
                phi_eff = self.effective_force_constant(T, omega_old)
    
                # 新しい振動数を計算
                omega_new = self.calculate_frequency(phi_eff)
    
                # 収束判定
                diff = abs(omega_new - omega_old)
                if diff < tol:
                    return omega_new, True
    
                # 混合（収束を安定化）
                mixing = 0.5
                omega_old = mixing * omega_new + (1 - mixing) * omega_old
    
            # 収束しなかった
            return omega_new, False
    
        def temperature_scan(self, T_min=0, T_max=1000, n_points=50):
            """
            温度範囲でフォノン振動数を計算
    
            Returns:
            --------
            temperatures : ndarray
                温度の配列
            frequencies : ndarray
                対応する振動数の配列
            """
            temperatures = np.linspace(T_min, T_max, n_points)
            frequencies = np.zeros(n_points)
    
            for i, T in enumerate(temperatures):
                omega, converged = self.self_consistent_solve(T)
                if not converged:
                    print(f"Warning: Not converged at T={T:.1f} K")
                frequencies[i] = omega
    
            return temperatures, frequencies
    
    
    # 使用例
    if __name__ == "__main__":
        # パラメータ設定（任意単位）
        phi2 = 100.0      # 調和力定数
        phi3 = -5.0       # 3次力定数（1次元では寄与しない）
        phi4 = 0.5        # 4次力定数（正：ハード化）
        mass = 1.0        # 原子質量
    
        # SCPオブジェクト作成
        scp = SimpleSCP(phi2, phi3, phi4, mass)
    
        # 温度スキャン
        temperatures, frequencies = scp.temperature_scan(T_min=0, T_max=500, n_points=50)
    
        # 結果の出力
        print("Temperature (K) | Frequency (rad/s)")
        print("-" * 40)
        for T, omega in zip(temperatures[::10], frequencies[::10]):
            print(f"{T:14.1f} | {omega:18.6e}")
    
        # プロット（matplotlib必要）
        try:
            import matplotlib.pyplot as plt
    
            plt.figure(figsize=(10, 6))
            plt.plot(temperatures, frequencies / 1e12, 'b-', linewidth=2)
            plt.xlabel('Temperature (K)', fontsize=14)
            plt.ylabel('Frequency (THz)', fontsize=14)
            plt.title('Temperature-Dependent Phonon Frequency (SCP)', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('scp_temperature_dependence.png', dpi=150)
            print("\nPlot saved as 'scp_temperature_dependence.png'")
        except ImportError:
            print("\nMatplotlib not available. Skipping plot.")
    

### 12.2 ソフトモード検出

フォノン計算から構造不安定性を検出する例：
    
    
    import numpy as np
    
    def detect_soft_modes(frequencies, q_points, threshold=1.0):
        """
        ソフトモード（低振動数モード）を検出
    
        Parameters:
        -----------
        frequencies : ndarray, shape (n_q, n_modes)
            各q点でのフォノン振動数 (THz)
        q_points : ndarray, shape (n_q, 3)
            q点の座標
        threshold : float
            ソフトモードの閾値 (THz)
    
        Returns:
        --------
        soft_modes : list of dict
            ソフトモードの情報
        """
        soft_modes = []
    
        for iq, (q, omega_q) in enumerate(zip(q_points, frequencies)):
            for imode, omega in enumerate(omega_q):
                if omega < threshold:
                    soft_mode_info = {
                        'q_index': iq,
                        'q_point': q,
                        'mode_index': imode,
                        'frequency': omega,
                        'is_unstable': omega < 0  # 虚振動数
                    }
                    soft_modes.append(soft_mode_info)
    
        return soft_modes
    
    
    def analyze_instability(soft_modes):
        """構造不安定性を解析"""
        if not soft_modes:
            print("No soft modes detected. Structure is stable.")
            return
    
        print(f"Found {len(soft_modes)} soft mode(s):")
        print("-" * 60)
    
        for i, mode in enumerate(soft_modes):
            q = mode['q_point']
            omega = mode['frequency']
            status = "UNSTABLE" if mode['is_unstable'] else "soft"
    
            print(f"Mode {i+1}:")
            print(f"  q-point: ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")
            print(f"  Frequency: {omega:.4f} THz [{status}]")
    
            # q点の特殊点判定
            if np.allclose(q, [0, 0, 0]):
                print("  → Γ-point mode: uniform distortion")
            elif np.allclose(np.abs(q), [0.5, 0.5, 0.5]):
                print("  → R-point mode: unit cell doubling")
            print()
    
    
    # 使用例
    if __name__ == "__main__":
        # ダミーデータ（実際にはフォノン計算から取得）
        n_qpoints = 10
        n_modes = 12
    
        # ランダムな振動数（一部にソフトモードを含む）
        np.random.seed(42)
        frequencies = np.random.uniform(0.5, 15.0, (n_qpoints, n_modes))
    
        # Γ点にソフトモードを追加
        frequencies[0, 0] = 0.3  # ソフト
        frequencies[0, 1] = -0.5  # 不安定
    
        # q点（簡略化）
        q_points = np.random.uniform(-0.5, 0.5, (n_qpoints, 3))
        q_points[0] = [0, 0, 0]  # Γ点
    
        # ソフトモード検出
        soft_modes = detect_soft_modes(frequencies, q_points, threshold=1.0)
    
        # 解析
        analyze_instability(soft_modes)
    

### 12.3 ランダウ自由エネルギー

ランダウ理論の自由エネルギーを可視化：
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def landau_free_energy(eta, T, Tc, a0, b, c=0):
        """
        ランダウ自由エネルギー
    
        F(η,T) = (a₀/2)(T-Tc)η² + (b/4)η⁴ + (c/6)η⁶
    
        Parameters:
        -----------
        eta : float or ndarray
            秩序変数
        T : float
            温度
        Tc : float
            臨界温度
        a0, b, c : float
            ランダウ係数
        """
        a = a0 * (T - Tc)
        F = 0.5 * a * eta**2 + 0.25 * b * eta**4
        if c != 0:
            F += (1.0/6.0) * c * eta**6
        return F
    
    
    def plot_landau_evolution(Tc=400, a0=1.0, b=1.0, c=0, T_range=None):
        """
        温度変化に伴うランダウ自由エネルギーの変化を可視化
        """
        if T_range is None:
            T_range = [Tc + 100, Tc + 50, Tc, Tc - 50, Tc - 100]
    
        eta = np.linspace(-2, 2, 500)
    
        plt.figure(figsize=(12, 8))
    
        for T in T_range:
            F = landau_free_energy(eta, T, Tc, a0, b, c)
            label = f'T = {T:.0f} K'
            if T > Tc:
                label += ' (T > Tc)'
            elif T == Tc:
                label += ' (T = Tc)'
            else:
                label += ' (T < Tc)'
            plt.plot(eta, F, linewidth=2, label=label)
    
        plt.xlabel('Order Parameter η', fontsize=14)
        plt.ylabel('Free Energy F(η,T)', fontsize=14)
        plt.title('Landau Free Energy vs Temperature', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('landau_free_energy.png', dpi=150)
        print("Plot saved as 'landau_free_energy.png'")
    
    
    def find_equilibrium(T, Tc, a0, b, c=0):
        """平衡状態の秩序変数を求める"""
        from scipy.optimize import minimize_scalar
    
        # η > 0 の領域で最小化
        result = minimize_scalar(
            lambda eta: landau_free_energy(eta, T, Tc, a0, b, c),
            bounds=(0, 5),
            method='bounded'
        )
    
        eta_eq = result.x
        F_eq = result.fun
    
        # 対称性を考慮（±η）
        if T < Tc and abs(eta_eq) > 1e-6:
            return eta_eq, F_eq
        else:
            return 0.0, landau_free_energy(0, T, Tc, a0, b, c)
    
    
    # 使用例
    if __name__ == "__main__":
        # パラメータ
        Tc = 400  # 臨界温度 (K)
        a0 = 1.0
        b = 1.0
    
        # 2次相転移の場合
        print("Second-order phase transition (b > 0):")
        plot_landau_evolution(Tc=Tc, a0=a0, b=1.0, c=0)
    
        temperatures = np.linspace(Tc - 150, Tc + 150, 50)
        order_params = []
    
        for T in temperatures:
            eta_eq, _ = find_equilibrium(T, Tc, a0, b=1.0, c=0)
            order_params.append(eta_eq)
    
        plt.figure(figsize=(10, 6))
        plt.plot(temperatures, order_params, 'b-', linewidth=2)
        plt.xlabel('Temperature (K)', fontsize=14)
        plt.ylabel('Order Parameter η', fontsize=14)
        plt.title('Order Parameter vs Temperature (2nd Order)', fontsize=16)
        plt.axvline(x=Tc, color='r', linestyle='--', label=f'Tc = {Tc} K')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('order_parameter_vs_T.png', dpi=150)
        print("Plot saved as 'order_parameter_vs_T.png'")
    

## 13\. ソフトウェアツール

### 13.1 ALAMODE

#### 概要

  * **開発** ：東北大学（寺倉研究室）
  * **機能** ：格子力学、フォノン-フォノン相互作用
  * **特徴** ：高次力定数の抽出、熱伝導率計算
  * **ライセンス** ：MIT License（オープンソース）


#### 主な機能

  * 2次、3次、4次の力定数抽出（有限変位データから）
  * フォノン分散、状態密度の計算
  * 3フォノン散乱率の計算
  * 格子熱伝導率の計算（Boltzmann輸送方程式）
  * 温度依存フォノン振動数（SCP理論）


#### ワークフロー

  1. 超格子を作成（`alm` ツール）
  2. 第一原理計算で変位構造のエネルギー・力を計算
  3. 力定数を抽出（最小二乗法）
  4. `anphon` でフォノン計算、熱伝導率計算


#### インストール
    
    
    # GitHubからクローン
    git clone https://github.com/ttadano/ALM.git
    git clone https://github.com/ttadano/ANPHON.git
    
    # ビルド
    cd ALM
    mkdir build && cd build
    cmake ..
    make
    
    cd ../../ANPHON
    mkdir build && cd build
    cmake ..
    make
    

### 13.2 TDEP（Temperature Dependent Effective Potential）

#### 概要

  * **開発** ：Linköping大学（Olle Hellman）
  * **手法** ：分子動力学ベースの効果的力定数抽出
  * **特徴** ：有限温度での実効フォノン


#### 原理

TDEPは、第一原理分子動力学（AIMD）のトラジェクトリから、 温度依存の効果的力定数を抽出します： 

  1. 有限温度でAIMDシミュレーション
  2. 時系列の原子座標と力を記録
  3. 効果的な調和力定数を最小二乗法で決定
  4. この力定数でフォノンを計算


#### 利点

  * 強い非調和性も扱える
  * 相転移を自然に記述
  * 高温でも安定


#### 制限

  * 計算コストが高い（AIMD必要）
  * 統計的サンプリングが重要


### 13.3 SSCHA（Python実装）

#### 概要

  * **開発** ：イタリア、フランスの研究グループ
  * **言語** ：Python
  * **手法** ：確率的自己無撞着調和近似
  * **URL** ：[http://sscha.eu/](http://sscha.eu/)


#### インストール
    
    
    # pipでインストール
    pip install python-sscha
    
    # または GitHubから
    git clone https://github.com/SSCHAcode/python-sscha.git
    cd python-sscha
    python setup.py install
    

#### 基本的な使い方
    
    
    import sscha, sscha.Ensemble
    
    # 初期動力学行列を読み込み
    dyn = sscha.Phonons.Phonons("harmonic_dyn", nqirr=1)
    
    # アンサンブルを生成
    ensemble = sscha.Ensemble.Ensemble(dyn, T=300, supercell=(2,2,2))
    ensemble.generate(N=100)  # 100構造
    
    # 各構造で第一原理計算を実行（外部で）
    # ...
    
    # エネルギーと力を読み込み
    ensemble.load_energies_forces("data_dir")
    
    # SCF最小化
    minimizer = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
    minimizer.init()
    minimizer.run()
    
    # 自由エネルギーヘッセ行列で安定性をチェック
    minimizer.compute_free_energy_hessian()
    

#### 主な機能

  * 自由エネルギー最小化
  * 温度依存フォノン分散
  * 構造相転移の予測
  * スペクトル関数の計算
  * 熱膨張の計算


### 13.4 Phonopy（参考：調和近似）

非調和計算の比較のため、Phonopy（調和）も有用です： 

  * 準調和近似（QHA）の実装あり
  * 温度依存性は体積効果のみ
  * 多くの第一原理コードと連携


**ツール選択のガイドライン：**

  * **ALAMODE** ：熱伝導、弱い非調和性
  * **TDEP** ：強い非調和性、高温
  * **SSCHA** ：相転移、構造不安定性
  * **Phonopy** ：ベンチマーク、調和近似で十分な系


## 14\. まとめ

### 重要ポイント

  * **非調和性の重要性** ：有限温度では調和近似では説明できない現象（熱膨張、フォノン寿命、相転移）が現れる
  * **SCP理論** ：非調和性を効果的な調和ポテンシャルに繰り込む自己無撞着な方法
  * **SCAILD** ：第一原理計算とSCPを組み合わせ、熱膨張と温度依存フォノンを計算
  * **SSCHA** ：確率的サンプリングにより、強い非調和性や相転移を扱える高度な手法
  * **ソフトモード** ：振動数がゼロに近づくモードで、構造不安定性と相転移を引き起こす
  * **ランダウ理論** ：秩序変数の自由エネルギー展開により、相転移を現象論的に記述
  * **転移のタイプ** ：変位型と秩序-無秩序型では、微視的メカニズムが異なる
  * **ペロブスカイト** ：多様な相転移を示す代表的な材料群
  * **負熱膨張** ：特殊な非調和相互作用により、加熱で収縮する材料
  * **計算ツール** ：ALAMODE、TDEP、SSCHAなど、目的に応じた選択が重要


### 次のステップ

次章では、フォノン-フォノン相互作用と格子熱伝導を詳しく学びます。

## 15\. 演習問題

### 演習 2.1: 調和近似の限界

**問題：**

調和近似において、なぜ熱膨張がゼロになるか、ポテンシャルエネルギーの対称性に基づいて説明せよ。 

解答例

調和近似では、ポテンシャルエネルギーは平衡位置\\(u=0\\)の周りで放物線的： \\(V(u) = \frac{1}{2}\Phi u^2\\)。このポテンシャルは\\(u \to -u\\)に対して対称なので、 熱平均した変位は\\(\langle u \rangle = 0\\)となります。非調和項（例：\\(u^3, u^4\\)）が あると対称性が破れ、\\(\langle u \rangle \neq 0\\)となり、熱膨張が生じます。 

### 演習 2.2: SCP方程式の反復

**問題：**

提供されたPythonコード（SimpleSCP）を実行し、4次力定数\\(\Phi_4\\)を変化させて （正、ゼロ、負）、温度依存性がどう変わるかプロットせよ。 

ヒント

\\(\Phi_4 > 0\\)：フォノンは温度とともにハード化（振動数増加）  
\\(\Phi_4 = 0\\)：温度依存性なし（調和近似）  
\\(\Phi_4 < 0\\)：フォノンは温度とともにソフト化（振動数減少、ある温度で不安定） 

### 演習 2.3: ランダウ自由エネルギー

**問題：**

1次相転移のランダウ自由エネルギー \\(F(\eta, T) = \frac{1}{2}a(T - T_c)\eta^2 - \frac{1}{4}|\\!b\\!|\eta^4 + \frac{1}{6}c\eta^6\\) （\\(b < 0, c > 0\\)）において、転移温度\\(T_t\\)（2つの極小のエネルギーが等しくなる温度）を 求めよ。 

解答例

\\(T > T_c\\)では\\(\eta = 0\\)が最小、\\(T < T_c\\)では有限の\\(\eta_0\\)が最小になります。 2つの極小\\(F(0, T_t) = F(\eta_0, T_t)\\)となる温度\\(T_t\\)が実際の転移温度です。 具体的には、\\(T_t = T_c - \frac{b^2}{4ac}\\)（\\(b < 0\\)）となり、\\(T_t < T_c\\)です。 これにより、過冷却が起こり得ます。 

### 演習 2.4: ソフトモードの同定

**問題：**

BaTiO₃の強誘電転移とSrTiO₃の構造転移におけるソフトモードの違い （q点、原子変位パターン、転移のタイプ）をまとめよ。 

解答例 特性 | BaTiO₃ | SrTiO₃  
---|---|---  
q点 | Γ点 (0,0,0) | R点 (π/a, π/a, π/a)  
変位 | Tiの極性変位 | 酸素八面体の回転  
物性 | 強誘電性 | 非極性（構造転移のみ）  
単位胞 | 変化なし | 2倍  
  
### 演習 2.5: 負熱膨張の機構

**問題：**

ZrW₂O₈の負熱膨張が、剛体ユニットモード（RUM）によって説明されることを、 Grüneisenパラメータの符号と関連付けて説明せよ。 

解答例

RUMは低振動数の横振動モードで、剛体ユニット（八面体、四面体）間の 結合角が変化します。このモードは、体積が減少すると振動数が下がるため、 負のGrüneisenパラメータ\\(\gamma < 0\\)を持ちます。熱膨張係数は \\(\alpha \propto \sum_i \gamma_i C_{V,i}\\)なので、負の\\(\gamma_i\\)を持つ RUMの寄与が支配的だと、全体として\\(\alpha < 0\\)（負熱膨張）になります。 

### 演習 2.6: SSCHAの適用（発展）

**問題：**

SSCHAを用いて、ある材料の構造相転移を予測する場合、どのような量を計算し、 どのような基準で転移を判定するか、手順を説明せよ。 

解答例

  1. **温度スキャン** ：複数の温度で自由エネルギー最小化
  2. **構造変化の観測** ：平衡原子座標の温度依存性
  3. **自由エネルギーヘッセ行列** ：各温度で計算し、固有値をチェック
  4. **不安定性の判定** ：負の固有値（虚振動数）が現れる温度が転移点の兆候
  5. **複数の極小の比較** ：異なる構造の自由エネルギーを比較し、最安定相を決定


転移温度は、2つの相の自由エネルギーが等しくなる点です。 

## 16\. 参考文献

### 教科書・レビュー

  1. Dove, M. T. (1993). _Introduction to Lattice Dynamics_. Cambridge University Press.   
格子力学の基礎、非調和効果の導入
  2. Wallace, D. C. (1998). _Thermodynamics of Crystals_. Dover Publications.   
統計力学的視点からの格子振動理論
  3. Baroni, S., de Gironcoli, S., Dal Corso, A., & Giannozzi, P. (2001). Phonons and related crystal properties from density-functional perturbation theory. _Reviews of Modern Physics_ , 73(2), 515.   
第一原理フォノン計算の基礎理論
  4. Hellman, O., Abrikosov, I. A., & Simak, S. I. (2011). Lattice dynamics of anharmonic solids from first principles. _Physical Review B_ , 84(18), 180301.   
TDEP法の原論文
  5. Errea, I., Calandra, M., & Mauri, F. (2014). Anharmonic free energies and phonon dispersions from the stochastic self-consistent harmonic approximation. _Physical Review B_ , 89(6), 064302.   
SSCHA法の理論的基礎


### ソフトモードと相転移

  6. Bruce, A. D., & Cowley, R. A. (1981). _Structural Phase Transitions_. Taylor & Francis.   
構造相転移の包括的教科書
  7. Shirane, G., Axe, J. D., Harada, J., & Remeika, J. P. (1970). Soft ferroelectric modes in lead titanate. _Physical Review B_ , 2(1), 155.   
PbTiO₃のソフトモードの古典的研究
  8. Müller, K. A., & Burkard, H. (1979). SrTiO₃: An intrinsic quantum paraelectric below 4 K. _Physical Review B_ , 19(7), 3593.   
SrTiO₃の量子常誘電性


### 負熱膨張

  9. Mary, T. A., Evans, J. S. O., Vogt, T., & Sleight, A. W. (1996). Negative thermal expansion from 0.3 to 1050 Kelvin in ZrW₂O₈. _Science_ , 272(5258), 90-92.   
ZrW₂O₈の負熱膨張の発見
  10. Dove, M. T., & Fang, H. (2016). Negative thermal expansion and associated anomalous physical properties. _Reports on Progress in Physics_ , 79(6), 066503.   
負熱膨張の包括的レビュー


### 計算ツール

  11. Tadano, T., Gohda, Y., & Tsuneyuki, S. (2014). Anharmonic force constants extracted from first-principles molecular dynamics. _Journal of the Physical Society of Japan_ , 83(2), 023702.   
ALAMODEの原論文
  12. Bianco, R., Errea, I., Paulatto, L., Calandra, M., & Mauri, F. (2017). Second-order structural phase transitions, free energy curvature, and temperature-dependent anharmonic phonons. _Physical Review B_ , 96(1), 014111.   
SSCHAによる相転移解析


### オンラインリソース

  * ALAMODE: [https://alamode.readthedocs.io/](https://alamode.readthedocs.io/)
  * SSCHA: [http://sscha.eu/](http://sscha.eu/)
  * TDEP: [https://ollehellman.github.io/program/index.html](https://ollehellman.github.io/program/index.html)

---

**ナビゲーション**

[← 第1章](chapter-1.md) | [目次](index.md) | [第3章 →](chapter-3.md)

---

## 免責事項

このコンテンツはAIの支援を受けて作成されており、正確性を保証するものではありません。重要な情報については一次資料や査読済み文献で確認することをお勧めします。
