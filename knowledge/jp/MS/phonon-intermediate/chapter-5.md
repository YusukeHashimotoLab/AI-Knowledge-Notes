---
title: "第5章：フォノン分光法"
chapter_title: "第5章：フォノン分光法"
subtitle: "中性子散乱、光学分光法、最新技術を含むフォノン特性測定の実験技術"
---

[🌐 EN](../../../en/MS/phonon-intermediate/chapter-5.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン物理学（中級）](index.md) > 第5章

# 第5章：フォノン分光法

**中性子散乱、光学分光法、最新技術を含むフォノン特性測定の実験技術**

## 学習目標

  * 非弾性中性子散乱（INS）の理論と断面積の計算方法を理解する
  * 三軸分光器と飛行時間分光器の動作原理と違いを説明できる
  * コヒーレント散乱とインコヒーレント散乱の違いを識別できる
  * ラマン散乱と赤外吸収の量子論と選択則を理解する
  * 電子エネルギー損失分光法（EELS）と非弾性X線散乱（IXS）の原理を把握する
  * 時間分解フォノン分光法の応用を理解する
  * 異なる分光技術を適切に選択できる
  * Pythonを用いてフォノンスペクトルをシミュレーションできる

## 目次

  1. [フォノン分光法の概要](<#introduction>)
  2. [非弾性中性子散乱（INS）](<#neutron-scattering>)
  3. [ラマン散乱分光法](<#raman-spectroscopy>)
  4. [赤外吸収分光法](<#ir-spectroscopy>)
  5. [電子エネルギー損失分光法（EELS）](<#eels>)
  6. [非弾性X線散乱（IXS）](<#ixs>)
  7. [時間分解フォノン分光法](<#time-resolved>)
  8. [フォノンスペクトルシミュレーション](<#simulation>)
  9. [技術の比較と選択](<#comparison>)
  10. [まとめ](<#summary>)
  11. [演習問題](<#exercises>)

## 1\. フォノン分光法の概要

フォノン分光法は、物質中の格子振動（フォノン）を実験的に観測し、 その分散関係やダイナミクスを明らかにする技術群です。異なるプローブ （中性子、光子、電子）を用いることで、様々な空間・エネルギースケールの フォノン情報を得ることができます。 

### 1.1 プローブ粒子の特性
    
    
    ```mermaid
    graph LR
                    A[プローブ粒子] --> B[中性子]
                    A --> C[光子]
                    A --> D[電子]
                    B --> E[熱中性子: E~25meV]
                    B --> F[波長: ~2Å]
                    C --> G1[可視光: E~2eV]
                    C --> G2[X線: E~10keV]
                    D --> H[電子: E~100keV]
    ```

技術 | プローブ | 運動量分解能 | エネルギー分解能 | 情報深さ  
---|---|---|---|---  
INS | 中性子 | 高（全BZ） | 高（~0.1 meV） | バルク（cm）  
Raman | 可視光 | 低（q≈0） | 中（~1 cm⁻¹） | ~1 μm  
IR | 赤外光 | 低（q≈0） | 中（~1 cm⁻¹） | ~10 μm  
EELS | 電子 | 中（~10 nm⁻¹） | 低（~10 meV） | ~10 nm  
IXS | X線 | 中（~1 nm⁻¹） | 中（~1 meV） | ~100 μm  
  
### 1.2 エネルギー・運動量保存則

すべての非弾性散乱において、以下の保存則が成立します：

$$ \begin{aligned} &\text{エネルギー保存：} \quad \hbar\omega = E_i - E_f \\\ &\text{運動量保存：} \quad \hbar\mathbf{q} = \mathbf{k}_i - \mathbf{k}_f + \mathbf{G} \end{aligned} $$ 

ここで、\\(E_i, \mathbf{k}_i\\)は入射粒子のエネルギーと波数ベクトル、 \\(E_f, \mathbf{k}_f\\)は散乱後のそれら、\\(\hbar\omega\\)と\\(\hbar\mathbf{q}\\)は フォノンのエネルギーと運動量、\\(\mathbf{G}\\)は逆格子ベクトルです。 

## 2\. 非弾性中性子散乱（INS）

### 2.1 中性子散乱の理論

中性子は電荷を持たないため、原子核との相互作用を通じて散乱されます。 この性質により、中性子は物質中を深く透過でき、バルクのフォノン情報を 得ることができます。 

#### 2.1.1 散乱断面積

一フォノン散乱の部分微分断面積は以下で与えられます：

$$ \frac{d^2\sigma}{d\Omega dE_f} = \frac{k_f}{k_i} \sum_{\mathbf{q},j} \frac{(\hbar q)^2}{2M\omega_{\mathbf{q}j}} \left|\sum_d b_d e^{-W_d} \frac{\mathbf{q} \cdot \mathbf{e}_{\mathbf{q}j}(d)}{\sqrt{M_d}} e^{i\mathbf{q} \cdot \mathbf{r}_d}\right|^2 S(\mathbf{q},\omega) $$ 

ここで、

  * \\(k_i, k_f\\)：入射・散乱中性子の波数
  * \\(b_d\\)：原子\\(d\\)の散乱長
  * \\(W_d\\)：Debye-Waller因子
  * \\(\mathbf{e}_{\mathbf{q}j}(d)\\)：フォノン固有ベクトル
  * \\(M_d\\)：原子\\(d\\)の質量

#### 動的構造因子

動的構造因子\\(S(\mathbf{q},\omega)\\)は、系の時間・空間相関を記述し、 一フォノン散乱では以下の形をとります： 

$$ S(\mathbf{q},\omega) = \frac{1}{2}\left[\left(n_{\mathbf{q}j} + 1\right)\delta(\omega - \omega_{\mathbf{q}j}) \+ n_{\mathbf{q}j}\delta(\omega + \omega_{\mathbf{q}j})\right] $$ 

\\(n_{\mathbf{q}j}\\)はBose-Einstein分布関数です。

#### 2.1.2 コヒーレント散乱とインコヒーレント散乱

散乱断面積は、コヒーレント成分とインコヒーレント成分に分解されます：

$$ \sigma = \sigma_{\text{coh}} + \sigma_{\text{inc}} = 4\pi\overline{b}^2 + 4\pi(\overline{b^2} - \overline{b}^2) $$ 

散乱タイプ | 物理的起源 | 得られる情報  
---|---|---  
コヒーレント散乱 | 異なる原子間の位相干渉 | フォノン分散関係\\(\omega(\mathbf{q})\\)  
インコヒーレント散乱 | 同一核種の異なる同位体、スピン状態 | 振動状態密度（VDOS）  
  
**例：** 水素（H）は大きなインコヒーレント散乱断面積 (\\(\sigma_{\text{inc}} = 80.3\\) barn)を持つため、水素を含む物質では VDOSの測定に有利ですが、分散関係の測定には重水素（D）置換が必要です。 

### 2.2 三軸分光器

三軸分光器（Triple-Axis Spectrometer, TAS）は、入射エネルギー、 散乱角、散乱エネルギーを独立に制御できる装置で、運動量空間の 任意の点を高精度で測定できます。 
    
    
    ```mermaid
    graph LR
                    A[原子炉/加速器] --> B[モノクロメータ]
                    B --> C[試料]
                    C --> D[アナライザー]
                    D --> E[検出器]
                    B -.-> F[ki選択]
                    D -.-> G[kf選択]
                    C -.-> H[散乱角 2θ]
    ```

#### 動作原理

  1. **モノクロメータ：** ブラッグ反射により入射中性子のエネルギー\\(E_i\\)を選択
  2. **試料台：** 散乱角\\(2\theta\\)を設定し、運動量移行\\(\mathbf{q}\\)を決定
  3. **アナライザー：** 散乱中性子のエネルギー\\(E_f\\)を選択
  4. **検出器：** 強度を測定

### 2.3 飛行時間分光器

飛行時間分光器（Time-of-Flight Spectrometer, TOF）は、パルス中性子源を用い、 中性子の飛行時間からエネルギーを決定します。広いエネルギー・運動量範囲を 同時に測定できる利点があります。 

#### エネルギー決定

$$ E = \frac{1}{2}m_n v^2 = \frac{1}{2}m_n \left(\frac{L}{t}\right)^2 $$ 

ここで、\\(m_n\\)は中性子質量、\\(L\\)は飛行距離、\\(t\\)は飛行時間です。 エネルギー分解能は： 

$$ \frac{\Delta E}{E} = 2\left(\frac{\Delta t}{t} + \frac{\Delta L}{L}\right) $$ 

### 2.4 測定例：シリコンのフォノン分散

#### シリコンの光学フォノン測定

三軸分光器を用いてSi(111)方向のLO（縦光学）フォノンを測定する場合： 

  * 入射エネルギー：\\(E_i = 35\\) meV
  * 散乱ベクトル：\\(\mathbf{q} = (1,1,1)\\) + \\(\delta\mathbf{q}\\)
  * エネルギー移行：\\(\hbar\omega \approx 64\\) meV (ゾーン中心)

\\(\delta\mathbf{q}\\)を変えながらスキャンすることで、 [111]方向の分散曲線が得られます。 

## 3\. ラマン散乱分光法

### 3.1 ラマン散乱の量子論

ラマン散乱は、光子とフォノンの非弾性散乱過程です。入射光子が物質と 相互作用し、フォノンを生成（Stokes散乱）または吸収（anti-Stokes散乱） しながら散乱されます。 

#### 3.1.1 散乱断面積

ラマン散乱の微分断面積は、偏極率テンソル\\(\alpha\\)の変化を通じて与えられます：

$$ \frac{d\sigma}{d\Omega} \propto \left|\frac{\partial \alpha_{ij}}{\partial Q_{\mathbf{q}j}}\right|^2 (n_{\mathbf{q}j} + 1) $$ 

ここで、\\(Q_{\mathbf{q}j}\\)は規準座標、\\(n_{\mathbf{q}j}\\)はBose-Einstein分布関数です。 Stokes散乱の強度は\\((n + 1)\\)、anti-Stokes散乱は\\(n\\)に比例します。 

#### 3.1.2 ラマンテンソル

ラマン散乱強度は、入射・散乱光の偏光とラマンテンソルの積で決まります：

$$ I \propto \left|\mathbf{e}_s \cdot \mathbf{R} \cdot \mathbf{e}_i\right|^2 $$ 

\\(\mathbf{e}_i, \mathbf{e}_s\\)は入射・散乱光の偏光ベクトル、 \\(\mathbf{R}\\)はラマンテンソルです。 

### 3.2 ラマン選択則

光学フォノンがラマン活性であるためには、対称性による選択則を満たす必要があります。 結晶の点群対称性により、ラマン活性なモードが決定されます。 

#### 選択則の物理的意味

フォノンモードがラマン活性であるためには、そのモードによる原子変位が 偏極率テンソル\\(\alpha\\)を変化させる必要があります。これは結晶の対称性と 密接に関連しています。 

#### 3.2.1 具体例：ダイヤモンド構造

ダイヤモンド構造（点群\\(O_h\\)）では、ゾーン中心に以下のモードがあります： 

  * 1つの\\(F_{2g}\\)モード（ラマン活性）
  * 1つの\\(F_{1u}\\)モード（赤外活性）

シリコンのラマンスペクトルでは、約520 cm⁻¹に鋭いピークが観測され、 これが\\(F_{2g}\\)三重縮退光学フォノンに対応します。 

### 3.3 一次および二次ラマン過程

#### 3.3.1 一次ラマン散乱

一フォノン過程で、ほぼ\\(\mathbf{q} \approx 0\\)（ゾーン中心）のフォノンのみが 観測されます。これは可視光の波長（~500 nm）がフォノンの波長（~数Å）に 比べて非常に長いためです。 

#### 3.3.2 二次ラマン散乱

二フォノン過程では、\\(\mathbf{q}_1 + \mathbf{q}_2 \approx 0\\)を満たす 2つのフォノンが関与します。これにより、ブリルアンゾーン全体の フォノン状態密度の情報が得られます。 

$$ \omega_{\text{2nd}} = \omega_{\mathbf{q}_1,j_1} + \omega_{\mathbf{q}_2,j_2} $$ 

例えば、シリコンでは約300 cm⁻¹と約950 cm⁻¹に二次ラマンピークが 観測されます。 

### 3.4 共鳴ラマン散乱

入射光子のエネルギーが電子遷移エネルギーに近い場合、ラマン散乱強度は 数桁増大します。これを共鳴ラマン効果と呼びます。 

$$ I_{\text{resonance}} \propto \frac{1}{(E_{\text{laser}} - E_{\text{gap}})^2 + \Gamma^2} $$ 

ここで、\\(E_{\text{gap}}\\)はバンドギャップ、\\(\Gamma\\)は線幅です。 この効果は、特定のフォノンモードを選択的に増強するために利用されます。 

### 3.5 実験配置と測定例
    
    
    ```mermaid
    graph LR
                    A[レーザー光源] --> B[フィルター]
                    B --> C[対物レンズ]
                    C --> D[試料]
                    D --> E[集光レンズ]
                    E --> F[ノッチフィルター]
                    F --> G[分光器]
                    G --> H[CCD検出器]
    ```

#### グラフェンのラマンスペクトル

グラフェンは特徴的な3つのピークを示します：

  * **Dバンド** （~1350 cm⁻¹）：欠陥由来の二次過程
  * **Gバンド** （~1580 cm⁻¹）：E₂g面内伸縮振動
  * **2Dバンド** （~2700 cm⁻¹）：二次過程、層数に敏感

2Dバンドの形状から、グラフェンの層数を非破壊的に決定できます。 

## 4\. 赤外吸収分光法

### 4.1 IR吸収の理論

赤外吸収分光法は、物質が赤外光を吸収してフォノンを励起する現象を 利用します。吸収が起こるためには、フォノンモードが双極子モーメントの 変化を伴う必要があります。 

#### 4.1.1 双極子選択則

IR活性であるための条件は：

$$ \left|\frac{\partial \mathbf{\mu}}{\partial Q_{\mathbf{q}j}}\right| \neq 0 $$ 

ここで、\\(\mathbf{\mu}\\)は双極子モーメント、\\(Q_{\mathbf{q}j}\\)は規準座標です。 この条件から、中心対称性を持つ結晶では、ラマン活性モードとIR活性モードは 互いに排他的です（相互排他律）。 

#### 相互排他律

中心対称性を持つ結晶（例：ダイヤモンド、NaCl）では： 

  * 偶パリティのモード → ラマン活性、IR不活性
  * 奇パリティのモード → IR活性、ラマン不活性

この規則により、ラマンとIRスペクトルは相補的な情報を提供します。 

### 4.2 吸収係数とフォノンパラメータ

赤外吸収係数は以下で与えられます：

$$ \alpha(\omega) = \sum_j \frac{S_j \omega_j^2 \gamma_j}{(\omega_j^2 - \omega^2)^2 + \omega^2\gamma_j^2} $$ 

ここで、

  * \\(S_j\\)：振動子強度（双極子モーメント変化に比例）
  * \\(\omega_j\\)：フォノン周波数
  * \\(\gamma_j\\)：減衰定数（フォノン寿命に反比例）

### 4.3 極性材料のポラリトン分散

極性結晶（イオン結晶）では、光学フォノンと電磁波が結合してフォノン ポラリトンと呼ばれる混成モードを形成します。 

#### 4.3.1 ポラリトン分散関係

横波ポラリトンの分散関係は：

$$ \omega^2 = \frac{c^2 q^2}{\epsilon_\infty} + \frac{\omega_{\text{TO}}^2 (\epsilon_0 - \epsilon_\infty)}{\epsilon_0} $$ 

ここで、\\(\epsilon_0\\)と\\(\epsilon_\infty\\)は静的および高周波誘電率、 \\(\omega_{\text{TO}}\\)は横光学フォノン周波数です。 

#### 4.3.2 Lyddane-Sachs-Teller関係

極性結晶では、横光学（TO）および縦光学（LO）フォノンの周波数と 誘電率の間に以下の関係が成立します： 

$$ \frac{\omega_{\text{LO}}^2}{\omega_{\text{TO}}^2} = \frac{\epsilon_0}{\epsilon_\infty} $$ 

この関係は、IR反射スペクトルからLO周波数を決定するのに利用されます。 

#### GaAsのフォノンポラリトン

GaAsでは：

  * \\(\omega_{\text{TO}} = 268\\) cm⁻¹
  * \\(\omega_{\text{LO}} = 292\\) cm⁻¹
  * \\(\epsilon_0 = 12.9\\)、\\(\epsilon_\infty = 10.9\\)

LST関係：\\((292/268)^2 = 1.189 \approx 12.9/10.9 = 1.183\\) で良好に一致します。 

### 4.4 測定技術

#### 4.4.1 フーリエ変換赤外分光法（FTIR）

FTIRは、Michelson干渉計を用いて全波長域を同時に測定し、 フーリエ変換により吸収スペクトルを得ます。高速測定と高S/N比が利点です。 

#### 4.4.2 反射・透過測定

  * **透過測定：** 薄膜試料で吸収係数を直接測定
  * **反射測定：** バルク試料で複素誘電率を決定
  * **ATR法：** 高吸収試料の表面近傍を測定

## 5\. 電子エネルギー損失分光法（EELS）

### 5.1 EELSの原理

EELS（Electron Energy Loss Spectroscopy）は、高エネルギー電子線を試料に 照射し、非弾性散乱により失われたエネルギーを測定することで、試料の 電子状態やフォノンを調べる技術です。 

#### 5.1.1 損失関数

EELSで測定される強度は、誘電関数の虚部に関連します：

$$ I(\mathbf{q}, \omega) \propto \text{Im}\left[-\frac{1}{\epsilon(\mathbf{q}, \omega)}\right] $$ 

ここで、\\(\epsilon(\mathbf{q}, \omega)\\)は誘電関数です。フォノン励起による ピークは、この損失関数に現れます。 

### 5.2 表面フォノンの測定

EELSは表面に敏感で、表面フォノンの測定に適しています。表面フォノンは バルクフォノンとは異なる分散関係を持ち、表面再構成や吸着種の影響を 受けます。 

#### Fuchs-Kliewer表面フォノン

極性結晶の表面には、以下の周波数を持つ表面フォノンが存在します：

$$ \omega_{\text{surface}} = \sqrt{\frac{\omega_{\text{LO}}^2 + \omega_{\text{TO}}^2}{2}} $$ 

### 5.3 空間分解能とナノスケール測定

走査透過電子顕微鏡（STEM）と組み合わせたEELSは、ナノメートルスケールの 空間分解能を持ちます。これにより、以下が可能になります： 

  * ナノ構造のローカルフォノン測定
  * 界面や粒界のフォノン特性評価
  * 欠陥周辺のフォノン状態変化の観察

**最近の進展：** モノクロメーター付きSTEM-EELSにより、 エネルギー分解能が~10 meVまで向上し、より詳細なフォノン測定が 可能になっています。 

## 6\. 非弾性X線散乱（IXS）

### 6.1 IXSの特徴

非弾性X線散乱（Inelastic X-ray Scattering, IXS）は、高エネルギーX線を用いた フォノン測定技術です。第三世代放射光施設の高輝度X線により、高精度測定が 可能になりました。 

#### 6.1.1 IXSの利点

  * **元素選択性：** すべての元素に適用可能（中性子の同位体依存性なし）
  * **高運動量分解能：** ~10⁻³ Å⁻¹の精度
  * **微小試料測定：** μmサイズの試料でも測定可能
  * **極限環境：** 高圧、高温での測定が容易

### 6.2 散乱断面積

IXSの二重微分散乱断面積は：

$$ \frac{d^2\sigma}{d\Omega dE_f} = r_0^2 \frac{E_f}{E_i} S(\mathbf{q}, \omega) $$ 

ここで、\\(r_0\\)は古典電子半径、動的構造因子\\(S(\mathbf{q}, \omega)\\)は 中性子散乱と同じ形式です。 

### 6.3 測定技術と装置

#### 6.3.1 バックスキャッタリング分光器

高エネルギー分解能（~1 meV）を得るため、ブラッグ角が90°に近い バックスキャッタリング配置を用います： 

$$ \frac{\Delta E}{E} \approx \cot\theta_B \Delta\theta_B $$ 

\\(\theta_B \approx 90°\\)では\\(\cot\theta_B \to 0\\)となり、高分解能が実現されます。

### 6.4 測定例：高圧下のフォノン

#### ダイヤモンドアンビルセル中の鉄

IXSを用いて、地球中心核の圧力条件（~360 GPa）下での鉄のフォノンを 測定することができます。これにより： 

  * 高圧相の弾性定数を決定
  * 地球内部の地震波速度と比較
  * 核の組成に制約を与える

## 7\. 時間分解フォノン分光法

### 7.1 ポンプ-プローブ分光法

時間分解フォノン分光法は、超短パルスレーザーを用いてフォノンダイナミクスを ピコ秒からフェムト秒の時間スケールで観測する技術です。 

#### 7.1.1 測定原理
    
    
    ```mermaid
    sequenceDiagram
                    participant Pump as ポンプパルス
                    participant Sample as 試料
                    participant Probe as プローブパルス
                    participant Detector as 検出器
    
                    Pump->>Sample: 励起（t=0）
                    Note over Sample: フォノン生成
                    Note over Sample: 待機時間 Δt
                    Probe->>Sample: 測定（t=Δt）
                    Sample->>Detector: 反射/透過
                    Note over Detector: Δtを走査して動的過程を追跡
    ```

  1. **ポンプパルス：** 試料を励起し、フォノンを生成
  2. **遅延時間：** 可変遅延ステージで制御
  3. **プローブパルス：** フォノン状態を光学的に測定
  4. **時間走査：** 遅延時間を変えて動的過程を再構成

### 7.2 コヒーレント光学フォノンの生成と検出

フェムト秒パルス励起により、位相の揃ったコヒーレント光学フォノンが 生成されます。その振動は、試料の反射率や透過率の振動として観測されます。 

#### 7.2.1 検出信号

$$ \Delta R(t) = \sum_j A_j e^{-\gamma_j t} \cos(\omega_j t + \phi_j) $$ 

ここで、\\(A_j\\)は振幅、\\(\gamma_j\\)は減衰率、\\(\omega_j\\)はフォノン周波数、 \\(\phi_j\\)は初期位相です。フーリエ変換により周波数スペクトルが得られます。 

### 7.3 フォノン寿命と非平衡ダイナミクス

時間分解測定から以下の情報が得られます：

  * **フォノン寿命：** 減衰率\\(\gamma_j\\)から決定
  * **フォノン-フォノン散乱：** 温度依存性から散乱過程を解析
  * **非平衡状態：** 励起直後の非熱的分布の時間発展
  * **相転移ダイナミクス：** 構造相転移の超高速過程

#### ビスマスのA₁gモード

ビスマスでは、完全対称A₁gモードのコヒーレント振動が観測されます： 

  * 周波数：~2.9 THz（~97 cm⁻¹）
  * 寿命：室温で~7 ps
  * 励起メカニズム：電子系の突然の遮蔽変化（DECP機構）

温度上昇とともに寿命が短くなり、非調和フォノン散乱の温度依存性が 明らかになります。 

### 7.4 時間分解X線・電子回折

超短パルスX線や電子線を用いることで、原子の動きを直接観測できます。 

  * **時間分解X線回折：** 格子定数の変化から歪み波を観測
  * **超高速電子回折（UED）：** 構造因子の時間変化から原子変位を決定
  * **X線自由電子レーザー（XFEL）：** フェムト秒時間分解能でのフォノン測定

## 8\. フォノンスペクトルシミュレーション

### 8.1 フォノン分散のモデル計算

簡単なモデルを用いて、フォノン分散関係とスペクトルをシミュレート してみましょう。ここでは、1次元単原子鎖と2原子鎖のモデルを実装します。 

#### 8.1.1 1次元単原子鎖
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # パラメータ設定
    M = 1.0  # 原子質量（任意単位）
    K = 1.0  # ばね定数（任意単位）
    a = 1.0  # 格子定数（任意単位）
    
    # 波数ベクトル
    q = np.linspace(-np.pi/a, np.pi/a, 200)
    
    # 分散関係
    omega = 2 * np.sqrt(K/M) * np.abs(np.sin(q*a/2))
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(q*a/np.pi, omega, 'b-', linewidth=2)
    plt.xlabel('q (π/a)', fontsize=14)
    plt.ylabel('ω (√K/M)', fontsize=14)
    plt.title('1次元単原子鎖のフォノン分散', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"ゾーン中心での群速度: {2*np.sqrt(K/M)*a/2:.3f} √(K/M)·a")
    print(f"ゾーン境界での周波数: {2*np.sqrt(K/M):.3f} √(K/M)")

#### 8.1.2 1次元2原子鎖（光学・音響モード）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # パラメータ設定
    M1 = 1.0   # 原子1の質量
    M2 = 2.0   # 原子2の質量
    K = 1.0    # ばね定数
    a = 2.0    # 格子定数（2原子含む）
    
    # 波数ベクトル
    q = np.linspace(-np.pi/a, np.pi/a, 200)
    
    # 分散関係の計算
    def phonon_dispersion_diatomic(q, M1, M2, K, a):
        """2原子鎖のフォノン分散を計算"""
        M_sum = M1 + M2
        M_prod = M1 * M2
    
        # ω²の2つの解（光学・音響ブランチ）
        term1 = K * M_sum / M_prod
        term2 = K * np.sqrt(M_sum**2 - 4*M_prod*np.sin(q*a/2)**2) / M_prod
    
        omega_optical = np.sqrt(term1 + term2)
        omega_acoustic = np.sqrt(term1 - term2)
    
        return omega_acoustic, omega_optical
    
    omega_ac, omega_op = phonon_dispersion_diatomic(q, M1, M2, K, a)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(q*a/np.pi, omega_ac, 'b-', linewidth=2, label='音響ブランチ')
    plt.plot(q*a/np.pi, omega_op, 'r-', linewidth=2, label='光学ブランチ')
    plt.xlabel('q (π/a)', fontsize=14)
    plt.ylabel('ω (√K/M)', fontsize=14)
    plt.title('1次元2原子鎖のフォノン分散', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ゾーン中心の周波数
    omega_op_center = np.sqrt(2*K*(M1+M2)/(M1*M2))
    omega_ac_center = 0
    
    print(f"光学フォノン（q=0）: {omega_op_center:.3f} √(K/M)")
    print(f"音響フォノン（q=0）: {omega_ac_center:.3f}")
    print(f"ゾーン境界ギャップ: {omega_op[-1] - omega_ac[-1]:.3f}")

### 8.2 振動状態密度（VDOS）の計算

フォノン分散から振動状態密度を計算します。これはINSのインコヒーレント 散乱やIRの二次吸収と比較できます。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    
    def calculate_vdos(q, omega, omega_bins):
        """
        フォノン分散から振動状態密度を計算
    
        Parameters:
        -----------
        q : array
            波数ベクトル
        omega : array or tuple
            角周波数（単一配列または複数ブランチのタプル）
        omega_bins : array
            DOSを計算する周波数ビン
    
        Returns:
        --------
        dos : array
            振動状態密度
        """
        dos = np.zeros_like(omega_bins)
    
        # 複数ブランチの場合
        if isinstance(omega, tuple):
            omega_list = omega
        else:
            omega_list = [omega]
    
        for omega_branch in omega_list:
            # ヒストグラムでDOSを計算
            hist, _ = np.histogram(omega_branch, bins=omega_bins)
            dos += hist
    
        # 群速度による補正（より正確なDOS）
        # DOS(ω) ∝ 1/|dω/dq|
    
        return dos / (omega_bins[1] - omega_bins[0])
    
    # 2原子鎖の例
    q = np.linspace(-np.pi/a, np.pi/a, 1000)
    omega_ac, omega_op = phonon_dispersion_diatomic(q, M1, M2, K, a)
    
    # DOSの計算
    omega_bins = np.linspace(0, 2.0, 200)
    dos = calculate_vdos(q, (omega_ac, omega_op), omega_bins)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 分散関係
    ax1.plot(q*a/np.pi, omega_ac, 'b-', linewidth=2, label='音響')
    ax1.plot(q*a/np.pi, omega_op, 'r-', linewidth=2, label='光学')
    ax1.set_xlabel('q (π/a)', fontsize=12)
    ax1.set_ylabel('ω (√K/M)', fontsize=12)
    ax1.set_title('フォノン分散', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # VDOS
    ax2.plot(omega_bins, dos, 'k-', linewidth=2)
    ax2.fill_between(omega_bins, dos, alpha=0.3)
    ax2.set_xlabel('ω (√K/M)', fontsize=12)
    ax2.set_ylabel('状態密度（任意単位）', fontsize=12)
    ax2.set_title('振動状態密度', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

### 8.3 ラマンスペクトルのシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def lorentzian(omega, omega0, gamma, intensity=1.0):
        """
        ローレンツ関数（フォノンピークの線形）
    
        Parameters:
        -----------
        omega : array
            周波数
        omega0 : float
            ピーク中心周波数
        gamma : float
            半値全幅（FWHM）
        intensity : float
            ピーク強度
        """
        return intensity * (gamma/2)**2 / ((omega - omega0)**2 + (gamma/2)**2)
    
    def bose_einstein(omega, T):
        """
        Bose-Einstein分布関数
    
        Parameters:
        -----------
        omega : array
            周波数
        T : float
            温度（K）
        """
        hbar = 1.054571817e-34  # J·s
        kB = 1.380649e-23       # J/K
    
        # cm⁻¹からJへの変換
        omega_J = omega * 100 * 3e8 * hbar * 2 * np.pi
    
        if T == 0:
            return np.zeros_like(omega)
        else:
            x = omega_J / (kB * T)
            return 1.0 / (np.exp(x) - 1)
    
    def raman_spectrum(omega, peaks, T=300):
        """
        ラマンスペクトルをシミュレート
    
        Parameters:
        -----------
        omega : array
            ラマンシフト（cm⁻¹）
        peaks : list of tuples
            (周波数, 線幅, 強度)のリスト
        T : float
            温度（K）
        """
        spectrum_stokes = np.zeros_like(omega)
        spectrum_antistokes = np.zeros_like(omega)
    
        for omega0, gamma, intensity in peaks:
            # Stokes散乱
            n = bose_einstein(np.array([omega0]), T)[0]
            stokes_intensity = intensity * (n + 1)
            spectrum_stokes += lorentzian(omega, omega0, gamma, stokes_intensity)
    
            # Anti-Stokes散乱
            antistokes_intensity = intensity * n
            spectrum_antistokes += lorentzian(-omega, omega0, gamma, antistokes_intensity)
    
        return spectrum_stokes + spectrum_antistokes
    
    # シリコンのラマンスペクトルをシミュレート
    omega = np.linspace(-600, 600, 2000)
    
    # Si: ゾーン中心の3重縮退F₂gモード
    peaks_si = [
        (520, 3, 1.0),  # (周波数 cm⁻¹, FWHM cm⁻¹, 相対強度)
    ]
    
    # 異なる温度でのスペクトル
    temperatures = [77, 300, 500]
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(12, 6))
    
    for T, color in zip(temperatures, colors):
        spectrum = raman_spectrum(omega, peaks_si, T)
        plt.plot(omega, spectrum, color=color, linewidth=2, label=f'T = {T} K')
    
    plt.xlabel('ラマンシフト (cm⁻¹)', fontsize=14)
    plt.ylabel('強度（任意単位）', fontsize=14)
    plt.title('シリコンのラマンスペクトル（温度依存性）', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='レイリー散乱')
    plt.xlim(-600, 600)
    plt.tight_layout()
    plt.show()
    
    # 温度によるピークシフトと線幅の効果を解析
    print("\n温度依存性の分析：")
    for T in temperatures:
        n = bose_einstein(np.array([520]), T)[0]
        ratio = n / (n + 1) if n > 0 else 0
        print(f"T = {T} K: n = {n:.3f}, Anti-Stokes/Stokes = {ratio:.3f}")

### 8.4 分光データの解析例
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def fit_raman_peak(omega, intensity):
        """
        ラマンピークをローレンツ関数でフィッティング
    
        Returns:
        --------
        fit_params : dict
            フィッティングパラメータ（中心周波数、線幅、強度）
        """
        # 初期推定値
        omega0_guess = omega[np.argmax(intensity)]
        gamma_guess = 5.0
        I_guess = np.max(intensity)
        baseline_guess = np.min(intensity)
    
        # フィッティング関数（ベースライン含む）
        def fit_func(x, omega0, gamma, I, baseline):
            return lorentzian(x, omega0, gamma, I) + baseline
    
        # フィッティング実行
        popt, pcov = curve_fit(
            fit_func,
            omega,
            intensity,
            p0=[omega0_guess, gamma_guess, I_guess, baseline_guess]
        )
    
        # 標準偏差
        perr = np.sqrt(np.diag(pcov))
    
        fit_params = {
            'omega0': popt[0],
            'gamma': popt[1],
            'intensity': popt[2],
            'baseline': popt[3],
            'omega0_err': perr[0],
            'gamma_err': perr[1]
        }
    
        return fit_params, fit_func(omega, *popt)
    
    # 疑似実験データの生成（ノイズ付き）
    omega_exp = np.linspace(500, 540, 100)
    true_params = (520, 3, 100, 5)  # 真のパラメータ
    signal = lorentzian(omega_exp, *true_params[:3]) + true_params[3]
    noise = np.random.normal(0, 2, len(omega_exp))
    intensity_exp = signal + noise
    
    # フィッティング
    fit_params, fitted_curve = fit_raman_peak(omega_exp, intensity_exp)
    
    # 結果のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(omega_exp, intensity_exp, 'o', label='実験データ', alpha=0.6)
    plt.plot(omega_exp, fitted_curve, 'r-', linewidth=2, label='フィッティング')
    plt.xlabel('ラマンシフト (cm⁻¹)', fontsize=14)
    plt.ylabel('強度（任意単位）', fontsize=14)
    plt.title('ラマンピークのフィッティング', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 結果の表示
    print("フィッティング結果：")
    print(f"ピーク位置: {fit_params['omega0']:.2f} ± {fit_params['omega0_err']:.2f} cm⁻¹")
    print(f"線幅（FWHM）: {fit_params['gamma']:.2f} ± {fit_params['gamma_err']:.2f} cm⁻¹")
    print(f"強度: {fit_params['intensity']:.2f}")
    print(f"ベースライン: {fit_params['baseline']:.2f}")

## 9\. 技術の比較と選択

### 9.1 技術選択のフローチャート
    
    
    ```mermaid
    graph TD
                    A[測定目的] --> B{運動量分解能が必要？}
                    B -->|Yes| C{試料サイズは？}
                    B -->|No| D{表面 or バルク？}
    
                    C -->|大型 >1mm³| E[INS - 三軸/TOF]
                    C -->|微小 <1mm³| F[IXS]
    
                    D -->|バルク| G{極限環境？}
                    D -->|表面| H[EELS/STEM-EELS]
    
                    G -->|Yes| I[IXS/高圧INS]
                    G -->|No| J{選択則情報も欲しい}
    
                    J -->|Yes| K[Raman + IR]
                    J -->|No| L[Raman]
    
                    E --> M{高圧・高温？}
                    M -->|Yes| N[TOF - Paris-Edinburgh]
                    M -->|No| O[三軸 - 高分解能]
    ```

### 9.2 詳細比較表

項目 | INS | Raman | IR | EELS | IXS  
---|---|---|---|---|---  
**運動量範囲** | 全BZ | q≈0 | q≈0 | 0.1-10 nm⁻¹ | 0.1-10 nm⁻¹  
**エネルギー範囲** | 0.1-500 meV | 10-4000 cm⁻¹ | 50-4000 cm⁻¹ | 10-500 meV | 1-1000 meV  
**エネルギー分解能** | ~0.1 meV | ~1 cm⁻¹ | ~1 cm⁻¹ | ~10-50 meV | ~1 meV  
**空間分解能** | なし | ~1 μm | ~10 μm | ~1 nm | ~1 μm  
**試料サイズ** | mm³-cm³ | μm³-mm³ | mm²-cm² | nm²-μm² | μm³-mm³  
**測定時間** | 時間-日 | 秒-分 | 秒-分 | 分-時間 | 時間-日  
**試料前処理** | 最小限 | 最小限 | 薄片化必要 | 薄片化必須 | 最小限  
**非破壊性** | ◎ | ◎ | ◯ | △ | ◎  
**費用** | 高（施設利用） | 低-中 | 低-中 | 高 | 高（施設利用）  
**アクセス性** | 限定的 | 容易 | 容易 | 中 | 限定的  
  
### 9.3 応用分野別の推奨技術

#### 応用例と最適技術

**1\. 新材料の完全なフォノン分散決定**

  * 第一選択：INS（三軸分光器）
  * 代替案：IXS（微小試料の場合）
  * 補完：Raman/IR（ゾーン中心の詳細）

**2\. 2次元材料・薄膜のフォノン**

  * 第一選択：Raman（層数・歪み評価）
  * 補完：EELS（空間分解）
  * 理論：第一原理計算との比較

**3\. 極限環境（高圧・高温）**

  * 第一選択：IXS（ダイヤモンドアンビル対応）
  * 代替案：Raman（簡便だがq=0のみ）
  * バルク試料：TOF-INS（Paris-Edinburghセル）

**4\. ナノ構造・界面のローカルフォノン**

  * 第一選択：STEM-EELS（~1 nm分解能）
  * 補完：μ-Raman（~1 μm分解能）
  * 理論：分子動力学シミュレーション

**5\. 超高速フォノンダイナミクス**

  * 第一選択：ポンプ-プローブ分光
  * 発展：時間分解X線回折（XFEL）
  * 理論：非平衡Green関数法

### 9.4 複数技術の相補的利用

多くの研究では、複数の分光技術を組み合わせることで、包括的な フォノン理解が得られます。 

#### ペロブスカイト太陽電池材料の研究例

CH₃NH₃PbI₃のフォノン特性を解明するための測定戦略：

  1. **Raman分光（室温）：**
     * 有機カチオンの回転モード
     * 無機フレームワークの振動
     * 相転移の温度依存性
  2. **IR分光：**
     * C-N、N-H伸縮振動
     * Pb-I振動（Ramanで観測困難）
     * 誘電応答とポラリトン
  3. **INS：**
     * 完全なフォノン分散
     * 水素原子のVDOS（大きな断面積）
     * 低エネルギー格子振動
  4. **時間分解分光：**
     * ホットキャリア-フォノン相互作用
     * フォノンボトルネック効果
     * ポーラロン形成ダイナミクス

これらの技術を統合することで、材料の光電変換効率と フォノンの関係が明らかになります。 

## 10\. まとめ

### 本章の要点

#### 1\. フォノン分光法の多様性

  * 中性子、光子、電子という異なるプローブにより、様々な空間・エネルギースケールのフォノン情報が得られる
  * 各技術には固有の長所と制約があり、研究目的に応じた選択が重要

#### 2\. 非弾性中性子散乱（INS）

  * 全ブリルアンゾーンのフォノン分散を直接測定できる最も強力な手法
  * コヒーレント散乱から分散関係、インコヒーレント散乱からVDOSを取得
  * 三軸分光器は高精度、TOF分光器は広範囲測定に適する

#### 3\. 光学分光法（Raman・IR）

  * ゾーン中心（q≈0）のフォノンを測定、選択則により相補的情報を提供
  * Raman：偏極率変化、対称モード活性
  * IR：双極子モーメント変化、極性モード活性
  * 簡便で非破壊、空間分解測定も可能

#### 4\. 先端技術（EELS・IXS）

  * EELS：ナノスケール空間分解能、表面・界面のフォノン測定
  * IXS：微小試料・極限環境での高精度測定、元素選択性
  * 両技術とも運動量分解能を持ち、光学技術を補完

#### 5\. 時間分解分光法

  * ポンプ-プローブ法により、フォノンの生成・緩和ダイナミクスを観測
  * フォノン寿命、非平衡過程、超高速相転移の解明に有用
  * XFELやUEDにより、原子スケールの構造ダイナミクスが直接観測可能に

#### 6\. 技術の選択と統合

  * 運動量分解能、エネルギー分解能、空間分解能、試料サイズなどを考慮
  * 多くの場合、複数技術の相補的利用が最も効果的
  * 実験と理論計算（第一原理・分子動力学）の組み合わせが重要

### 重要な概念

  * **エネルギー・運動量保存則：** すべての非弾性散乱の基礎
  * **動的構造因子：** 時空間相関を記述、各技術で測定される物理量
  * **選択則：** 対称性により観測可能なフォノンモードが決定
  * **Bose-Einstein分布：** 温度によるフォノン占有数、Stokes/anti-Stokes比
  * **分散関係 vs VDOS：** 運動量分解能の有無による情報の違い

### 他章との関連

  * **第1章（格子力学）：** 測定されたフォノン分散から力定数を逆算
  * **第2章（フォノン相互作用）：** 線幅からフォノン寿命と散乱過程を評価
  * **第3章（熱伝導）：** 実測分散から熱伝導率を計算・検証
  * **第4章（電子-フォノン結合）：** 共鳴Ramanや時間分解測定で結合強度を評価

## 11\. 演習問題

### 問題1：基礎理論

(a) 25 meVの熱中性子の波長を計算せよ。この中性子は、どの程度の 格子定数を持つ結晶の構造解析に適しているか？ 

(b) 可視光（波長500 nm）のラマン散乱で、なぜq≈0のフォノンのみが 観測されるのか、ブリルアンゾーンのサイズと比較して説明せよ。 

ヒント

(a) de Broglie波長 λ = h/p、中性子の運動エネルギー E = p²/2m

(b) 典型的な格子定数（~5Å）から第一ブリルアンゾーンの大きさを見積もる

### 問題2：散乱断面積

水素（H）と重水素（D）の中性子散乱長と断面積を比較し、 なぜ重水素化試料がフォノン分散測定に有利なのか説明せよ。 

核種 | コヒーレント散乱長 (fm) | インコヒーレント断面積 (barn)  
---|---|---  
¹H | -3.74 | 80.3  
²H (D) | 6.67 | 2.0  
ヒント

コヒーレント散乱は分散関係、インコヒーレント散乱はVDOSを与える

### 問題3：選択則

立方晶ZnS（閃亜鉛鉱型、点群T_d）のゾーン中心には、 以下のフォノンモードがある： 

  * A₁ (1モード)
  * E (1モード、2重縮退)
  * T₂ (2モード、3重縮退)

(a) どのモードがRaman活性か？  
(b) どのモードがIR活性か？  
(c) 相互排他律は成立するか？なぜか？ 

ヒント

T_d群は中心対称性を持たない。指標表を参照して、x²やxyzなどの変換性を確認

### 問題4：ラマン散乱の温度依存性

シリコンの520 cm⁻¹ラマンピークについて、300 Kと77 Kにおける anti-Stokes/Stokes強度比を計算せよ。 

ヒント：強度比は \\(I_{\text{anti}}/I_{\text{Stokes}} = \exp(-\hbar\omega/k_BT)\\) 

ヒント

520 cm⁻¹ = 0.0645 eV、k_B = 8.617×10⁻⁵ eV/K

### 問題5：フォノン寿命

ラマンスペクトルで観測されたフォノンピークの半値全幅（FWHM）が 3 cm⁻¹であった。このフォノンの寿命を見積もれ。 

ヒント：不確定性原理 \\(\Delta E \cdot \tau \sim \hbar\\) 

解答の方針

1\. FWHMをエネルギー単位（eV）に変換  
2\. \\(\tau = \hbar/\Delta E\\) から寿命を計算  
3\. 結果をpsまたはfs単位で表現 

### 問題6：技術選択

以下の研究課題に対して、最適なフォノン分光技術を選択し、 その理由を説明せよ。複数技術の組み合わせも検討せよ。 

(a) グラフェン単層膜の層数を非破壊的に決定したい

(b) 新規超伝導体の全ブリルアンゾーンのフォノン分散を測定したい

(c) ダイヤモンドアンビルセル中の試料を300 GPaの圧力下で測定したい

(d) 量子ドット（直径5 nm）の局所フォノンモードを観測したい

考慮すべき点

  * 必要な運動量分解能
  * 試料サイズ・形状
  * 測定環境（温度・圧力）
  * 空間分解能の必要性
  * アクセス性・コスト

### 問題7：プログラミング演習

本章のPythonコードを参考に、以下を実装せよ： 

(a) GaAsの2バンドモデル（M_Ga = 69.7, M_As = 74.9, ω_TO = 268 cm⁻¹, ω_LO = 292 cm⁻¹）から、 フォノン分散とVDOSを計算しプロットせよ。 

(b) グラフェンの典型的なラマンスペクトル（Dバンド1350 cm⁻¹、 Gバンド1580 cm⁻¹、2Dバンド2700 cm⁻¹）をシミュレートし、 単層と2層での2Dバンドの違いを表現せよ。 

(c) 温度を100-500 Kで変化させたときの、Stokes/anti-Stokes 強度比の変化をグラフ化せよ（任意のフォノンエネルギーで）。 

拡張課題

時間分解測定のシミュレーション：コヒーレントフォノンの 減衰振動を、複数のモードが重畳した場合でモデル化し、 フーリエ変換によりスペクトルを抽出せよ。 

[← 第4章：電子-フォノン結合](<chapter-4.html>) [目次](<index.html>) 次章準備中
