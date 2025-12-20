---
title: "第3章: 熱輸送計算"
subtitle: "フォノンボルツマン輸送方程式と第一原理熱伝導率計算"
series: "フォノン物理学(上級)"
chapter: 3
language: "ja"
---

[🌐 EN](../../../en/MS/phonon-advanced/chapter-3.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン物理学(上級)](index.md) > 第3章

# 第3章: 熱輸送計算

**フォノンボルツマン輸送方程式と第一原理熱伝導率計算**

## 学習目標

  * フォノンボルツマン輸送方程式（PBTE）の物理的意味と数学的定式化を理解する
  * 緩和時間近似（RTA）と反復解法の違いと適用範囲を説明できる
  * 第一原理計算による三フォノン散乱率の計算手法を習得する
  * 四フォノン散乱の重要性と計算方法を理解する
  * ShengBTE、almaBTE、Phono3pyの使用方法と特徴を把握する
  * 収束問題の診断と解決策を実践できる
  * 平均自由行程分布とコヒーレント/インコヒーレント輸送を解析できる
  * 機械学習を用いた熱伝導率予測手法の基礎を理解する


## 導入

熱伝導率は材料の最も基本的な物性の一つであり、熱電材料、熱管理材料、 電子デバイスの設計において中心的な役割を果たします。フォノンが支配的な 熱キャリアである絶縁体や半導体では、フォノンボルツマン輸送方程式（PBTE）が 熱伝導を記述する基本方程式となります。 

本章では、PBTEの詳細な定式化から始め、第一原理計算に基づく散乱率の 計算手法、現代的なソフトウェアツールの使用方法、そして機械学習を含む 最新のアプローチまでを包括的に扱います。理論的理解と実践的な計算技術を 結びつけることで、材料の熱輸送特性を予測・設計する能力を養います。 

## 3.1 フォノンボルツマン輸送方程式

### 3.1.1 分布関数とボルツマン方程式

フォノンの非平衡状態を記述するために、フォノン分布関数 \(n(\mathbf{q}, j, \mathbf{r}, t)\) を導入します。 これは、時刻 \(t\) に位置 \(\mathbf{r}\) において、波数 \(\mathbf{q}\)、ブランチ \(j\) のフォノンの 平均占有数を表します。 

#### 定義: フォノンボルツマン輸送方程式

フォノン分布関数の時間発展は以下のボルツマン方程式で記述されます：

\[\frac{\partial n}{\partial t} + \mathbf{v}_g \cdot \nabla_\mathbf{r} n + \mathbf{F} \cdot \nabla_\mathbf{q} n = \left(\frac{\partial n}{\partial t}\right)_\text{scatt}\] 

ここで：

  * \(\mathbf{v}_g = \nabla_\mathbf{q}\omega(\mathbf{q}, j)\) : 群速度
  * \(\mathbf{F}\) : 外力（通常はゼロ）
  * \((\partial n/\partial t)_\text{scatt}\) : 散乱項（衝突積分）


### 3.1.2 定常状態と線形応答

温度勾配 \(\nabla T\) が小さい線形応答領域では、分布関数を平衡分布からの 小さな偏差として展開できます： 

\[n(\mathbf{q}, j, \mathbf{r}) = n_0(\mathbf{q}, j) + \delta n(\mathbf{q}, j, \mathbf{r})\] 

ここで、\(n_0 = [\exp(\hbar\omega/k_B T) - 1]^{-1}\) はBose-Einstein分布です。 定常状態（\(\partial n/\partial t = 0\)）かつ外力がない場合、方程式は： 

\[\mathbf{v}_g \cdot \nabla_\mathbf{r} n_0 = -\left(\frac{\partial \delta n}{\partial t}\right)_\text{scatt}\] 

### 3.1.3 フーリエの法則との関係

熱流束 \(\mathbf{J}_Q\) は分布関数を用いて以下のように表されます： 

\[\mathbf{J}_Q = \frac{1}{V}\sum_{\mathbf{q}, j} \hbar\omega(\mathbf{q}, j) \mathbf{v}_g(\mathbf{q}, j) \delta n(\mathbf{q}, j)\] 

線形応答理論により、\(\mathbf{J}_Q = -\kappa \nabla T\) の関係から熱伝導率テンソル \(\kappa\) が得られます。 

#### 物理的解釈

ボルツマン方程式は、フォノンの流れ（ドリフト項：\(\mathbf{v}_g \cdot \nabla_\mathbf{r} n\)）と 散乱による緩和（衝突項）のバランスを記述します。温度勾配があると、高温側から 低温側への正味のエネルギー流が生じますが、散乱過程がこれを平衡状態へ戻そうとします。 この競合が有限の熱抵抗（有限の熱伝導率）を生み出します。 

### 3.1.4 散乱項の構造

散乱項は、様々な散乱メカニズムの寄与の和として表されます： 

\[\left(\frac{\partial n}{\partial t}\right)_\text{scatt} = \left(\frac{\partial n}{\partial t}\right)_\text{ph-ph} + \left(\frac{\partial n}{\partial t}\right)_\text{boundary} + \left(\frac{\partial n}{\partial t}\right)_\text{defect} + \cdots\] 

主要な散乱メカニズム：

  * **フォノン-フォノン散乱** : 非調和効果による三フォノン・四フォノン過程
  * **境界散乱** : 試料の有限サイズ効果
  * **欠陥散乱** : 点欠陥、転位、粒界など
  * **電子-フォノン散乱** : 金属や高温での寄与


## 3.2 緩和時間近似と反復解法

### 3.2.1 緩和時間近似（RTA）

最も単純な近似は、散乱項を緩和時間 \(\tau(\mathbf{q}, j)\) で特徴づける方法です： 

\[\left(\frac{\partial \delta n}{\partial t}\right)_\text{scatt} = -\frac{\delta n}{\tau(\mathbf{q}, j)}\] 

#### 緩和時間近似での熱伝導率

RTAでは、熱伝導率は以下の形に簡略化されます：

\[\kappa_{\alpha\beta} = \frac{1}{V k_B T^2}\sum_{\mathbf{q}, j} \hbar\omega(\mathbf{q}, j) n_0(n_0 + 1) v_{g,\alpha} v_{g,\beta} \tau(\mathbf{q}, j)\] 

等方的な系では、\(\kappa = \frac{1}{3}\sum_{\mathbf{q}, j} C(\mathbf{q}, j) v_g^2 \tau(\mathbf{q}, j)\) となり、これは単純な \(C v^2 \tau\) 公式の和です。 

#### RTAの利点と限界

**利点：**

  * 計算が簡単で直感的な解釈が可能
  * 各モードの寄与を明確に分離できる
  * 平均自由行程 \(\Lambda = v_g \tau\) が自然に定義される


**限界：**

  * 運動量保存則（Normal process）を無視
  * 高純度結晶や低温で10-30%の過小評価
  * 異方性の強い材料での精度低下


### 3.2.2 反復的ボルツマン輸送方程式（Iterative BTE）

より厳密なアプローチは、BTE を直接解く反復法です。分布関数の偏差を 変分パラメータ \(\mathbf{F}(\mathbf{q}, j)\) で表します： 

\[\delta n(\mathbf{q}, j) = -\tau(\mathbf{q}, j) n_0(n_0 + 1) \mathbf{F}(\mathbf{q}, j) \cdot \nabla T\] 

\(\mathbf{F}\) に対する自己無撞着方程式は： 

\[\mathbf{F}(\mathbf{q}, j) = \mathbf{v}_g(\mathbf{q}, j) + \tau(\mathbf{q}, j) \sum_{\mathbf{q}', j'} \Gamma(\mathbf{q}j, \mathbf{q}'j') \mathbf{F}(\mathbf{q}', j')\] 

ここで、\(\Gamma\) は散乱行列です。この方程式を反復的に解くことで、 運動量保存則を正確に取り入れた熱伝導率が得られます。 

### 3.2.3 Normal vs Umklapp 過程

graph LR A[三フォノン散乱] --> B[Normal過程] A --> C[Umklapp過程] B --> D[運動量保存  
q + q' = q''] C --> E[運動量保存+逆格子ベクトル  
q + q' = q'' + G] D --> F[熱流を妨げない  
RTAで無視] E --> G[熱抵抗の主因  
RTAで捕捉] style B fill:#90EE90 style C fill:#FFB6C1 

反復解法では Normal 過程を適切に扱うことで、特に高純度結晶や 低温領域で RTA よりも正確な熱伝導率を予測できます。 

## 3.3 第一原理からの三フォノン散乱率

### 3.3.1 三次の力定数

三フォノン散乱は、結晶ポテンシャルの三次項から生じます： 

\[\Phi^{(3)} = \frac{1}{3!}\sum_{n\kappa\alpha, n'\kappa'\alpha', n''\kappa''\alpha''} \Phi_{\alpha\alpha'\alpha''}(n\kappa, n'\kappa', n''\kappa'') u_{n\kappa\alpha} u_{n'\kappa'\alpha'} u_{n''\kappa''\alpha''}\] 

#### 三次の力定数

\[\Phi_{\alpha\alpha'\alpha''}(n\kappa, n'\kappa', n''\kappa'') = \frac{\partial^3 \Phi}{\partial u_{n\kappa\alpha} \partial u_{n'\kappa'\alpha'} \partial u_{n''\kappa''\alpha''}}\] 

これは第一原理計算（DFT）で直接計算するか、有限変位法で数値的に評価します。 

### 3.3.2 散乱振幅とFermiの黄金律

三次の力定数から、フォノン相互作用ハミルトニアンの行列要素が得られます： 

\[V_3(\mathbf{q}j, \mathbf{q}'j', \mathbf{q}''j'') = \sum_{\kappa\kappa'\kappa'', \alpha\alpha'\alpha''} \frac{\Phi_{\alpha\alpha'\alpha''} e_\alpha e'_{\alpha'} e''_{\alpha''}}{\sqrt{8M_\kappa M_{\kappa'} M_{\kappa''} \omega\omega'\omega''}}\] 

Fermi の黄金律により、散乱率は： 

\[\Gamma_{\mathbf{q}j}^{(3)} = \frac{\pi\hbar}{N}\sum_{\mathbf{q}'j', \mathbf{q}''j''} |V_3|^2 \left[(n' + n'' + 1)\delta(\omega - \omega' - \omega'') + 2(n' - n'')\delta(\omega + \omega' - \omega'')\right]\] 

### 3.3.3 選択則と保存則

#### 三フォノン過程の選択則

**吸収過程（Absorption）：**

\[\mathbf{q} + \mathbf{q}' = \mathbf{q}'' + \mathbf{G}, \quad \omega + \omega' = \omega''\] 

**放出過程（Emission）：**

\[\mathbf{q} = \mathbf{q}' + \mathbf{q}'' + \mathbf{G}, \quad \omega = \omega' + \omega''\] 

\(\mathbf{G}\) はゼロまたは逆格子ベクトル。\(\mathbf{G} = 0\) なら Normal 過程、 \(\mathbf{G} \neq 0\) なら Umklapp 過程。 

### 3.3.4 有限変位法による三次力定数計算

実用的には、対称性を考慮した有限変位パターンを用いて三次力定数を計算します： 

  1. スーパーセルを構築（典型的には 2×2×2 から 4×4×4）
  2. 原子ペアを変位させた配置を生成（対称性により独立な配置のみ）
  3. 各配置で力を DFT 計算
  4. 力の差分から三次力定数を数値的に求める


#### Phono3py での三次力定数計算の流れ
    
    
    # 1. 変位パターン生成
    phono3py --dim="2 2 2" -c POSCAR-unitcell
    
    # 2. 各変位配置で DFT 計算（VASP例）
    # POSCAR-{00001..00XXX} に対して自己無撞着計算
    
    # 3. 力を収集して力定数を計算
    phono3py --cf3 vasprun.xml-{00001..00XXX}
    
    # 4. 散乱率と熱伝導率を計算
    phono3py --br --mesh="20 20 20" --ts="300"

## 3.4 四フォノン散乱とその重要性

### 3.4.1 なぜ四フォノン過程が重要か

従来、三フォノン過程のみが考慮されてきましたが、近年の研究で 以下の系で四フォノン散乱が支配的になることが明らかになりました： 

  * **高温領域** : フォノン占有数が大きくなると四フォノン過程の確率が増大
  * **軽元素材料** : Si, C, BN など高周波数フォノンを持つ系
  * **弱い三フォノン散乱** : 高対称性により三フォノン過程が抑制される系
  * **二次元材料** : グラフェン、h-BN など


#### 例: シリコンの熱伝導率

室温以上の Si では、四フォノン過程を無視すると熱伝導率を 約 20-50% 過大評価します。1000 K では四フォノン過程が 全散乱の 50% 以上を占めます。 

graph TD A[Si @ 300K] --> B[三フォノンのみ: κ ≈ 180 W/mK] A --> C[三+四フォノン: κ ≈ 145 W/mK] C --> D[実験値: κ ≈ 140-150 W/mK] B --> E[20-25% 過大評価] style C fill:#90EE90 style D fill:#87CEEB 

### 3.4.2 四次の力定数と散乱率

四フォノン過程は四次の力定数 \(\Phi^{(4)}\) から生じます： 

\[\Gamma_{\mathbf{q}j}^{(4)} = \frac{\pi\hbar}{N}\sum_{\mathbf{q}'j', \mathbf{q}''j'', \mathbf{q}'''j'''} |V_4|^2 \times (\text{占有数因子}) \times (\text{エネルギー保存})\] 

四次力定数の計算は計算コストが高い（変位配置数が \(O(N^3)\)）ため、 機械学習ポテンシャルや対称性を最大限活用した効率的手法が開発されています。 

### 3.4.3 実装と計算ツール

四フォノン散乱を扱えるソフトウェア： 

  * **Phono3py (v2.0+)** : 四フォノン散乱に対応（FourPhonon オプション）
  * **ShengBTE** : 一部の四フォノン過程をサポート
  * **almaBTE** : 完全な四フォノン実装


#### Phono3py での四フォノン計算
    
    
    # 四次力定数の計算（大きなスーパーセルが必要）
    phono3py --dim="3 3 3" -c POSCAR --fc4
    
    # 四フォノン散乱を含む熱伝導率計算
    phono3py --br --mesh="24 24 24" --ts="100 200 300 500 1000" \
             --fc3 --fc4 --write-gamma

## 3.5 ShengBTE: 理論と実用

### 3.5.1 ShengBTE の概要

ShengBTE は、第一原理三次力定数から熱伝導率を計算するための 高性能Fortranコードです。Wu Li らによって開発され、広く使われています。 

#### ShengBTE の特徴

  * **完全な BTE 解法** : RTA と反復解法の両方をサポート
  * **適応的積分** : デルタ関数の四面体法による正確な評価
  * **並列化** : MPI/OpenMP による大規模並列計算
  * **柔軟な入力** : VASP, Quantum ESPRESSO, Phonopy と連携
  * **詳細な解析** : モード分解、平均自由行程分布など


### 3.5.2 ShengBTE のワークフロー

graph TB A[DFT計算で二次・三次力定数] --> B[thirdorder.py で  
三次力定数抽出] B --> C[CONTROL ファイル作成] C --> D[ShengBTE 実行] D --> E[BTE.kappa: 熱伝導率] D --> F[BTE.w_final: 散乱率] D --> G[BTE.cumulative_kappa_*:  
累積分布] D --> H[BTE.ReciprocalLatticeVectors:  
逆格子情報] style E fill:#90EE90 style F fill:#FFD700 style G fill:#87CEEB 

### 3.5.3 CONTROL ファイルの設定

#### 典型的な CONTROL ファイル（Si の例）
    
    
    # ShengBTE CONTROL file for Silicon
    &allocations
      nelements=1      # 元素数
      natoms=2         # 単位格子内の原子数
      ngrid(:)=20 20 20  # q点メッシュ
      norientations=0  # 異方性を考慮しない場合
    &end
    
    &crystal
      lfactor=0.5431   # 格子定数 (nm)
      lattvec(:,1)=0.0 0.5 0.5  # 格子ベクトル
      lattvec(:,2)=0.5 0.0 0.5
      lattvec(:,3)=0.5 0.5 0.0
      types(:)=1 1     # 原子タイプ
      elements='Si'    # 元素名
      positions(:,1)=0.00 0.00 0.00  # 原子位置
      positions(:,2)=0.25 0.25 0.25
      scell(:)=5 5 5   # スーパーセル（力定数計算用）
      masses(:)=28.0855  # 原子質量
    &end
    
    ¶meters
      T=300.0          # 温度 (K)
      scalebroad=0.5   # ブロードニング係数
      isotopes=.TRUE.  # 同位体散乱を含む
      onlyharmonic=.FALSE.  # 非調和効果を含む
      convergence=1e-4 # 反復解法の収束条件
      maxiter=100      # 最大反復回数
    &end
    
    &flags
      espresso=.FALSE. # VASP を使用
      nonanalytic=.TRUE.  # LO-TO分裂を考慮
    &end

### 3.5.4 結果の解釈

ShengBTE の主要な出力ファイル： 

  * **BTE.kappa** : 温度の関数としての熱伝導率テンソル
  * **BTE.w_final** : 各モードの散乱率（逆緩和時間）
  * **BTE.kappa_tensor** : 完全な3×3熱伝導率テンソル
  * **BTE.cumulative_kappa_*** : 平均自由行程の関数としての累積熱伝導率


## 3.6 almaBTE と Phono3py

### 3.6.1 Phono3py の特徴

Phono3py は Phonopy の拡張として開発され、Python で記述された 使いやすいフォノン-フォノン相互作用計算ツールです。 

#### Phono3py vs ShengBTE 比較

項目 | Phono3py | ShengBTE  
---|---|---  
プログラミング言語 | Python + C | Fortran  
インストール | pip install（簡単） | コンパイル必要  
DFTコードサポート | VASP, QE, LAMMPS 等 | 主に VASP, QE  
四フォノン散乱 | 完全サポート（v2.0+） | 限定的  
実行速度 | 中程度 | 高速  
拡張性 | 高（Python スクリプト） | 低  
  
### 3.6.2 Phono3py の基本的な使い方

#### 完全なワークフロー例（VASP使用）
    
    
    # ステップ1: 変位ファイル生成
    phono3py --dim="2 2 2" -c POSCAR-unitcell
    
    # ステップ2: DFT計算（各 POSCAR-{00001..00XXX} で）
    # （VASP INCAR設定: PREC=Accurate, ENCUT=高め, tight convergence）
    
    # ステップ3: 力定数の計算
    phono3py --cf3 vasprun.xml-{00001..00XXX}
    
    # ステップ4: 熱伝導率計算
    phono3py --mesh="19 19 19" --br --ts="200 300 500 800"
    
    # ステップ5: RTA vs 反復解法の比較
    phono3py --mesh="19 19 19" --br --ts="300" --write-gamma
    phono3py --mesh="19 19 19" --br --ts="300" --write-gamma --full-pp
    
    # ステップ6: 累積熱伝導率（MFP分布）
    phono3py --mesh="19 19 19" --br --ts="300" --write-kappa
    
    # ステップ7: 四フォノン散乱を含む計算
    phono3py --dim="3 3 3" --fc4 -c POSCAR-unitcell  # 変位生成
    # DFT計算...
    phono3py --cf4 vasprun.xml-{00001..XXXXX}  # 四次力定数
    phono3py --mesh="19 19 19" --br --ts="300 500 1000" --fc3 --fc4

### 3.6.3 almaBTE: 高度な解析ツール

almaBTE は、より詳細な熱輸送解析のための C++ ライブラリとツール群です。 特徴： 

  * **サイズ効果** : ナノ構造や薄膜の熱伝導率計算
  * **コヒーレント効果** : 波動的輸送の取り扱い
  * **時間依存輸送** : 過渡的熱応答のシミュレーション
  * **可視化ツール** : 散乱率、MFP、寄与解析の高度な可視化


#### almaBTE での解析例
    
    
    import almabte
    
    # ShengBTEやPhono3pyの結果を読み込み
    solver = almabte.Solver.from_shengbte("./")
    
    # 詳細な平均自由行程分布を計算
    mfp_spectrum = solver.compute_mfp_spectrum(T=300)
    
    # サイズ効果の評価（Casimir限界）
    kappa_L = solver.compute_kappa_vs_length(T=300, L_range=[10e-9, 10e-6])
    
    # バリスティック-拡散的輸送の転移
    regime_map = solver.analyze_transport_regime(T=300)

## 3.7 収束問題と解決策

### 3.7.1 主要な収束パラメータ

第一原理熱伝導率計算では、以下のパラメータに対する収束確認が必須です： 

#### 収束確認チェックリスト

  1. **q点メッシュ密度** : 散乱率計算の精度 
     * 典型値: 15×15×15 から 30×30×30
     * 収束基準: \(\kappa\) の変化 < 5%
  2. **スーパーセルサイズ** : 力定数のカットオフ 
     * 典型値: 2×2×2 から 5×5×5
     * 判定法: 遠距離力定数の減衰を確認
  3. **ブロードニング係数** : デルタ関数の近似 
     * 適応的設定: \(\sigma \propto \omega\) を推奨
     * 過小: ノイズ増大、過大: 散乱率の過小評価
  4. **DFT計算精度** : エネルギーカットオフ、k点、収束条件 
     * 通常の構造最適化より厳しい設定が必要
     * 力の収束: < 0.001 eV/Å


### 3.7.2 q点メッシュ収束テスト

#### 自動収束テストスクリプト（Python）
    
    
    import numpy as np
    import subprocess
    
    # テストするメッシュサイズ
    mesh_sizes = [11, 15, 19, 23, 27, 31]
    kappa_values = []
    
    for mesh in mesh_sizes:
        # Phono3py実行
        cmd = f"phono3py --mesh='{mesh} {mesh} {mesh}' --br --ts=300"
        subprocess.run(cmd, shell=True, check=True)
    
        # 結果読み込み
        kappa = np.loadtxt("kappa-m{0:02d}{0:02d}{0:02d}.hdf5".format(mesh))
        kappa_avg = np.mean(kappa[0, 1:4])  # 300Kでの平均
        kappa_values.append(kappa_avg)
    
        print(f"Mesh {mesh}³: κ = {kappa_avg:.2f} W/mK")
    
    # 収束判定
    kappa_values = np.array(kappa_values)
    convergence = np.abs(kappa_values[-1] - kappa_values[-2]) / kappa_values[-1] * 100
    print(f"\n最終収束: {convergence:.2f}%")
    
    if convergence < 5.0:
        print(f"✓ 収束達成（推奨メッシュ: {mesh_sizes[-2]}³）")
    else:
        print(f"✗ さらに高密度メッシュが必要")

### 3.7.3 収束を妨げる一般的な問題

#### よくある収束問題とその解決策

問題 | 症状 | 解決策  
---|---|---  
虚数フォノンモード | 負の振動数、計算クラッシュ | 構造最適化の精度向上、対称性の確認  
音響和則の破れ | 音響枝が原点でゼロにならない | NAC (Non-Analytic Correction) の適用  
散乱率の振動 | q点依存性が不規則 | ブロードニング増加、メッシュ高密度化  
反復解法の非収束 | max iteration 到達 | 緩和係数調整、初期値変更、メッシュ見直し  
異常に高い/低い κ | 物理的に不合理な値 | 力定数の妥当性確認、DFT設定の見直し  
  
### 3.7.4 四面体法とスミアリング法

エネルギー保存デルタ関数 \(\delta(\omega \pm \omega' \mp \omega'')\) の 評価方法が精度に大きく影響します： 

  * **Gaussian スミアリング** : 実装簡単、収束遅い
  * **適応的ブロードニング** : \(\sigma(\mathbf{q}) = C \cdot \omega(\mathbf{q})\)
  * **四面体法** : 最も正確だが実装複雑（ShengBTE の強み）


## 3.8 平均自由行程分布解析

### 3.8.1 平均自由行程の定義

各フォノンモード \((\mathbf{q}, j)\) の平均自由行程（MFP）は： 

\[\Lambda(\mathbf{q}, j) = v_g(\mathbf{q}, j) \tau(\mathbf{q}, j)\] 

これは、フォノンが散乱されるまでに移動する平均距離を表します。 MFP 分布は材料のナノスケール熱輸送特性を理解する鍵となります。 

### 3.8.2 累積熱伝導率

MFP が \(\Lambda\) 以下のモードの寄与を積算した累積熱伝導率： 

\[\kappa_\text{cum}(\Lambda) = \sum_{\mathbf{q}, j} C(\mathbf{q}, j) v_g^2(\mathbf{q}, j) \tau(\mathbf{q}, j) \times \Theta(\Lambda - \Lambda(\mathbf{q}, j))\] 

ここで、\(\Theta\) はステップ関数です。\(\kappa_\text{cum}(\Lambda)\) を プロットすることで、どの長さスケールのフォノンが熱伝導に寄与しているかが分かります。 

#### MFP分布の解釈例：Si vs PbTe

**Si（高熱伝導率材料）:**

  * MFP の範囲: 10 nm - 10 μm
  * 50% の寄与: MFP ~ 1 μm のフォノンから
  * バリスティック輸送が重要（ナノデバイスで顕著）


**PbTe（低熱伝導率材料）:**

  * MFP の範囲: 1 nm - 100 nm
  * 50% の寄与: MFP ~ 10 nm のフォノンから
  * 拡散的輸送が支配的


### 3.8.3 サイズ効果の予測

試料サイズ \(L\) が MFP と同程度になると、境界散乱が支配的になり 熱伝導率が低下します（Casimir 限界）： 

\[\kappa(L) \approx \int_0^\infty \kappa'(\Lambda) \frac{1}{1 + \Lambda/L} d\Lambda\] 

これにより、ナノワイヤや薄膜の熱伝導率をバルク値から予測できます。 

#### MFP分布の可視化（Python）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
    
    # Phono3pyのHDF5出力を読み込み
    with h5py.File("kappa-m191919.hdf5", "r") as f:
        temperatures = f["temperature"][:]
        gamma = f["gamma"][:]  # 散乱率
        group_velocity = f["group_velocity"][:]
        heat_capacity = f["heat_capacity"][:]
        qpoints = f["qpoint"][:]
        weights = f["weight"][:]
    
    T_idx = 1  # 300 K を選択
    v_norm = np.linalg.norm(group_velocity, axis=2)
    mfp = v_norm / (gamma[:, :, T_idx] + 1e-12)  # MFP = v / Gamma
    
    # 各モードの熱伝導率寄与
    kappa_mode = heat_capacity[:, :, T_idx] * v_norm**2 / gamma[:, :, T_idx, None]
    kappa_mode = np.nanmean(kappa_mode, axis=2) * weights[:, None]
    
    # MFPをソートして累積分布計算
    mfp_flat = mfp.flatten()
    kappa_flat = kappa_mode.flatten()
    sort_idx = np.argsort(mfp_flat)
    mfp_sorted = mfp_flat[sort_idx]
    kappa_cumulative = np.cumsum(kappa_flat[sort_idx])
    kappa_cumulative /= kappa_cumulative[-1]  # 正規化
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 累積分布
    ax1.semilogx(mfp_sorted * 1e9, kappa_cumulative * 100)
    ax1.set_xlabel("Mean Free Path (nm)")
    ax1.set_ylabel("Cumulative κ (%)")
    ax1.set_title("Cumulative Thermal Conductivity")
    ax1.grid(True, alpha=0.3)
    
    # ヒストグラム（微分分布）
    bins = np.logspace(-1, 4, 50)  # 0.1 nm to 10 μm
    hist, bin_edges = np.histogram(mfp_sorted * 1e9, bins=bins, weights=kappa_flat[sort_idx])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.semilogx(bin_centers, hist / np.sum(hist))
    ax2.set_xlabel("Mean Free Path (nm)")
    ax2.set_ylabel("κ Contribution (normalized)")
    ax2.set_title("MFP Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("mfp_distribution.png", dpi=300)
    print(f"50% of κ from MFP < {mfp_sorted[np.argmin(np.abs(kappa_cumulative - 0.5))] * 1e9:.1f} nm")

## 3.9 コヒーレント vs インコヒーレント輸送

### 3.9.1 輸送レジームの分類

フォノン輸送は、長さスケールと温度に応じて異なるレジームを示します： 

graph LR A[フォノン輸送] --> B[バリスティック] A --> C[拡散的] A --> D[コヒーレント] B --> E[L << MFP  
散乱なし] C --> F[L >> MFP  
古典的拡散] D --> G[波動干渉  
周期構造] style E fill:#90EE90 style F fill:#87CEEB style G fill:#FFD700 

### 3.9.2 コヒーレント輸送の条件

フォノンが波としての性質を保つ（位相コヒーレンス）条件： 

  * **低温** : フォノン-フォノン散乱が弱い（\(k_B T \ll \hbar\omega\)）
  * **短距離** : コヒーレンス長 \(l_\phi\) 内での伝播
  * **周期構造** : 超格子、フォノニック結晶など
  * **クリーンな界面** : 無秩序散乱が少ない


### 3.9.3 コヒーレント効果の例

#### 超格子における熱伝導率の低減

A/B 二層超格子では、界面でのフォノン反射・干渉により 以下の効果が生じます： 

  * **フォノンバンドフォールディング** : ミニバンド形成により群速度低下
  * **ゾーンフォールディング** : 有効BZが縮小し、Umklapp散乱増加
  * **界面散乱** : 音響ミスマッチによる反射
  * **コヒーレントフィルタリング** : 特定周波数のフォノンをブロック


結果: バルク平均より 50-90% 低い熱伝導率（熱電材料に有利） 

### 3.9.4 AGF法: コヒーレント輸送の計算

原子グリーン関数（Atomistic Green's Function, AGF）法は、 波動輸送を直接扱う手法です： 

\[G(\omega) = \left[(\omega + i\eta)^2 M - K - \Sigma_L - \Sigma_R\right]^{-1}\] 

ここで、\(K\) は力定数行列、\(\Sigma_{L/R}\) は左右のリードからの自己エネルギーです。 熱伝導率は透過係数 \(T(\omega)\) から： 

\[\kappa = \frac{1}{2\pi} \int_0^\infty d\omega \, \hbar\omega \frac{\partial n_0}{\partial T} T(\omega)\] 

#### BTE vs AGF の使い分け

  * **BTE（インコヒーレント）** : バルク、高温、長距離輸送
  * **AGF（コヒーレント）** : ナノスケール、低温、界面支配的な系
  * **ハイブリッド法** : 両方の効果を組み合わせる（開発中）


## 3.10 熱伝導率予測のための機械学習

### 3.10.1 なぜ機械学習か

第一原理BTE計算は高精度ですが、計算コストが高く（数日〜数週間/材料）、 大規模材料スクリーニングには不向きです。機械学習（ML）により： 

  * 高速予測（秒〜分/材料）
  * 記述子ベースの物理的解釈
  * 未知材料への外挿
  * 逆設計（目標物性から構造を提案）


### 3.10.2 機械学習アプローチの分類

graph TB A[ML for κ prediction] --> B[記述子ベース回帰] A --> C[グラフニューラルネット] A --> D[機械学習力場  
+BTE] B --> E[組成・構造記述子  
→ κ] C --> F[結晶グラフ  
→ κ] D --> G[MLIP → FC  
→ BTE → κ] style E fill:#90EE90 style F fill:#87CEEB style G fill:#FFD700 

### 3.10.3 記述子ベース機械学習

材料の構造・組成情報から特徴量を抽出し、熱伝導率を予測： 

#### 記述子ベース予測の例（Python + scikit-learn）
    
    
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.structure import SiteStatsFingerprint
    
    # データセット読み込み（例：Materials Projectから収集）
    # compositions: 組成のリスト
    # structures: pymatgen Structure objects
    # kappa_values: 第一原理計算された熱伝導率
    
    # 特徴量生成
    featurizer = ElementProperty.from_preset("magpie")
    X_comp = featurizer.featurize_many(compositions, ignore_errors=True)
    
    struct_featurizer = SiteStatsFingerprint.from_preset("CoordinationNumber")
    X_struct = [struct_featurizer.featurize(s) for s in structures]
    
    # 特徴量結合
    X = np.hstack([X_comp, X_struct])
    y = np.array(kappa_values)
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # モデル訓練
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    
    # 評価
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    
    # 特徴量重要度
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    print("Top 10 important features:")
    for idx in reversed(top_features):
        print(f"  {featurizer.feature_labels()[idx]}: {feature_importance[idx]:.3f}")

### 3.10.4 グラフニューラルネットワーク（GNN）

結晶構造を原子ノードと結合エッジからなるグラフとして表現し、 GNN で直接物性を予測する手法が注目されています。 

#### 代表的なGNNモデル

  * **CGCNN (Crystal Graph CNN)** : 結晶構造から物性予測の先駆的手法
  * **MEGNet** : エネルギー、バンドギャップ、熱伝導率など多物性予測
  * **SchNet, DimeNet** : 原子間距離と角度を考慮した高精度モデル
  * **M3GNet** : Materials Project の 100万構造で訓練された汎用モデル


### 3.10.5 機械学習力場 + BTE ハイブリッド

最先端のアプローチは、機械学習力場（MLIP）で力定数を高速計算し、 その後BTE解法で熱伝導率を求める二段階法です： 

  1. 少数の第一原理計算で MLIP を訓練（GAP, SNAP, NequIP など）
  2. MLIP を用いて大きなスーパーセルでの力定数計算（DFT の 100-1000 倍高速）
  3. 得られた力定数で通常の BTE 計算


#### ハイブリッド法の利点

  * 精度: 純粋 ML より高精度（物理モデルを保持）
  * 速度: 純粋 DFT より 10-100 倍高速
  * 解釈性: BTE の枠組みで物理的解釈可能
  * 拡張性: 複雑な系（欠陥、界面）への適用が容易


### 3.10.6 ハイスループットスクリーニング

ML モデルを用いて、材料データベース（Materials Project, AFLOWなど）の 数十万材料の熱伝導率を一括予測し、有望候補を絞り込む戦略： 

#### 高効率熱電材料スクリーニングの例
    
    
    from pymatgen.ext.matproj import MPRester
    import pandas as pd
    
    # Materials Project から半導体を取得
    mpr = MPRester("YOUR_API_KEY")
    entries = mpr.query(
        criteria={"band_gap": {"\(gt": 0.5, "\)lt": 2.0}},  # 半導体
        properties=["material_id", "pretty_formula", "structure", "band_gap"]
    )
    
    # MLモデルで熱伝導率予測（上記で訓練済み）
    structures = [e["structure"] for e in entries]
    X_pred = generate_features(structures)  # 特徴量生成関数
    kappa_pred = model.predict(X_pred)
    
    # 結果を DataFrame に
    df = pd.DataFrame({
        "material_id": [e["material_id"] for e in entries],
        "formula": [e["pretty_formula"] for e in entries],
        "band_gap": [e["band_gap"] for e in entries],
        "kappa_predicted": kappa_pred
    })
    
    # 低熱伝導率材料を抽出（熱電材料候補）
    candidates = df[df["kappa_predicted"] < 2.0].sort_values("kappa_predicted")
    print(f"Found {len(candidates)} low-κ candidates")
    print(candidates.head(20))
    
    # 上位候補を第一原理で検証
    for idx, row in candidates.head(5).iterrows():
        print(f"\nValidating {row['formula']} (mp-{row['material_id']})")
        # ここで詳細なDFT+BTE計算を実行...

## 3.11 実践例：Si と PbTe の熱伝導率

### 3.11.1 ケーススタディ1: Silicon (Si)

シリコンは高熱伝導率材料の代表例で、熱伝導率計算の精度検証に 広く用いられています。 

#### Si の計算詳細

**計算条件:**

  * 結晶構造: ダイヤモンド構造、\(a = 5.431\) Å
  * スーパーセル: 4×4×4 (128 atoms)
  * DFT: VASP, PBE, ENCUT = 500 eV
  * k点: 8×8×8 (力計算)
  * q点メッシュ: 25×25×25


**結果（300 K）:**

手法 | κ (W/mK) | 計算時間  
---|---|---  
RTA (三フォノン) | ~180 | ~6 時間  
反復解法 (三フォノン) | ~210 | ~12 時間  
三+四フォノン | ~145 | ~48 時間  
実験値 | 140-150 | -  
  
### 3.11.2 Si の解析スクリプト

#### 完全な解析ワークフロー
    
    
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    
    # Phono3py 結果の読み込み
    def load_phono3py_data(filename):
        with h5py.File(filename, "r") as f:
            data = {
                "temperature": f["temperature"][:],
                "kappa": f["kappa"][:],  # shape: (T, 6)
                "gamma": f["gamma"][:],
                "group_velocity": f["group_velocity"][:],
                "frequencies": f["frequency"][:],
                "qpoints": f["qpoint"][:],
            }
        return data
    
    # データ読み込み
    data_rta = load_phono3py_data("kappa-m252525.hdf5")
    data_full = load_phono3py_data("kappa-m252525-full.hdf5")
    
    # 温度依存性プロット
    T = data_rta["temperature"]
    kappa_rta = np.mean(data_rta["kappa"][:, 1:4], axis=1)  # xx, yy, zz 平均
    kappa_full = np.mean(data_full["kappa"][:, 1:4], axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(T, kappa_rta, 'o-', label="RTA", linewidth=2)
    plt.loglog(T, kappa_full, 's-', label="Iterative BTE", linewidth=2)
    
    # 実験データ（文献値）
    T_exp = np.array([100, 200, 300, 400, 500, 800, 1000])
    kappa_exp = np.array([900, 260, 145, 95, 70, 40, 30])
    plt.loglog(T_exp, kappa_exp, 'k^', label="Experiment", markersize=8)
    
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel("Thermal Conductivity (W/mK)", fontsize=14)
    plt.title("Silicon Thermal Conductivity", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("Si_kappa_vs_T.png", dpi=300)
    
    # モード分解分析（300 K）
    T_idx = np.argmin(np.abs(T - 300))
    freq = data_rta["frequencies"]
    gamma = data_rta["gamma"][:, :, T_idx]
    v_g = np.linalg.norm(data_rta["group_velocity"], axis=2)
    
    # 音響 vs 光学モード（6 branches: 3 acoustic + 3 optical）
    freq_threshold = 8.0  # THz（LA/TA と LO/TO の境界）
    acoustic_mask = freq < freq_threshold
    
    gamma_ac = gamma[acoustic_mask]
    gamma_op = gamma[~acoustic_mask]
    
    print(f"300 K での散乱率統計:")
    print(f"  音響モード: 平均 Γ = {np.mean(gamma_ac):.2e} THz")
    print(f"  光学モード: 平均 Γ = {np.mean(gamma_op):.2e} THz")
    print(f"  音響/光学 比: {np.mean(gamma_op) / np.mean(gamma_ac):.1f}")

### 3.11.3 ケーススタディ2: PbTe（熱電材料）

PbTe は代表的な熱電材料で、低熱伝導率と高い電気伝導率を併せ持ちます。 

#### PbTe の特徴

**構造:**

  * 岩塩構造（NaCl型）、\(a = 6.46\) Å
  * 重元素（Pb, Te）→ 低フォノン周波数
  * 強いフォノン-フォノン散乱


**熱伝導率（300 K）:**

  * 第一原理BTE: κ ≈ 2.0-2.5 W/mK
  * 実験値: κ ≈ 2.2 W/mK
  * Si と比べて約 70倍 低い


**低熱伝導率の理由:**

  1. 重い原子 → 低い群速度（\(v_g \propto 1/\sqrt{M}\)）
  2. ソフトなフォノン → 強い非調和性
  3. 複雑なバンド構造 → 多くの散乱チャネル
  4. 共鳴散乱（Pb 原子の局所振動）


### 3.11.4 PbTe の計算実例

#### PbTe 計算セットアップ（VASP + Phono3py）
    
    
    #!/bin/bash
    # PbTe 熱伝導率計算の完全なワークフロー
    
    # ステップ1: 構造最適化
    cat > INCAR_relax << EOF
    ISTART = 0
    PREC = Accurate
    ENCUT = 400
    EDIFF = 1E-8
    EDIFFG = -1E-6
    NSW = 200
    IBRION = 2
    ISIF = 3
    ISMEAR = 0
    SIGMA = 0.05
    LREAL = .FALSE.
    EOF
    
    # VASP 実行...
    
    # ステップ2: Phono3py 変位生成
    phono3py --dim="3 3 3" -c POSCAR-unitcell --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0"
    
    # ステップ3: 力計算用 INCAR（タイトな収束条件）
    cat > INCAR_forces << EOF
    ISTART = 0
    PREC = Accurate
    ENCUT = 450
    EDIFF = 1E-9
    ISMEAR = 0
    SIGMA = 0.02
    LREAL = .FALSE.
    ADDGRID = .TRUE.
    EOF
    
    # 各変位配置で力計算（並列化推奨）
    for i in {00001..00XXX}; do
        mkdir disp-$i
        cp POSCAR-\(i disp-\)i/POSCAR
        cd disp-$i
        # VASP 実行
        cd ..
    done
    
    # ステップ4: 力定数計算
    phono3py --cf3 disp-*/vasprun.xml
    
    # ステップ5: 熱伝導率計算（RTA）
    phono3py --mesh="21 21 21" --br --ts="300 400 500 600 700 800"
    
    # ステップ6: 反復解法
    phono3py --mesh="21 21 21" --br --ts="300 500 800" --full-pp --write-gamma
    
    # ステップ7: MFP分布解析
    phono3py --mesh="21 21 21" --br --ts="300" --write-kappa --gv-delta-q=1e-4

### 3.11.5 Si vs PbTe の比較

#### 熱輸送特性の対比

物性 | Si (300 K) | PbTe (300 K)  
---|---|---  
熱伝導率 κ | ~145 W/mK | ~2.2 W/mK  
平均群速度 | ~5 km/s | ~1.5 km/s  
平均散乱率 | ~1 THz | ~10 THz  
平均MFP | ~1 μm | ~10 nm  
Debye温度 | ~645 K | ~136 K  
RTA vs 反復 | 15-20% 差 | <5% 差  
四フォノン寄与 | ~30% (300K) | <10%  
  
## まとめ

本章では、フォノンボルツマン輸送方程式に基づく熱伝導率の第一原理計算手法を 包括的に学びました。以下が主要なポイントです： 

  * **PBTEの定式化** : フォノン分布関数の発展方程式として熱輸送を記述
  * **緩和時間近似** : 簡便だが Normal 過程を無視（10-30% 過小評価）
  * **反復解法** : より正確だが計算コスト増（高純度材料で重要）
  * **三フォノン散乱** : 非調和性の主要効果、第一原理で計算可能
  * **四フォノン散乱** : 高温・軽元素材料で重要（Si, graphene など）
  * **ShengBTE** : 高速・正確な Fortran 実装、四面体法が強み
  * **Phono3py** : Python ベースで使いやすい、四フォノン完全対応
  * **収束問題** : q点メッシュ、スーパーセル、ブロードニング係数の注意深い選択
  * **MFP分布** : ナノスケール熱輸送とサイズ効果の理解に重要
  * **コヒーレント輸送** : 低温・ナノ構造で波動効果が顕在化
  * **機械学習** : 高速予測とハイスループットスクリーニングを実現


これらの手法を駆使することで、熱電材料、熱管理材料、電子デバイス用材料の 熱輸送特性を予測・設計できます。次章では、フォノンバンド構造における トポロジカル概念を探求します。 

## 演習問題

### 演習 3.1: BTE の物理的理解

**問題:** フォノンボルツマン輸送方程式の各項 \(\partial n/\partial t\), \(\mathbf{v}_g \cdot \nabla_\mathbf{r} n\), \((\partial n/\partial t)_\text{scatt}\) の物理的意味を説明し、定常状態 (\(\partial n/\partial t = 0\)) で なぜドリフト項と散乱項がバランスするのか論じなさい。 

### 演習 3.2: RTA 熱伝導率の計算

**問題:** 1次元単原子鎖（質量 \(M\)、格子定数 \(a\)、力定数 \(K\)）を考える。 緩和時間 \(\tau = \tau_0\) が一定と仮定して、RTA での熱伝導率を導出せよ。 Debye 温度より十分高温（\(T \gg \Theta_D\)）での漸近形を求めよ。 

**ヒント:** \(\omega(q) = 2\sqrt{K/M}|\sin(qa/2)|\)、\(v_g = d\omega/dq\)、 高温で \(n_0 \approx k_B T/\hbar\omega\) を使用。

### 演習 3.3: Umklapp vs Normal 過程

**問題:** 三フォノン過程 \(\mathbf{q}_1 + \mathbf{q}_2 = \mathbf{q}_3 + \mathbf{G}\) において、 Normal (\(\mathbf{G} = 0\)) と Umklapp (\(\mathbf{G} \neq 0\)) 過程の 熱抵抗への寄与が異なる理由を、運動量保存則の観点から説明せよ。 また、温度依存性がどう異なるか議論せよ。 

### 演習 3.4: Phono3py 実習

**問題:** 以下の手順で Si の熱伝導率を計算せよ： 

  1. Phono3py で 2×2×2 スーパーセルの変位パターンを生成
  2. 各変位配置で DFT 計算（VASP または Quantum ESPRESSO）
  3. 三次力定数を求め、15×15×15 メッシュで熱伝導率を計算
  4. 300 K での値を実験値（~145 W/mK）と比較
  5. 20×20×20 メッシュで再計算し、収束を確認


### 演習 3.5: 四フォノン効果の評価

**問題:** Si の 500 K と 1000 K での熱伝導率を、 (a) 三フォノンのみ、(b) 三+四フォノン で計算し、 四フォノン散乱の相対的寄与を評価せよ。温度とともに 四フォノン効果が増大する理由を、占有数因子の観点から説明せよ。 

### 演習 3.6: MFP 分布解析

**問題:** Phono3py の出力から Si の累積熱伝導率 \(\kappa_\text{cum}(\Lambda)\) をプロットし、以下を求めよ： 

  1. 50% の熱伝導に寄与する平均 MFP
  2. 100 nm サイズの Si ナノワイヤでの熱伝導率を Casimir 限界で予測
  3. PbTe でも同様の解析を行い、Si との違いを議論


### 演習 3.7: 機械学習による予測

**問題:** Materials Project から 200 個の半導体化合物の 組成・構造データと熱伝導率（文献値または計算値）を収集し、 機械学習モデル（RandomForest または GNN）で予測モデルを構築せよ。 テストセットでの R² スコアを報告し、重要な特徴量を特定せよ。 

### 演習 3.8: 収束テスト

**問題:** PbTe において、以下のパラメータの収束テストを実施せよ： 

  1. q点メッシュ: 11³, 15³, 19³, 23³ で熱伝導率を計算
  2. スーパーセル: 2³, 3³, 4³ で三次力定数を計算し、差を評価
  3. ブロードニング係数: 0.1, 0.5, 1.0 での影響を確認
  4. 推奨パラメータセットを提案せよ


### 演習 3.9: 同位体効果

**問題:** Si の同位体散乱率は \(\Gamma_\text{iso} \propto g \omega^4\)（\(g\) は質量分散パラメータ）で与えられる。 自然存在比の Si と同位体純化 \(^{28}\)Si での熱伝導率を計算し、 室温と 100 K での改善率を評価せよ。 

**データ:** 自然 Si: \(g = 2.0 \times 10^{-4}\)、\(^{28}\)Si: \(g \approx 0\)

### 演習 3.10: 研究課題

**問題:** 以下のいずれかのトピックについて、文献調査と 簡単な計算を実施し、レポート（2-3ページ）を作成せよ： 

  1. 二次元材料（グラフェン、MoS₂）での四フォノン散乱の重要性
  2. 超格子のコヒーレント熱伝導とバンドフォールディング効果
  3. 熱電材料の figure of merit (ZT) 最適化における κ の役割
  4. 機械学習力場（MLIP）を用いた熱伝導率計算の精度と効率

---

**ナビゲーション**

[← 第2章](chapter-2.md) | [目次](index.md) | [第4章 →](chapter-4.md)

---

## 免責事項

このコンテンツはAIの支援を受けて作成されており、正確性を保証するものではありません。重要な情報については一次資料や査読済み文献で確認することをお勧めします。
