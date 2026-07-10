---
title: 第4章：材料噴射法・結合剤噴射法・その他AM技術
chapter_title: 第4章：材料噴射法・結合剤噴射法・その他AM技術
subtitle: 液滴堆積・粉末結合・大型堆積・新興技術と、プロセス選定の考え方
code_examples: 3
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/3d-printing-introduction/chapter-4.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 4

## 学習目標

この章を完了すると、以下を説明できるようになります：

### 基本理解（Level 1）

  * 材料噴射法（Material Jetting, MJ／PolyJet）の液滴堆積原理と、マルチマテリアル・フルカラー造形の仕組み
  * 結合剤噴射法（Binder Jetting, BJ）のグリーンパート（未焼結の成形体）から焼結・含浸に至る後工程
  * 指向性エネルギー堆積（Directed Energy Deposition, DED／LENS）とシート積層法（Sheet Lamination, LOM／UAM）の位置づけ
  * ハイブリッド製造（付加＋除去加工の統合）と新興技術（バイオプリンティング、4Dプリンティング）の概要

### 実践スキル（Level 2）

  * インクジェット液滴の無次元数（Ohnesorge数・Weber数・Z数）を計算し、印刷可能性を判定できる
  * 相対密度から焼結収縮率を推定し、グリーンパートの寸法補正倍率を求められる
  * 重み付きスコアリングでAMプロセスを定量的に比較・選定できる

### 応用力（Level 3）

  * 精度・表面品質・速度・材料範囲・コスト・強度のトレードオフから用途に最適なプロセスを選べる
  * 各方式の後処理コストと歩留まりリスクを踏まえ、現実的な製造計画を立てられる
  * 新興技術の適用可能性と技術的成熟度を、誇張せずに評価できる

**💡 この章の位置づけ**

第1章では材料押出（MEX）、第2章・第3章では液槽光重合（VPP）と粉末床溶融結合（PBF）を扱いました。本章は、残る主要プロセスである**材料噴射法・結合剤噴射法・指向性エネルギー堆積・シート積層法** を横断的に整理し、最後に「どの方式をいつ選ぶか」という選定の考え方まで踏み込みます。個々の装置操作よりも、**プロセス間の比較軸** を身につけることを目標とします。

## 4.1 材料噴射法（Material Jetting, MJ）

### 4.1.1 原理：液滴を「印刷」して積層する

材料噴射法（Material Jetting, MJ）は、**インクジェットプリンタと同じ原理で液状材料を微小な液滴として噴射し、その場で紫外線（UV）硬化させて積層する** 方式です。Stratasys社の商標名から**PolyJet** とも呼ばれます。プリントヘッドには数百〜数千のノズルが並び、各層で必要な位置に光硬化性樹脂（フォトポリマー）の液滴を打ち込み、直後のUVランプで即座に固化させます。

プロセス: 樹脂を液滴化 → ヘッドから噴射 → 着弾・レベリング → UV硬化 → 次層へ 

MJは面全体をノズルアレイで一度に扱えるため、点走査のSLAより高速でありながら、液滴サイズが小さいため高精細です。一方で、**使える材料が光硬化性樹脂に限られ** 、機械的性質（強度・耐熱性）は中程度にとどまる点が本質的な制約です。

### 4.1.2 液滴の物理：印刷可能性を決める無次元数

液滴が「きれいに1粒だけ」形成されるかどうかは、粘性・表面張力・慣性のバランスで決まります。これを表すのが以下の無次元数です。用語を初出で定義します。

  * **Ohnesorge数（オーネゾルゲ数, Oh）** : 粘性力と、慣性力・表面張力の比。Oh = μ / √(ρ σ D)（μ:粘度、ρ:密度、σ:表面張力、D:ノズル径）。
  * **Z数** : Ohの逆数（Z = 1/Oh）。インクジェット分野で印刷適性の指標として広く使われます。
  * **Weber数（ウェーバー数, We）** : 慣性力と表面張力の比。We = ρ v² D / σ（v:液滴速度）。液滴を射出するのに十分なエネルギーがあるかを表します。
  * **Reynolds数（レイノルズ数, Re）** : 慣性力と粘性力の比。Re = ρ v D / μ。

経験的に、安定した液滴形成の窓は**おおむね 1 < Z < 10**（文献により上限14程度まで）とされます。Zが小さすぎる（粘りすぎる）と液滴が切れず、大きすぎると尾を引いて**サテライト液滴（衛星液滴）** が生じます。加えて、射出には十分なWeが必要です。

**⚠️ 無次元数は「目安」であって保証ではありません**

1 < Z < 10 という窓は多くの実験から得られた経験則であり、樹脂の非ニュートン性（せん断依存の粘度変化）、波形駆動、ノズル形状によって実際の窓は前後します。設計初期のスクリーニングには有効ですが、最終判断は実機での射出観察（ドロップウォッチャー）に委ねるのが誠実な進め方です。

### 4.1.3 マルチマテリアルとフルカラー造形

MJの最大の強みは、**複数のノズル群に異なる材料を割り当て、1回の造形の中で材料や色を自在に切り替えられる** 点です。硬い樹脂と柔らかい樹脂を勾配的に混ぜてゴム状〜硬質までの中間硬度を作る「デジタルマテリアル」や、CMYK＋白＋透明の樹脂を組み合わせた1,000万色以上のフルカラー造形が可能です。医療の解剖モデルでは、骨（硬）と軟組織（柔）と血管（透明）を1体の中に作り分けられ、術前計画に活用されています。

### 4.1.4 サポート戦略

MJでは造形材とは別の**サポート材** （多くはゲル状またはワックス状）を同時に噴射し、オーバーハングや中空を支えます。除去方法は主に2通りです。

  * **水溶性・水崩壊性サポート** : 水流やアルカリ溶液で溶かす。複雑な内部流路も洗い流せるが、細部に残渣が残りやすい。
  * **機械的除去サポート** : 手作業やウォータージェットで剥がす。速いが、微細形状を傷めるリスクがある。

サポート材のコストと除去の手間は、MJの実効コストと歩留まりに直結します。造形自体は高精度でも、**後処理まで含めた総コスト** で評価することが重要です。

## 4.2 結合剤噴射法（Binder Jetting, BJ）

### 4.2.1 原理：粉末を「糊付け」して成形する

結合剤噴射法（Binder Jetting, BJ）は、**薄く敷いた粉末床に液状の結合剤（バインダー）をインクジェットで噴射し、粉末粒子どうしを糊付けして各層を形成する** 方式です。レーザーや熱源で溶かさないため、造形自体は室温で高速に進みます。造形直後の成形体を**グリーンパート（緑体、未焼結の脆い状態）** と呼びます。

プロセス: 粉末敷設 → バインダー噴射（各層） → 硬化 → グリーンパート取り出し → 脱脂 → 焼結／含浸 

### 4.2.2 グリーンパートと後工程

グリーンパートはそのままでは強度が不足するため、後工程で緻密化します。金属・セラミックスでは主に次の経路をとります。

  * **脱脂（debinding）** : 加熱や溶剤でバインダーを除去する。急激に行うと割れるため、慎重な昇温が必要。
  * **焼結（sintering）** : 融点未満の高温（金属で1200〜1400℃程度）で粒子どうしを拡散結合させ、緻密化する。この過程で**大きな収縮** が起きる。
  * **含浸（infiltration）** : 焼結の代わりに、多孔質のグリーン／ブラウンパートへ低融点金属（青銅など）を毛細管現象で吸い込ませ、空隙を埋める。収縮が小さく寸法安定性に優れるが、材質は複合的になる。

### 4.2.3 焼結収縮の推定と寸法補正

焼結では相対密度（理論密度に対する割合）が上がるぶん、体積が減り、部品が縮みます。等方収縮を仮定すると、線収縮は密度比の立方根で表せます。

線収縮率 = (1 − (ρ_green / ρ_sinter)^(1/3)) × 100 ［%］ 

ここで ρ_green はグリーン相対密度、ρ_sinter は焼結後相対密度です。目標寸法を得るには、この収縮を見越して**グリーンパートを大きめに設計** します。倍率は (ρ_sinter / ρ_green)^(1/3) です。金属BJで線収縮が15〜20%に達することも珍しくなく、これを補正しないと部品が使い物になりません。この計算は後半のコード例2で実行します。

**💡 BJが得意な領域**

  * **砂型鋳造用の砂型・中子** : 焼結不要で、複雑な冷却・湯道を一体成形。エンジンブロックなど大型鋳物の量産で実用化。
  * **金属量産部品** : Desktop Metal や HP Metal Jet などが、射出成形に近い単価を狙う。
  * **フルカラー石膏モデル** : 記念品・教育モデル向け。強度は低いが安価でカラフル。

## 4.3 指向性エネルギー堆積（Directed Energy Deposition, DED）

指向性エネルギー堆積（Directed Energy Deposition, DED）は、**金属の粉末またはワイヤーを供給しながら、レーザー・電子ビーム・アークで溶かして基板上に肉盛りしていく** 方式です。ノズルとエネルギー源が一体で動くため、多軸ロボットアームに載せれば造形範囲の制約が小さく、大型部品にも対応できます。LENS（Laser Engineered Net Shaping）はレーザー粉末方式の代表的商標です。

  * **高い堆積速度** : 1〜5 kg/h とPBFの10〜50倍。ただし精度は粗い（±0.5〜2 mm）。
  * **補修・肉盛り** : 摩耗したタービンブレードや金型の欠損部を、既存部品の上に直接肉盛りして再生できる。ここがDED最大の実用価値。
  * **傾斜機能材料** : 供給する粉末の比率を連続的に変え、位置によって組成を変える（例: 基部は靭性重視、表層は耐摩耗重視）。

DEDは「ゼロから精密な形を作る」より、**「大きく速く盛る」「壊れたものを直す」** 用途で真価を発揮します。仕上げには機械加工が前提となるため、次節のハイブリッド製造と密接に結びつきます。

## 4.4 シート積層法とハイブリッド製造

### 4.4.1 シート積層法（Sheet Lamination, SL）

シート積層法（Sheet Lamination, SL）は、**紙・金属箔・プラスチックフィルムなどのシート材を積み重ね、接着または溶接し、各層の輪郭を切断する** 方式です。代表技術は次の2つです。

  * **LOM（Laminated Object Manufacturing）** : 接着剤付きの紙やフィルムを積層し、レーザーやブレードで輪郭を切る。大型・低コストだが、内部は中実で、用途は視覚モデル中心。
  * **UAM（Ultrasonic Additive Manufacturing）** : 金属箔を超音波で固相接合し、CNC切削で形を整える。低温接合のため、**造形物の中にセンサーや光ファイバーを埋め込める** という他方式にない特徴を持つ。

### 4.4.2 ハイブリッド製造（付加＋除去の統合）

ハイブリッド製造は、**付加加工（AM）と除去加工（切削）を1台の機械の中で交互に行う** アプローチです。DEDで素形材を盛り、まだアクセスできるうちにフライス加工で表面を仕上げ、また盛る——という工程を繰り返すことで、AMの形状自由度とCNCの表面精度・寸法精度を両立します。既存部品への肉盛り補修＋仕上げを一貫して行える点も、産業での採用が進む理由です。

## 4.5 プロセス選定の考え方

ここまでの各方式に「万能なもの」はありません。選定は**用途要求に対するトレードオフの評価** です。主要な比較軸を整理します。

プロセス | 精度・表面 | 速度 | 主な材料 | 強度 | 得意な用途  
---|---|---|---|---|---  
材料噴射（MJ） | 非常に高い | 中 | 光硬化樹脂（多材料・フルカラー） | 低〜中 | 意匠モデル、医療解剖モデル  
結合剤噴射（BJ） | 中 | 高い | 金属・セラミックス・砂・石膏 | 中（焼結後） | 砂型、金属量産、フルカラー像  
DED／LENS | 低い（要後加工） | 非常に高い（肉盛り） | 金属（粉末・ワイヤー） | 高い | 補修、大型部品、傾斜材  
シート積層（LOM／UAM） | 中 | 高い | 紙・金属箔 | 低〜中 | 視覚モデル、センサー埋込  
  
**⚠️ 「精度が高い＝良い」ではありません**

選定でよくある誤りは、単一指標（例: 精度）だけで方式を決めてしまうことです。砂型を1個作るのにMJの超高精度は不要ですし、補修にBJは使えません。**要求を満たす最小十分な方式を、総コスト（材料＋後処理＋歩留まり）で選ぶ** のが実務の原則です。コード例3では、この考え方を重み付きスコアリングとして定量化します。

## 4.6 新興トレンド

### 4.6.1 バイオプリンティング（概要）

バイオプリンティング（Bioprinting）は、**生細胞を含む「バイオインク」（細胞＋ハイドロゲル担体）を吐出し、組織や臓器モデルを構築する** 研究分野です。細胞の生存率を保つため、噴射時のせん断応力を抑える低圧・低粘度条件が求められ、本章で扱った液滴物理の知見がそのまま関わります。現状は**創薬スクリーニング用の組織チップや、皮膚・軟骨の小片** が中心で、移植可能な臓器はまだ研究段階です。過度な期待を避け、成熟度を正直に見極める姿勢が大切です。

### 4.6.2 4Dプリンティング（概要）

4Dプリンティング（4D Printing）は、**造形後に温度・湿度・光などの刺激で形状が変化する** ように設計する技術です。「4次元目」は時間、すなわち**時間とともに形が変わること** を指します。形状記憶ポリマーや、吸湿で膨潤する材料を方向性をもって配置し、平面から立体へ自己折り畳みするような構造を作ります。展開アンテナ、自己組立部品、ソフトロボットなどが応用候補ですが、こちらも実用化はごく限られた事例にとどまります。

## コード例

本章の要点を、実行可能なPythonコードで確認します。以下の出力はすべて実際に `python3` で実行した結果です（NumPyを使用）。

### コード例1: 液滴の印刷可能性マップ（Oh／Z／We／Re）

代表的な噴射流体について無次元数を計算し、印刷可能な窓（1 < Z < 10 かつ We > 4）に入るかを判定します。
    
    
    import numpy as np
    
    # Material Jetting droplet printability: Ohnesorge / Reynolds / Weber / Z number
    # Z = 1/Oh ; printable window commonly cited as 1 < Z < 10 (some report up to 14)
    # Oh = mu / sqrt(rho * sigma * D)
    
    def dimensionless(rho, mu, sigma, D, v):
        Oh = mu / np.sqrt(rho * sigma * D)
        Z = 1.0 / Oh
        Re = rho * v * D / mu
        We = rho * v**2 * D / sigma
        return Oh, Z, Re, We
    
    # Representative jetting fluids (SI units)
    # rho [kg/m3], mu [Pa.s], sigma [N/m], D nozzle [m], v drop [m/s]
    fluids = [
        ("UV acrylate resin (PolyJet)", 1100, 0.012, 0.030, 30e-6, 8.0),
        ("Molten wax (support)",         900, 0.020, 0.025, 30e-6, 6.0),
        ("Water-thin binder (BJ)",      1000, 0.001, 0.072, 40e-6, 9.0),
        ("Nanoparticle metal ink",      1500, 0.015, 0.035, 20e-6, 7.0),
        ("Over-viscous resin (fail)",   1150, 0.080, 0.030, 30e-6, 8.0),
    ]
    
    print(f"{'Fluid':32s}{'Oh':>8s}{'Z=1/Oh':>9s}{'Re':>8s}{'We':>8s}  Printable(1<Z<10)")
    print("-"*80)
    for name, rho, mu, sigma, D, v in fluids:
        Oh, Z, Re, We = dimensionless(rho, mu, sigma, D, v)
        ok = "YES" if (1.0 < Z < 10.0 and We > 4.0) else "NO"
        print(f"{name:32s}{Oh:8.3f}{Z:9.2f}{Re:8.1f}{We:8.1f}   {ok}")
    
    print()
    print("Interpretation:")
    print(" - Z < 1  : too viscous, droplet won't form cleanly")
    print(" - Z > 10 : satellite droplets / instability")
    print(" - We < 4 : insufficient energy to eject a droplet")

**実行結果：**
    
    
    Fluid                                 Oh   Z=1/Oh      Re      We  Printable(1<Z<10)
    --------------------------------------------------------------------------------
    UV acrylate resin (PolyJet)        0.381     2.62    22.0    70.4   YES
    Molten wax (support)               0.770     1.30     8.1    38.9   YES
    Water-thin binder (BJ)             0.019    53.67   360.0    45.0   NO
    Nanoparticle metal ink             0.463     2.16    14.0    42.0   YES
    Over-viscous resin (fail)          2.487     0.40     3.5    73.6   NO
    
    Interpretation:
     - Z < 1  : too viscous, droplet won't form cleanly
     - Z > 10 : satellite droplets / instability
     - We < 4 : insufficient energy to eject a droplet

粘度の高すぎる樹脂（Z=0.40）は液滴が切れず、逆に水のように低粘度の結合剤（Z=53.7）はサテライト液滴を生じやすいことが数値で確認できます。PolyJet樹脂やナノ粒子インクは窓の中に収まっています。

### コード例2: 結合剤噴射の焼結収縮と寸法補正

相対密度から線収縮率・体積収縮率を求め、目標寸法を得るためのグリーン設計倍率を計算します。
    
    
    import numpy as np
    
    # Binder Jetting: green part -> sintered part shrinkage from densification.
    # Isotropic linear shrinkage from relative density change:
    #   L_sinter / L_green = (rho_green / rho_sinter)^(1/3)
    # Linear shrinkage (%) = (1 - (rho_g/rho_s)^(1/3)) * 100
    
    def linear_shrinkage(rho_green, rho_sinter):
        ratio = (rho_green / rho_sinter) ** (1.0/3.0)
        lin = (1.0 - ratio) * 100.0
        vol = (1.0 - rho_green / rho_sinter) * 100.0
        return lin, vol
    
    # rho values are RELATIVE density (fraction of theoretical)
    cases = [
        ("316L stainless (metal BJ)", 0.55, 0.98),
        ("Ti-6Al-4V (metal BJ)",      0.50, 0.96),
        ("Alumina ceramic",           0.45, 0.95),
        ("Bronze-infiltrated steel",  0.60, 0.90),
    ]
    
    print(f"{'System':30s}{'rho_green':>10s}{'rho_sint':>9s}{'Lin.shr%':>10s}{'Vol.shr%':>10s}")
    print("-"*70)
    for name, rg, rs in cases:
        lin, vol = linear_shrinkage(rg, rs)
        print(f"{name:30s}{rg:10.2f}{rs:9.2f}{lin:10.2f}{vol:10.2f}")
    
    # Compensation: to hit a 50.00 mm target after sintering, scale the green CAD.
    target = 50.00  # mm final dimension
    rg, rs = 0.55, 0.98
    scale = (rs / rg) ** (1.0/3.0)   # green must be LARGER by this factor
    green_dim = target * scale
    print()
    print(f"Design compensation (316L, rho_g=0.55 -> rho_s=0.98):")
    print(f"  required green scale factor = {scale:.4f}")
    print(f"  to obtain {target:.2f} mm final, model green part at {green_dim:.3f} mm")

**実行結果：**
    
    
    System                         rho_green rho_sint  Lin.shr%  Vol.shr%
    ----------------------------------------------------------------------
    316L stainless (metal BJ)           0.55     0.98     17.51     43.88
    Ti-6Al-4V (metal BJ)                0.50     0.96     19.54     47.92
    Alumina ceramic                     0.45     0.95     22.05     52.63
    Bronze-infiltrated steel            0.60     0.90     12.64     33.33
    
    Design compensation (316L, rho_g=0.55 -> rho_s=0.98):
      required green scale factor = 1.2123
      to obtain 50.00 mm final, model green part at 60.617 mm

316Lステンレスでは線収縮が約17.5%、体積収縮は約44%に達します。最終50.00 mmの寸法を得るには、グリーンパートを60.6 mmで設計する必要があり、収縮補正がいかに重要かがわかります。

### コード例3: AMプロセスの重み付きスコアリング

6つの評価軸（精度・表面・速度・材料範囲・コスト効率・強度）に重みを与え、用途ごとに最適な方式を定量比較します。重みを変えると推奨が変わる（感度）ことも示します。
    
    
    import numpy as np
    
    # AM process selection by weighted scoring.
    # Criteria scored 1-5 (5 = best for that criterion).
    criteria = ["accuracy", "surface", "speed", "material_range", "cost_eff", "strength"]
    weights  = np.array([0.25, 0.15, 0.15, 0.15, 0.15, 0.15])  # sums to 1.0
    
    # rows = processes, cols = criteria (expert-assigned 1-5)
    processes = {
        "Material Jetting (MJ)":   [5, 5, 3, 2, 2, 2],
        "Binder Jetting (BJ)":     [3, 3, 5, 4, 4, 3],
        "DED / LENS":              [2, 1, 4, 4, 3, 5],
        "Sheet Lamination (LOM)":  [2, 2, 4, 2, 5, 2],
        "PBF (SLM/SLS)":           [4, 3, 2, 4, 2, 5],
        "Material Extrusion (FDM)":[2, 2, 3, 3, 5, 3],
    }
    
    print(f"weights: {dict(zip(criteria, weights))}")
    print()
    print(f"{'Process':28s}{'Score':>7s}   Ranked criteria contribution")
    print("-"*70)
    results = []
    for name, sc in processes.items():
        sc = np.array(sc, dtype=float)
        total = float(np.dot(weights, sc))
        results.append((name, total))
    for name, total in sorted(results, key=lambda x: -x[1]):
        bar = "#" * int(round(total*6))
        print(f"{name:28s}{total:7.3f}   {bar}")
    
    best = max(results, key=lambda x: x[1])
    print()
    print(f"Recommended (accuracy-weighted use case): {best[0]}  (score {best[1]:.3f})")
    
    # Re-run with a "cheap large metal part" weighting to show sensitivity
    w2 = np.array([0.05, 0.05, 0.25, 0.15, 0.30, 0.20])
    print()
    print("Re-weighted for 'low-cost large metal part' (cost & speed heavy):")
    r2 = [(n, float(np.dot(w2, np.array(s, dtype=float)))) for n, s in processes.items()]
    for name, total in sorted(r2, key=lambda x: -x[1])[:3]:
        print(f"  {name:28s}{total:7.3f}")

**実行結果：**
    
    
    weights: {'accuracy': np.float64(0.25), 'surface': np.float64(0.15), 'speed': np.float64(0.15), 'material_range': np.float64(0.15), 'cost_eff': np.float64(0.15), 'strength': np.float64(0.15)}
    
    Process                       Score   Ranked criteria contribution
    ----------------------------------------------------------------------
    Binder Jetting (BJ)           3.600   ######################
    PBF (SLM/SLS)                 3.400   ####################
    Material Jetting (MJ)         3.350   ####################
    DED / LENS                    3.050   ##################
    Material Extrusion (FDM)      2.900   #################
    Sheet Lamination (LOM)        2.750   ################
    
    Recommended (accuracy-weighted use case): Binder Jetting (BJ)  (score 3.600)
    
    Re-weighted for 'low-cost large metal part' (cost & speed heavy):
      Binder Jetting (BJ)           3.950
      DED / LENS                    3.650
      Material Extrusion (FDM)      3.500

精度を重視した重みでは結合剤噴射とPBF・材料噴射が拮抗しますが、「安く大きな金属部品」向けにコストと速度を重く取り直すと順位が入れ替わります。**選定は重み（＝要求）次第で変わる** ことを、この感度分析が端的に示しています。

## 演習問題

理解を確認するための演習です。まず自分で考えてから解答を開いてください。

演習1（基礎）: プロセスの対応づけ

次の用途に最も適したAM方式を、MJ／BJ／DED／SL から選び、理由を一言添えてください。  
(a) 摩耗したタービンブレードの補修 (b) エンジンブロックの砂型 (c) 硬軟を作り分けた医療解剖モデル (d) 金属箔の中にセンサーを埋め込む構造

解答を見る

(a) **DED** : 既存部品への肉盛り補修が唯一実用的。  
(b) **BJ** : 砂型を焼結不要で高速・大型に成形できる。  
(c) **MJ** : 1回の造形で複数硬度・透明材を作り分けられる。  
(d) **SL（UAM）** : 低温の固相接合ゆえ、内部にセンサーを埋設できる。

演習2（計算）: Ohnesorge数とZ数

密度 ρ = 1100 kg/m³、粘度 μ = 0.010 Pa·s、表面張力 σ = 0.030 N/m、ノズル径 D = 30 μm の樹脂について、Ohnesorge数とZ数を求め、印刷可能な窓（1 < Z < 10）に入るか判定してください。

解答を見る

Oh = μ / √(ρ σ D) = 0.010 / √(1100 × 0.030 × 30e-6) = 0.010 / √(9.9e-4) = 0.010 / 0.03146 ≈ **0.318** 。  
Z = 1/Oh ≈ **3.15** 。1 < 3.15 < 10 なので**印刷可能な窓の中** にあります。コード例1の関数 `dimensionless` に同じ値を入れて検算できます。

演習3（計算）: 焼結収縮の寸法補正

グリーン相対密度 0.52、焼結後相対密度 0.97 の材料で、最終寸法 40.0 mm を得たい。グリーンパートを何 mm で設計すべきですか。線収縮率も求めてください。

解答を見る

倍率 = (0.97/0.52)^(1/3) = (1.865)^(1/3) ≈ **1.231** 。グリーン寸法 = 40.0 × 1.231 ≈ **49.2 mm** 。  
線収縮率 = (1 − (0.52/0.97)^(1/3)) × 100 = (1 − 0.812) × 100 ≈ **18.8%** 。

演習4（考察）: サテライト液滴

ある樹脂を温めて粘度を下げたところ、Z数が 12 まで上がりました。印刷品質にどんな問題が予想され、どう対処できますか。

解答を見る

Zが窓の上限（約10）を超えると、液滴が尾を引いて分裂し**サテライト液滴** が発生しやすくなります。着弾位置の乱れやミスト汚染につながります。対処としては、(1) 加熱を控えて粘度をやや戻しZを窓内へ下げる、(2) 駆動波形を調整して尾切れを良くする、(3) 液滴速度（We）を最適化する、などが考えられます。ただし最終確認はドロップウォッチャーによる実観察が必要です。

演習5（応用）: 選定重みの設計

「歯科用の高精度フルカラー模型を、少量多品種で作りたい」という要求に対し、コード例3の6軸（精度・表面・速度・材料範囲・コスト効率・強度）の重みをどう設定し、どの方式が選ばれると予想しますか。

解答を見る

精度・表面・材料範囲（フルカラー）を重く、速度・強度・コストを軽くします（例: 精度0.30/表面0.25/材料0.20/速度0.10/コスト0.05/強度0.10）。この重みでは、フルカラーと高精度を両立する**材料噴射（MJ）** が上位に来ると予想されます。実際にコード例3の `weights` を書き換えて確認すると、重み設計が結論を左右することが体感できます。

## まとめ

本章では、材料押出・光造形・粉末床溶融結合に続く残りの主要AMプロセスと、その選定の考え方を学びました。要点は次のとおりです。

  * **材料噴射法（MJ）** : 液滴を印刷して積層。マルチマテリアル・フルカラーが最大の強み。印刷可能性は Oh／Z／We で見積もれるが、最終判断は実観察。
  * **結合剤噴射法（BJ）** : 粉末を糊付けしてグリーンパートを高速成形し、焼結／含浸で緻密化。焼結収縮の寸法補正が必須。
  * **DED／LENS** : 大きく速く盛る・壊れたものを直す方式。精度は粗く、機械加工との併用が前提。
  * **シート積層・ハイブリッド製造** : 大型・センサー埋込・付加＋除去の統合という独自の価値。
  * **選定** : 単一指標でなく、要求に対する総コストとトレードオフで選ぶ。重み付きスコアリングは有効な意思決定支援。
  * **新興技術** : バイオ・4Dプリンティングは有望だが、成熟度を誇張せず正直に評価すること。

**✅ 次章に向けて**

これで主要なAMプロセスの全体像がそろいました。次章では、ここまでの知識を統合し、Pythonによる3Dプリンティングのシミュレーションと解析に取り組みます。

## 次のステップ

第4章では、材料噴射法・結合剤噴射法・指向性エネルギー堆積・シート積層法・ハイブリッド製造を横断的に学び、プロセス選定の定量的な考え方と新興トレンドまで概観しました。次の第5章では、Pythonを用いた3Dプリンティングのシミュレーションと実践的な解析に取り組みます。

[← 第3章へ戻る](<./chapter-3.html>) [第5章へ進む →](<./chapter-5.html>)

## 参考文献

  1. Gibson, I., Rosen, D., & Stucker, B. (2021). _Additive Manufacturing Technologies_ (3rd ed.). Springer. - 材料噴射・結合剤噴射・DED・シート積層を含む全プロセスの標準的教科書
  2. ISO/ASTM 52900:2021. _Additive manufacturing — General principles — Fundamentals and vocabulary_. - AMプロセス分類と用語の国際標準規格
  3. Derby, B. (2010). "Inkjet Printing of Functional and Structural Materials: Fluid Property Requirements, Feature Stability, and Resolution." _Annual Review of Materials Research_ , 40, 395-414. - 液滴印刷可能性とOhnesorge数・Z数の理論的基礎
  4. Ziaee, M., & Crane, N.B. (2019). "Binder Jetting: A Review of Process, Materials, and Methods." _Additive Manufacturing_ , 28, 781-801. - 結合剤噴射のプロセス・材料・焼結の包括的レビュー
  5. Dass, A., & Moridi, A. (2019). "State of the Art in Directed Energy Deposition: From Additive Manufacturing to Materials Design." _Coatings_ , 9(7), 418. - DEDの原理・補修・傾斜機能材料に関する総説
  6. Murphy, S.V., & Atala, A. (2014). "3D Bioprinting of Tissues and Organs." _Nature Biotechnology_ , 32, 773-785. - バイオプリンティングの原理と課題の代表的総説
  7. Tibbits, S. (2014). "4D Printing: Multi-Material Shape Change." _Architectural Design_ , 84(1), 116-121. - 4Dプリンティングの概念を提示した基礎文献

## 使用ツールとライブラリ

  * **NumPy** (v1.24+): 数値計算ライブラリ - <https://numpy.org/>
  * **Matplotlib** (v3.7+): データ可視化ライブラリ - <https://matplotlib.org/>
  * **Python** (v3.10+): 本章のコード例の実行環境 - <https://www.python.org/>

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
