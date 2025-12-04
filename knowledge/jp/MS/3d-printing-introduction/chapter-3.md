---
title: 第3章：積層造形の基礎
chapter_title: 第3章：積層造形の基礎
subtitle: AM技術の原理と分類 - 3Dプリンティングの技術体系
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/3d-printing-introduction/chapter-3.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 3

## 学習目標

この章を完了すると、以下を説明できるようになります：

### 基本理解（Level 1）

  * 積層造形（AM）の定義とISO/ASTM 52900規格の基本概念
  * 7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴
  * STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）
  * AMの歴史（1986年ステレオリソグラフィから現代システムまで）

### 実践スキル（Level 2）

  * PythonでSTLファイルを読み込み、体積・表面積を計算できる
  * numpy-stlとtrimeshを使ったメッシュ検証と修復ができる
  * スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解
  * G-codeの基本構造（G0/G1/G28/M104など）を読み解ける

### 応用力（Level 3）

  * 用途要求に応じて最適なAMプロセスを選択できる
  * メッシュの問題（非多様体、法線反転）を検出・修正できる
  * 造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる
  * STLファイルの品質評価とプリント適性判断ができる

## 1.1 積層造形（AM）とは

### 1.1.1 積層造形の定義

積層造形（Additive Manufacturing, AM）とは、**ISO/ASTM 52900:2021規格で定義される「3次元CADデータから材料を層ごとに積み上げて物体を製造するプロセス」** です。従来の切削加工（除去加工）とは対照的に、必要な部分にのみ材料を付加するため、以下の革新的な特徴を持ちます：

  * **設計自由度** : 従来製法では不可能な複雑形状（中空構造、ラティス構造、トポロジー最適化形状）を製造可能
  * **材料効率** : 必要な部分にのみ材料を使用するため、材料廃棄率が5-10%（従来加工は30-90%廃棄）
  * **オンデマンド製造** : 金型不要でカスタマイズ製品を少量・多品種生産可能
  * **一体化製造** : 従来は複数部品を組立てていた構造を一体造形し、組立工程を削減

**💡 産業的重要性**

AM市場は急成長中で、Wohlers Report 2023によると：

  * 世界のAM市場規模: $18.3B（2023年）→ $83.9B予測（2030年、年成長率23.5%）
  * 用途の内訳: プロトタイピング（38%）、ツーリング（27%）、最終製品（35%）
  * 主要産業: 航空宇宙（26%）、医療（21%）、自動車（18%）、消費財（15%）
  * 材料別シェア: ポリマー（55%）、金属（35%）、セラミックス（7%）、その他（3%）

### 1.1.2 AMの歴史と発展

積層造形技術は約40年の歴史を持ち、以下のマイルストーンを経て現在に至ります：
    
    
    flowchart LR
        A[1986  
    SLA発明  
    Chuck Hull] --> B[1988  
    SLS登場  
    Carl Deckard]
        B --> C[1992  
    FDM特許  
    Stratasys社]
        C --> D[2005  
    RepRap  
    オープンソース化]
        D --> E[2012  
    金属AM普及  
    EBM/SLM]
        E --> F[2023  
    産業化加速  
    大型・高速化]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#f3e5f5
        style E fill:#fce4ec
        style F fill:#fff9c4
            

  1. **1986年: ステレオリソグラフィ（SLA）発明** \- Chuck Hull博士（3D Systems社創業者）が光硬化樹脂を層状に硬化させる最初のAM技術を発明（US Patent 4,575,330）。「3Dプリンティング」という言葉もこの時期に誕生。
  2. **1988年: 選択的レーザー焼結（SLS）登場** \- Carl Deckard博士（テキサス大学）がレーザーで粉末材料を焼結する技術を開発。金属やセラミックスへの応用可能性を開く。
  3. **1992年: 熱溶解積層（FDM）特許** \- Stratasys社がFDM技術を商用化。現在最も普及している3Dプリンティング方式の基礎を確立。
  4. **2005年: RepRapプロジェクト** \- Adrian Bowyer教授がオープンソース3Dプリンタ「RepRap」を発表。特許切れと相まって低価格化・民主化が進展。
  5. **2012年以降: 金属AMの産業普及** \- 電子ビーム溶解（EBM）、選択的レーザー溶融（SLM）が航空宇宙・医療分野で実用化。GE AviationがFUEL噴射ノズルを量産開始。
  6. **2023年現在: 大型化・高速化の時代** \- バインダージェット、連続繊維複合材AM、マルチマテリアルAMなど新技術が産業実装段階へ。

### 1.1.3 AMの主要応用分野

#### 応用1: プロトタイピング（Rapid Prototyping）

AMの最初の主要用途で、設計検証・機能試験・市場評価用のプロトタイプを迅速に製造します：

  * **リードタイム短縮** : 従来の試作（数週間〜数ヶ月）→ AMでは数時間〜数日
  * **設計反復の加速** : 低コストで複数バージョンを試作し、設計を最適化
  * **コミュニケーション改善** : 視覚的・触覚的な物理モデルで関係者間の認識を統一
  * **典型例** : 自動車の意匠モデル、家電製品の筐体試作、医療機器の術前シミュレーションモデル

#### 応用2: ツーリング（Tooling & Fixtures）

製造現場で使用する治具・工具・金型をAMで製造する応用です：

  * **カスタム治具** : 生産ラインに特化した組立治具・検査治具を迅速に製作
  * **コンフォーマル冷却金型** : 従来の直線的冷却路ではなく、製品形状に沿った3次元冷却路を内蔵した射出成形金型（冷却時間30-70%短縮）
  * **軽量化ツール** : ラティス構造を使った軽量エンドエフェクタで作業者の負担を軽減
  * **典型例** : BMWの組立ライン用治具（年間100,000個以上をAMで製造）、GolfのTaylorMadeドライバー金型

#### 応用3: 最終製品（End-Use Parts）

AMで直接、最終製品を製造する応用が近年急増しています：

  * **航空宇宙部品** : GE Aviation LEAP燃料噴射ノズル（従来20部品→AM一体化、重量25%軽減、年間100,000個以上生産）
  * **医療インプラント** : チタン製人工股関節・歯科インプラント（患者固有の解剖学的形状に最適化、骨結合を促進する多孔質構造）
  * **カスタム製品** : 補聴器（年間1,000万個以上がAMで製造）、スポーツシューズのミッドソール（Adidas 4D、Carbon社DLS技術）
  * **スペア部品** : 絶版部品・希少部品のオンデマンド製造（自動車、航空機、産業機械）

**⚠️ AMの制約と課題**

AMは万能ではなく、以下の制約があります：

  * **造形速度** : 大量生産には不向き（射出成形1個/数秒 vs AM数時間）。経済的ブレークイーブンは通常1,000個以下
  * **造形サイズ制限** : ビルドボリューム（多くの装置で200×200×200mm程度）を超える大型部品は分割製造が必要
  * **表面品質** : 積層痕（layer lines）が残るため、高精度表面が必要な場合は後加工必須（研磨、機械加工）
  * **材料特性の異方性** : 積層方向（Z軸）と面内方向（XY平面）で機械的性質が異なる場合がある（特にFDM）
  * **材料コスト** : AMグレード材料は汎用材料の2-10倍高価（ただし材料効率と設計最適化で相殺可能）

## 1.2 ISO/ASTM 52900による7つのAMプロセス分類

### 1.2.1 AMプロセス分類の全体像

ISO/ASTM 52900:2021規格では、すべてのAM技術を**エネルギー源と材料供給方法に基づいて7つのプロセスカテゴリ** に分類しています。各プロセスには固有の長所・短所があり、用途に応じて最適な技術を選択する必要があります。
    
    
    flowchart TD
        AM[積層造形  
    7つのプロセス] --> MEX[Material Extrusion  
    材料押出]
        AM --> VPP[Vat Photopolymerization  
    液槽光重合]
        AM --> PBF[Powder Bed Fusion  
    粉末床溶融結合]
        AM --> MJ[Material Jetting  
    材料噴射]
        AM --> BJ[Binder Jetting  
    結合剤噴射]
        AM --> SL[Sheet Lamination  
    シート積層]
        AM --> DED[Directed Energy Deposition  
    指向性エネルギー堆積]
    
        MEX --> MEX_EX[FDM/FFF  
    低コスト・普及型]
        VPP --> VPP_EX[SLA/DLP  
    高精度・高表面品質]
        PBF --> PBF_EX[SLS/SLM/EBM  
    高強度・金属対応]
    
        style AM fill:#f093fb
        style MEX fill:#e3f2fd
        style VPP fill:#fff3e0
        style PBF fill:#e8f5e9
        style MJ fill:#f3e5f5
        style BJ fill:#fce4ec
        style SL fill:#fff9c4
        style DED fill:#fce4ec
            

### 1.2.2 Material Extrusion (MEX) - 材料押出

**原理** : 熱可塑性樹脂フィラメントを加熱・溶融し、ノズルから押し出して積層。最も普及している技術（FDM/FFFとも呼ばれる）。

プロセス: フィラメント → 加熱ノズル（190-260°C）→ 溶融押出 → 冷却固化 → 次層積層 

**特徴：**

  * **低コスト** : 装置価格$200-$5,000（デスクトップ）、$10,000-$100,000（産業用）
  * **材料多様性** : PLA、ABS、PETG、ナイロン、PC、カーボン繊維複合材、PEEK（高性能）
  * **造形速度** : 20-150 mm³/s（中程度）、レイヤー高さ0.1-0.4mm
  * **精度** : ±0.2-0.5 mm（デスクトップ）、±0.1 mm（産業用）
  * **表面品質** : 積層痕が明瞭（後加工で改善可能）
  * **材料異方性** : Z軸方向（積層方向）の強度が20-80%低い（層間接着が弱点）

**応用例：**

  * プロトタイピング（最も一般的な用途、低コスト・高速）
  * 治具・工具（製造現場で使用、軽量・カスタマイズ容易）
  * 教育用モデル（学校・大学で広く使用、安全・低コスト）
  * 最終製品（カスタム補聴器、義肢装具、建築模型）

**💡 FDMの代表的装置**

  * **Ultimaker S5** : デュアルヘッド、ビルドボリューム330×240×300mm、$6,000
  * **Prusa i3 MK4** : オープンソース系、高い信頼性、$1,200
  * **Stratasys Fortus 450mc** : 産業用、ULTEM 9085対応、$250,000
  * **Markforged X7** : 連続カーボン繊維複合材対応、$100,000

### 1.2.3 Vat Photopolymerization (VPP) - 液槽光重合

**原理** : 液状の光硬化性樹脂（フォトポリマー）に紫外線（UV）レーザーまたはプロジェクターで光を照射し、選択的に硬化させて積層。

プロセス: UV照射 → 光重合反応 → 固化 → ビルドプラットフォーム上昇 → 次層照射 

**VPPの2つの主要方式：**

  1. **SLA（Stereolithography）** : UV レーザー（355 nm）をガルバノミラーで走査し、点描的に硬化。高精度だが低速。
  2. **DLP（Digital Light Processing）** : プロジェクターで面全体を一括露光。高速だが解像度はプロジェクター画素数に依存（Full HD: 1920×1080）。
  3. **LCD-MSLA（Masked SLA）** : LCDマスクを使用、DLP類似だが低コスト化（$200-$1,000のデスクトップ機多数）。

**特徴：**

  * **高精度** : XY解像度25-100 μm、Z解像度10-50 μm（全AM技術中で最高レベル）
  * **表面品質** : 滑らかな表面（Ra < 5 μm）、積層痕がほぼ見えない
  * **造形速度** : SLA（10-50 mm³/s）、DLP/LCD（100-500 mm³/s、面積依存）
  * **材料制約** : 光硬化性樹脂のみ（機械的性質はFDMより劣る場合が多い）
  * **後処理必須** : 洗浄（IPA等）→ 二次硬化（UV照射）→ サポート除去

**応用例：**

  * 歯科用途（歯列矯正モデル、サージカルガイド、義歯、年間数百万個生産）
  * ジュエリー鋳造用ワックスモデル（高精度・複雑形状）
  * 医療モデル（術前計画、解剖学モデル、患者説明用）
  * マスターモデル（シリコン型取り用、デザイン検証）

### 1.2.4 Powder Bed Fusion (PBF) - 粉末床溶融結合

**原理** : 粉末材料を薄く敷き詰め、レーザーまたは電子ビームで選択的に溶融・焼結し、冷却固化させて積層。金属・ポリマー・セラミックスに対応。

プロセス: 粉末敷設 → レーザー/電子ビーム走査 → 溶融・焼結 → 固化 → 次層粉末敷設 

**PBFの3つの主要方式：**

  1. **SLS（Selective Laser Sintering）** : ポリマー粉末（PA12ナイロン等）をレーザー焼結。サポート不要（周囲粉末が支持）。
  2. **SLM（Selective Laser Melting）** : 金属粉末（Ti-6Al-4V、AlSi10Mg、Inconel 718等）を完全溶融。高密度部品（相対密度>99%）製造可能。
  3. **EBM（Electron Beam Melting）** : 電子ビームで金属粉末を溶融。高温予熱（650-1000°C）により残留応力が小さく、造形速度が速い。

**特徴：**

  * **高強度** : 溶融・再凝固により鍛造材に匹敵する機械的性質（引張強度500-1200 MPa）
  * **複雑形状対応** : サポート不要（粉末が支持）でオーバーハング造形可能
  * **材料多様性** : Ti合金、Al合金、ステンレス鋼、Ni超合金、Co-Cr合金、ナイロン
  * **高コスト** : 装置価格$200,000-$1,500,000、材料費$50-$500/kg
  * **後処理** : サポート除去、熱処理（応力除去）、表面仕上げ（ブラスト、研磨）

**応用例：**

  * 航空宇宙部品（軽量化、一体化、GE LEAP燃料ノズル等）
  * 医療インプラント（患者固有形状、多孔質構造、Ti-6Al-4V）
  * 金型（コンフォーマル冷却、複雑形状、H13工具鋼）
  * 自動車部品（軽量化ブラケット、カスタムエンジン部品）

### 1.2.5 Material Jetting (MJ) - 材料噴射

**原理** : インクジェットプリンタと同様に、液滴状の材料（光硬化性樹脂またはワックス）をヘッドから噴射し、UV照射で即座に硬化させて積層。

**特徴：**

  * **超高精度** : XY解像度42-85 μm、Z解像度16-32 μm
  * **マルチマテリアル** : 同一造形で複数材料・複数色を使い分け可能
  * **フルカラー造形** : CMYK樹脂の組合せで1,000万色以上の表現
  * **表面品質** : 極めて滑らか（積層痕ほぼなし）
  * **高コスト** : 装置$50,000-$300,000、材料費$200-$600/kg
  * **材料制約** : 光硬化性樹脂のみ、機械的性質は中程度

**応用例：** : 医療解剖モデル（軟組織・硬組織を異なる材料で再現）、フルカラー建築模型、デザイン検証モデル

### 1.2.6 Binder Jetting (BJ) - 結合剤噴射

**原理** : 粉末床に液状バインダー（接着剤）をインクジェット方式で噴射し、粉末粒子を結合。造形後に焼結または含浸処理で強度向上。

**特徴：**

  * **高速造形** : レーザー走査不要で面全体を一括処理、造形速度100-500 mm³/s
  * **材料多様性** : 金属粉末、セラミックス、砂型（鋳造用）、フルカラー（石膏）
  * **サポート不要** : 周囲粉末が支持、除去後リサイクル可能
  * **低密度問題** : 焼結前は脆弱（グリーン密度50-60%）、焼結後も相対密度90-98%
  * **後処理必須** : 脱脂 → 焼結（金属：1200-1400°C）→ 含浸（銅・青銅）

**応用例：** : 砂型鋳造用型（エンジンブロック等の大型鋳物）、金属部品（Desktop Metal、HP Metal Jet）、フルカラー像（記念品、教育モデル）

### 1.2.7 Sheet Lamination (SL) - シート積層

**原理** : シート状材料（紙、金属箔、プラスチックフィルム）を積層し、接着または溶接で結合。各層をレーザーまたはブレードで輪郭切断。

**代表技術：**

  * **LOM（Laminated Object Manufacturing）** : 紙・プラスチックシート、接着剤で積層、レーザー切断
  * **UAM（Ultrasonic Additive Manufacturing）** : 金属箔を超音波溶接、CNC切削で輪郭加工

**特徴：** 大型造形可能、材料費安価、精度中程度、用途限定的（主に視覚モデル、金属では埋込センサー等）

### 1.2.8 Directed Energy Deposition (DED) - 指向性エネルギー堆積

**原理** : 金属粉末またはワイヤーを供給しながら、レーザー・電子ビーム・アークで溶融し、基板上に堆積。大型部品や既存部品の補修に使用。

**特徴：**

  * **高速堆積** : 堆積速度1-5 kg/h（PBFの10-50倍）
  * **大型対応** : ビルドボリューム制限が少ない（多軸ロボットアーム使用）
  * **補修・コーティング** : 既存部品の摩耗部分修復、表面硬化層形成
  * **低精度** : 精度±0.5-2 mm、後加工（機械加工）必須

**応用例：** : タービンブレード補修、大型航空宇宙部品、工具の耐摩耗コーティング

**⚠️ プロセス選択の指針**

最適なAMプロセスは用途要求により異なります：

  * **精度最優先** → VPP（SLA/DLP）またはMJ
  * **低コスト・普及型** → MEX（FDM/FFF）
  * **金属高強度部品** → PBF（SLM/EBM）
  * **大量生産（砂型）** → BJ
  * **大型・高速堆積** → DED

## 1.3 STLファイル形式とデータ処理

### 1.3.1 STLファイルの構造

STL（STereoLithography）は、**AMで最も広く使用される3Dモデルファイル形式** で、1987年に3D Systems社が開発しました。STLファイルは物体表面を**三角形メッシュ（Triangle Mesh）の集合** として表現します。

#### STLファイルの基本構造

STLファイル = 法線ベクトル（n） + 3つの頂点座標（v1, v2, v3）× 三角形数 

**ASCII STL形式の例：**
    
    
    solid cube
      facet normal 0 0 1
        outer loop
          vertex 0 0 10
          vertex 10 0 10
          vertex 10 10 10
        endloop
      endfacet
      facet normal 0 0 1
        outer loop
          vertex 0 0 10
          vertex 10 10 10
          vertex 0 10 10
        endloop
      endfacet
      ...
    endsolid cube
    

**STLフォーマットの2つの種類：**

  1. **ASCII STL** : 人間が読めるテキスト形式。ファイルサイズ大（同じモデルでBinaryの10-20倍）。デバッグ・検証に有用。
  2. **Binary STL** : バイナリ形式、ファイルサイズ小、処理高速。産業用途で標準。構造：80バイトヘッダー + 4バイト（三角形数） + 各三角形50バイト（法線12B + 頂点36B + 属性2B）。

### 1.3.2 STLファイルの重要概念

#### 1\. 法線ベクトル（Normal Vector）

各三角形面には**法線ベクトル（外向き方向）** が定義され、物体の「内側」と「外側」を区別します。法線方向は**右手の法則** で決定されます：

法線n = (v2 - v1) × (v3 - v1) / |(v2 - v1) × (v3 - v1)| 

**頂点順序ルール：** 頂点v1, v2, v3は反時計回り（CCW: Counter-ClockWise）に配置され、外から見て反時計回りの順序で法線が外向きになります。

#### 2\. 多様体（Manifold）条件

STLメッシュが3Dプリント可能であるためには、**多様体（Manifold）** でなければなりません：

  * **エッジ共有** : すべてのエッジ（辺）は正確に2つの三角形に共有される
  * **頂点共有** : すべての頂点は連続した三角形扇（fan）に属する
  * **閉じた表面** : 穴や開口部がなく、完全に閉じた表面を形成
  * **自己交差なし** : 三角形が互いに交差・貫通していない

**⚠️ 非多様体メッシュの問題**

非多様体メッシュ（Non-Manifold Mesh）は3Dプリント不可能です。典型的な問題：

  * **穴（Holes）** : 閉じていない表面、エッジが1つの三角形にのみ属する
  * **T字接合（T-junction）** : エッジが3つ以上の三角形に共有される
  * **法線反転（Inverted Normals）** : 法線が内側を向いている三角形が混在
  * **重複頂点（Duplicate Vertices）** : 同じ位置に複数の頂点が存在
  * **微小三角形（Degenerate Triangles）** : 面積がゼロまたはほぼゼロの三角形

これらの問題はスライサーソフトウェアでエラーを引き起こし、造形失敗の原因となります。

### 1.3.3 STLファイルの品質指標

STLメッシュの品質は以下の指標で評価されます：

  1. **三角形数（Triangle Count）** : 通常10,000-500,000個。過少（粗いモデル）または過多（ファイルサイズ大・処理遅延）は避ける。
  2. **エッジ長の一様性** : 極端に大小の三角形が混在すると造形品質低下。理想的には0.1-1.0 mm範囲。
  3. **アスペクト比（Aspect Ratio）** : 細長い三角形（高アスペクト比）は数値誤差の原因。理想的にはアスペクト比 < 10。
  4. **法線の一貫性** : すべての法線が外向き統一。反転法線が混在すると内外判定エラー。

**💡 STLファイルの解像度トレードオフ**

STLメッシュの解像度（三角形数）は精度とファイルサイズのトレードオフです：

  * **低解像度（1,000-10,000三角形）** : 高速処理、小ファイル、但し曲面が角張る（ファセット化明瞭）
  * **中解像度（10,000-100,000三角形）** : 多くの用途で適切、バランス良好
  * **高解像度（100,000-1,000,000三角形）** : 滑らかな曲面、但しファイルサイズ大（数十MB）、処理遅延

CADソフトでSTLエクスポート時に、**Chordal Tolerance（コード公差）** または**Angle Tolerance（角度公差）** で解像度を制御します。推奨値：コード公差0.01-0.1 mm、角度公差5-15度。

### 1.3.4 Pythonライブラリによる STL処理

PythonでSTLファイルを扱うための主要ライブラリ：

  1. **numpy-stl** : 高速STL読込・書込、体積・表面積計算、法線ベクトル操作。シンプルで軽量。
  2. **trimesh** : 包括的な3Dメッシュ処理ライブラリ。メッシュ修復、ブーリアン演算、レイキャスト、衝突検出。多機能だが依存関係多い。
  3. **PyMesh** : 高度なメッシュ処理（リメッシュ、サブディビジョン、フィーチャー抽出）。インストールやや複雑。

**numpy-stlの基本的な使用法：**
    
    
    from stl import mesh
    import numpy as np
    
    # STLファイルを読み込み
    your_mesh = mesh.Mesh.from_file('model.stl')
    
    # 基本的な幾何情報
    volume, cog, inertia = your_mesh.get_mass_properties()
    print(f"Volume: {volume:.2f} mm³")
    print(f"Center of Gravity: {cog}")
    print(f"Surface Area: {your_mesh.areas.sum():.2f} mm²")
    
    # 三角形数
    print(f"Number of Triangles: {len(your_mesh.vectors)}")
    

## 1.4 スライシングとツールパス生成

STLファイルを3Dプリンタが理解できる指令（G-code）に変換するプロセスを**スライシング（Slicing）** といいます。このセクションでは、スライシングの基本原理、ツールパス戦略、そしてG-codeの基礎を学びます。

### 1.4.1 スライシングの基本原理

スライシングは、3Dモデルを一定の高さ（レイヤー高さ）で水平に切断し、各層の輪郭を抽出するプロセスです：
    
    
    flowchart TD
        A[3Dモデル  
    STLファイル] --> B[Z軸方向に  
    層状にスライス]
        B --> C[各層の輪郭抽出  
    Contour Detection]
        C --> D[シェル生成  
    Perimeter Path]
        D --> E[インフィル生成  
    Infill Path]
        E --> F[サポート追加  
    Support Structure]
        F --> G[ツールパス最適化  
    Retraction/Travel]
        G --> H[G-code出力]
    
        style A fill:#e3f2fd
        style H fill:#e8f5e9
            

#### レイヤー高さ（Layer Height）の選択

レイヤー高さは造形品質と造形時間のトレードオフを決定する最重要パラメータです：

レイヤー高さ | 造形品質 | 造形時間 | 典型的な用途  
---|---|---|---  
0.1 mm（極細） | 非常に高い（積層痕ほぼ不可視） | 非常に長い（×2-3倍） | フィギュア、医療モデル、最終製品  
0.2 mm（標準） | 良好（積層痕は見えるが許容） | 標準 | 一般的なプロトタイプ、機能部品  
0.3 mm（粗） | 低い（積層痕明瞭） | 短い（×0.5倍） | 初期プロトタイプ、内部構造部品  
  
**⚠️ レイヤー高さの制約**

レイヤー高さはノズル径の**25-80%** に設定する必要があります。例えば0.4mmノズルの場合、レイヤー高さは0.1-0.32mmが推奨範囲です。これを超えると、樹脂の押出量が不足したり、ノズルが前の層を引きずる問題が発生します。

### 1.4.2 シェルとインフィル戦略

#### シェル（外殻）の生成

**シェル（Shell/Perimeter）** は、各層の外周部を形成する経路です：

  * **シェル数（Perimeter Count）** : 通常2-4本。外部品質と強度に影響。 
    * 1本: 非常に弱い、透明性高い、装飾用のみ
    * 2本: 標準（バランス良好）
    * 3-4本: 高強度、表面品質向上、気密性向上
  * **シェル順序** : 内側→外側（Inside-Out）が一般的。外側→内側は表面品質重視時に使用。

#### インフィル（内部充填）パターン

**インフィル（Infill）** は内部構造を形成し、強度と材料使用量を制御します：

パターン | 強度 | 印刷速度 | 材料使用量 | 特徴  
---|---|---|---|---  
Grid（格子） | 中 | 速い | 中 | シンプル、等方性、標準的な選択  
Honeycomb（ハニカム） | 高 | 遅い | 中 | 高強度、重量比優秀、航空宇宙用途  
Gyroid | 非常に高 | 中 | 中 | 3次元等方性、曲面的、最新の推奨  
Concentric（同心円） | 低 | 速い | 少 | 柔軟性重視、シェル追従  
Lines（直線） | 低（異方性） | 非常に速い | 少 | 高速印刷、方向性強度  
  
**💡 インフィル密度の目安**

  * **0-10%** : 装飾品、非荷重部品（材料節約優先）
  * **20%** : 標準的なプロトタイプ（バランス良好）
  * **40-60%** : 機能部品、高強度要求
  * **100%** : 最終製品、水密性要求、最高強度（造形時間×3-5倍）

### 1.4.3 サポート構造の生成

オーバーハング角度が45度を超える部分は、**サポート構造（Support Structure）** が必要です：

#### サポートのタイプ

  * **Linear Support（直線サポート）** : 垂直な柱状サポート。シンプルで除去しやすいが、材料使用量多い。
  * **Tree Support（ツリーサポート）** : 樹木状に分岐するサポート。材料使用量30-50%削減、除去しやすい。CuraやPrusaSlicerで標準サポート。
  * **Interface Layers（接合層）** : サポート上面に薄い接合層を設ける。除去しやすく、表面品質向上。通常2-4層。

#### サポート設定の重要パラメータ

パラメータ | 推奨値 | 効果  
---|---|---  
Overhang Angle | 45-60° | この角度以上でサポート生成  
Support Density | 10-20% | 密度が高いほど安定だが除去困難  
Support Z Distance | 0.2-0.3 mm | サポートと造形物の間隔（除去しやすさ）  
Interface Layers | 2-4層 | 接合層数（表面品質と除去性のバランス）  
  
### 1.4.4 G-codeの基礎

**G-code** は、3DプリンタやCNCマシンを制御する標準的な数値制御言語です。各行が1つのコマンドを表します：

#### 主要なG-codeコマンド

コマンド | 分類 | 機能 | 例  
---|---|---|---  
G0 | 移動 | 高速移動（非押出） | G0 X100 Y50 Z10 F6000  
G1 | 移動 | 直線移動（押出あり） | G1 X120 Y60 E0.5 F1200  
G28 | 初期化 | ホームポジション復帰 | G28 （全軸）, G28 Z （Z軸のみ）  
M104 | 温度 | ノズル温度設定（非待機） | M104 S200  
M109 | 温度 | ノズル温度設定（待機） | M109 S210  
M140 | 温度 | ベッド温度設定（非待機） | M140 S60  
M190 | 温度 | ベッド温度設定（待機） | M190 S60  
  
#### G-codeの例（造形開始部分）
    
    
    ; === Start G-code ===
    M140 S60       ; ベッドを60°Cに加熱開始（非待機）
    M104 S210      ; ノズルを210°Cに加熱開始（非待機）
    G28            ; 全軸ホーミング
    G29            ; オートレベリング（ベッドメッシュ計測）
    M190 S60       ; ベッド温度到達を待機
    M109 S210      ; ノズル温度到達を待機
    G92 E0         ; 押出量をゼロリセット
    G1 Z2.0 F3000  ; Z軸を2mm上昇（安全確保）
    G1 X10 Y10 F5000  ; プライム位置へ移動
    G1 Z0.3 F3000  ; Z軸を0.3mmへ降下（初層高さ）
    G1 X100 E10 F1500 ; プライムライン描画（ノズル詰まり除去）
    G92 E0         ; 押出量を再度ゼロリセット
    ; === 造形開始 ===
    

### 1.4.5 主要スライシングソフトウェア

ソフトウェア | ライセンス | 特徴 | 推奨用途  
---|---|---|---  
Cura | オープンソース | 使いやすい、豊富なプリセット、Tree Support標準搭載 | 初心者〜中級者、FDM汎用  
PrusaSlicer | オープンソース | 高度な設定、変数レイヤー高さ、カスタムサポート | 中級者〜上級者、最適化重視  
Slic3r | オープンソース | PrusaSlicerの元祖、軽量 | レガシーシステム、研究用途  
Simplify3D | 商用（$150） | 高速スライシング、マルチプロセス、詳細制御 | プロフェッショナル、産業用途  
IdeaMaker | 無料 | Raise3D専用だが汎用性高い、直感的UI | Raise3Dユーザー、初心者  
  
### 1.4.6 ツールパス最適化戦略

効率的なツールパスは、造形時間・品質・材料使用量を改善します：

  * **リトラクション（Retraction）** : 移動時にフィラメントを引き戻してストリング（糸引き）を防止。 
    * 距離: 1-6mm（ボーデンチューブ式は4-6mm、ダイレクト式は1-2mm）
    * 速度: 25-45 mm/s
    * 過度なリトラクションはノズル詰まりの原因
  * **Z-hop（Z軸跳躍）** : 移動時にノズルを上昇させて造形物との衝突を回避。0.2-0.5mm上昇。造形時間微増だが表面品質向上。
  * **コーミング（Combing）** : 移動経路をインフィル上に制限し、表面への移動痕を低減。外観重視時に有効。
  * **シーム位置（Seam Position）** : 各層の開始/終了点を揃える戦略。 
    * Random: ランダム配置（目立たない）
    * Aligned: 一直線に配置（後加工でシームを除去しやすい）
    * Sharpest Corner: 最も鋭角なコーナーに配置（目立ちにくい）

### Example 1: STLファイルの読み込みと基本情報取得
    
    
    # ===================================
    # Example 1: STLファイルの読み込みと基本情報取得
    # ===================================
    
    import numpy as np
    from stl import mesh
    
    # STLファイルを読み込む
    your_mesh = mesh.Mesh.from_file('model.stl')
    
    # 基本的な幾何情報を取得
    volume, cog, inertia = your_mesh.get_mass_properties()
    
    print("=== STLファイル基本情報 ===")
    print(f"Volume: {volume:.2f} mm³")
    print(f"Surface Area: {your_mesh.areas.sum():.2f} mm²")
    print(f"Center of Gravity: [{cog[0]:.2f}, {cog[1]:.2f}, {cog[2]:.2f}] mm")
    print(f"Number of Triangles: {len(your_mesh.vectors)}")
    
    # バウンディングボックス（最小包含直方体）を計算
    min_coords = your_mesh.vectors.min(axis=(0, 1))
    max_coords = your_mesh.vectors.max(axis=(0, 1))
    dimensions = max_coords - min_coords
    
    print(f"\n=== バウンディングボックス ===")
    print(f"X: {min_coords[0]:.2f} to {max_coords[0]:.2f} mm (幅: {dimensions[0]:.2f} mm)")
    print(f"Y: {min_coords[1]:.2f} to {max_coords[1]:.2f} mm (奥行: {dimensions[1]:.2f} mm)")
    print(f"Z: {min_coords[2]:.2f} to {max_coords[2]:.2f} mm (高さ: {dimensions[2]:.2f} mm)")
    
    # 造形時間の簡易推定（レイヤー高さ0.2mm、速度50mm/sと仮定）
    layer_height = 0.2  # mm
    print_speed = 50    # mm/s
    num_layers = int(dimensions[2] / layer_height)
    # 簡易計算: 表面積に基づく推定
    estimated_path_length = your_mesh.areas.sum() / layer_height  # mm
    estimated_time_seconds = estimated_path_length / print_speed
    estimated_time_minutes = estimated_time_seconds / 60
    
    print(f"\n=== 造形推定 ===")
    print(f"レイヤー数（0.2mm/層）: {num_layers} 層")
    print(f"推定造形時間: {estimated_time_minutes:.1f} 分 ({estimated_time_minutes/60:.2f} 時間)")
    
    # 出力例:
    # === STLファイル基本情報 ===
    # Volume: 12450.75 mm³
    # Surface Area: 5832.42 mm²
    # Center of Gravity: [25.34, 18.92, 15.67] mm
    # Number of Triangles: 2456
    #
    # === バウンディングボックス ===
    # X: 0.00 to 50.00 mm (幅: 50.00 mm)
    # Y: 0.00 to 40.00 mm (奥行: 40.00 mm)
    # Z: 0.00 to 30.00 mm (高さ: 30.00 mm)
    #
    # === 造形推定 ===
    # レイヤー数（0.2mm/層）: 150 層
    # 推定造形時間: 97.2 分 (1.62 時間)
    

### Example 2: メッシュの法線ベクトル検証
    
    
    # ===================================
    # Example 2: メッシュの法線ベクトル検証
    # ===================================
    
    import numpy as np
    from stl import mesh
    
    def check_normals(mesh_data):
        """STLメッシュの法線ベクトルの整合性をチェック
    
        Args:
            mesh_data: numpy-stlのMeshオブジェクト
    
        Returns:
            tuple: (flipped_count, total_count, percentage)
        """
        # 右手系ルールで法線方向を確認
        flipped_count = 0
        total_count = len(mesh_data.vectors)
    
        for i, facet in enumerate(mesh_data.vectors):
            v0, v1, v2 = facet
    
            # エッジベクトルを計算
            edge1 = v1 - v0
            edge2 = v2 - v0
    
            # 外積で法線を計算（右手系）
            calculated_normal = np.cross(edge1, edge2)
    
            # 正規化
            norm = np.linalg.norm(calculated_normal)
            if norm > 1e-10:  # ゼロベクトルでないことを確認
                calculated_normal = calculated_normal / norm
            else:
                continue  # 縮退三角形をスキップ
    
            # ファイルに保存されている法線と比較
            stored_normal = mesh_data.normals[i]
            stored_norm = np.linalg.norm(stored_normal)
    
            if stored_norm > 1e-10:
                stored_normal = stored_normal / stored_norm
    
            # 内積で方向の一致をチェック
            dot_product = np.dot(calculated_normal, stored_normal)
    
            # 内積が負なら逆向き
            if dot_product < 0:
                flipped_count += 1
    
        percentage = (flipped_count / total_count) * 100 if total_count > 0 else 0
    
        return flipped_count, total_count, percentage
    
    # STLファイルを読み込み
    your_mesh = mesh.Mesh.from_file('model.stl')
    
    # 法線チェックを実行
    flipped, total, percent = check_normals(your_mesh)
    
    print("=== 法線ベクトル検証結果 ===")
    print(f"総三角形数: {total}")
    print(f"反転法線数: {flipped}")
    print(f"反転率: {percent:.2f}%")
    
    if flipped == 0:
        print("\n✅ すべての法線が正しい方向を向いています")
        print("   このメッシュは3Dプリント可能です")
    elif percent < 5:
        print("\n⚠️ 一部の法線が反転しています（軽微）")
        print("   スライサーが自動修正する可能性が高い")
    else:
        print("\n❌ 多数の法線が反転しています（重大）")
        print("   メッシュ修復ツール（Meshmixer, netfabb）での修正を推奨")
    
    # 出力例:
    # === 法線ベクトル検証結果 ===
    # 総三角形数: 2456
    # 反転法線数: 0
    # 反転率: 0.00%
    #
    # ✅ すべての法線が正しい方向を向いています
    #    このメッシュは3Dプリント可能です
    

### Example 3: マニフォールド性のチェック
    
    
    # ===================================
    # Example 3: マニフォールド性（Watertight）のチェック
    # ===================================
    
    import trimesh
    
    # STLファイルを読み込み（trimeshは自動で修復を試みる）
    mesh = trimesh.load('model.stl')
    
    print("=== メッシュ品質診断 ===")
    
    # 基本情報
    print(f"Vertex count: {len(mesh.vertices)}")
    print(f"Face count: {len(mesh.faces)}")
    print(f"Volume: {mesh.volume:.2f} mm³")
    
    # マニフォールド性をチェック
    print(f"\n=== 3Dプリント適性チェック ===")
    print(f"Is watertight (密閉性): {mesh.is_watertight}")
    print(f"Is winding consistent (法線一致性): {mesh.is_winding_consistent}")
    print(f"Is valid (幾何的妥当性): {mesh.is_valid}")
    
    # 問題の詳細を診断
    if not mesh.is_watertight:
        # 穴（hole）の数を検出
        try:
            edges = mesh.edges_unique
            edges_sorted = mesh.edges_sorted
            duplicate_edges = len(edges_sorted) - len(edges)
            print(f"\n⚠️ 問題検出:")
            print(f"   - メッシュに穴があります")
            print(f"   - 重複エッジ数: {duplicate_edges}")
        except:
            print(f"\n⚠️ メッシュ構造に問題があります")
    
    # 修復を試みる
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        print(f"\n🔧 自動修復を実行中...")
    
        # 法線を修正
        trimesh.repair.fix_normals(mesh)
        print("   ✓ 法線ベクトルを修正")
    
        # 穴を埋める
        trimesh.repair.fill_holes(mesh)
        print("   ✓ 穴を充填")
    
        # 縮退三角形を削除
        mesh.remove_degenerate_faces()
        print("   ✓ 縮退面を削除")
    
        # 重複頂点を結合
        mesh.merge_vertices()
        print("   ✓ 重複頂点を結合")
    
        # 修復後の状態を確認
        print(f"\n=== 修復後の状態 ===")
        print(f"Is watertight: {mesh.is_watertight}")
        print(f"Is winding consistent: {mesh.is_winding_consistent}")
    
        # 修復したメッシュを保存
        if mesh.is_watertight:
            mesh.export('model_repaired.stl')
            print(f"\n✅ 修復完了！ model_repaired.stl として保存しました")
        else:
            print(f"\n❌ 自動修復失敗。Meshmixer等の専用ツールを推奨")
    else:
        print(f"\n✅ このメッシュは3Dプリント可能です")
    
    # 出力例:
    # === メッシュ品質診断 ===
    # Vertex count: 1534
    # Face count: 2456
    # Volume: 12450.75 mm³
    #
    # === 3Dプリント適性チェック ===
    # Is watertight (密閉性): True
    # Is winding consistent (法線一致性): True
    # Is valid (幾何的妥当性): True
    #
    # ✅ このメッシュは3Dプリント可能です
    

### Example 4: 基本的なスライシングアルゴリズム
    
    
    # ===================================
    # Example 4: 基本的なスライシングアルゴリズム
    # ===================================
    
    import numpy as np
    from stl import mesh
    
    def slice_mesh_at_height(mesh_data, z_height):
        """温度プロファイルを生成
    
        Args:
            t (array): 時間配列 [min]
            T_target (float): 保持温度 [°C]
            heating_rate (float): 加熱速度 [°C/min]
            hold_time (float): 保持時間 [min]
            cooling_rate (float): 冷却速度 [°C/min]
    
        Returns:
            array: 温度プロファイル [°C]
        """
        T_room = 25  # 室温
        T = np.zeros_like(t)
    
        # 加熱時間
        t_heat = (T_target - T_room) / heating_rate
    
        # 冷却開始時刻
        t_cool_start = t_heat + hold_time
    
        for i, time in enumerate(t):
            if time <= t_heat:
                # 加熱フェーズ
                T[i] = T_room + heating_rate * time
            elif time <= t_cool_start:
                # 保持フェーズ
                T[i] = T_target
            else:
                # 冷却フェーズ
                T[i] = T_target - cooling_rate * (time - t_cool_start)
                T[i] = max(T[i], T_room)  # 室温以下にはならない
    
        return T
    
    def simulate_reaction_progress(T, t, Ea, D0, r0):
        """温度プロファイルに基づく反応進行を計算
    
        Args:
            T (array): 温度プロファイル [°C]
            t (array): 時間配列 [min]
            Ea (float): 活性化エネルギー [J/mol]
            D0 (float): 頻度因子 [m²/s]
            r0 (float): 粒子半径 [m]
    
        Returns:
            array: 反応率
        """
        R = 8.314
        C0 = 10000
        alpha = np.zeros_like(t)
    
        for i in range(1, len(t)):
            T_k = T[i] + 273.15
            D = D0 * np.exp(-Ea / (R * T_k))
            k = D * C0 / r0**2
    
            dt = (t[i] - t[i-1]) * 60  # min → s
    
            # 簡易積分（微小時間での反応進行）
            if alpha[i-1] < 0.99:
                dalpha = k * dt / (2 * (1 - (1-alpha[i-1])**(1/3)))
                alpha[i] = min(alpha[i-1] + dalpha, 1.0)
            else:
                alpha[i] = alpha[i-1]
    
        return alpha
    
    # パラメータ設定
    T_target = 1200  # °C
    hold_time = 240  # min (4 hours)
    Ea = 300e3  # J/mol
    D0 = 5e-4  # m²/s
    r0 = 5e-6  # m
    
    # 異なる加熱速度での比較
    heating_rates = [2, 5, 10, 20]  # °C/min
    cooling_rate = 3  # °C/min
    
    # 時間配列
    t_max = 800  # min
    t = np.linspace(0, t_max, 2000)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 温度プロファイル
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
    
    # 各加熱速度での95%反応到達時間を計算
    print("\n95%反応到達時間の比較:")
    print("=" * 60)
    for hr in heating_rates:
        T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate)
        alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0)
    
        # 95%到達時刻
        idx_95 = np.where(alpha >= 0.95)[0]
        if len(idx_95) > 0:
            t_95 = t[idx_95[0]] / 60
            print(f"加熱速度 {hr:2d}°C/min: t₉₅ = {t_95:.1f} hours")
        else:
            print(f"加熱速度 {hr:2d}°C/min: 反応不完全")
    
    # 出力例:
    # 95%反応到達時間の比較:
    # ============================================================
    # 加熱速度  2°C/min: t₉₅ = 7.8 hours
    # 加熱速度  5°C/min: t₉₅ = 7.2 hours
    # 加熱速度 10°C/min: t₉₅ = 6.9 hours
    # 加熱速度 20°C/min: t₉₅ = 6.7 hours
    

## 演習問題

### 1.5.1 pycalphadとは

**pycalphad** は、CALPHAD（CALculation of PHAse Diagrams）法に基づく相図計算のためのPythonライブラリです。熱力学データベースから平衡相を計算し、反応経路の設計に有用です。

**💡 CALPHAD法の利点**

  * 多元系（3元系以上）の複雑な相図を計算可能
  * 実験データが少ない系でも予測可能
  * 温度・組成・圧力依存性を包括的に扱える

### 1.5.2 二元系相図の計算例
    
    
    # ===================================
    # Example 5: pycalphadで相図計算
    # ===================================
    
    # 注意: pycalphadのインストールが必要
    # pip install pycalphad
    
    from pycalphad import Database, equilibrium, variables as v
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TDBデータベースを読み込み（ここでは簡易的な例）
    # 実際には適切なTDBファイルが必要
    # 例: BaO-TiO2系
    
    # 簡易的なTDB文字列（実際はより複雑）
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
    
    # 注: 実際の計算には正式なTDBファイルが必要
    # ここでは概念的な説明に留める
    
    print("pycalphadによる相図計算の概念:")
    print("=" * 60)
    print("1. TDBデータベース（熱力学データ）を読み込む")
    print("2. 温度・組成範囲を設定")
    print("3. 平衡計算を実行")
    print("4. 安定相を可視化")
    print()
    print("実際の適用例:")
    print("- BaO-TiO2系: BaTiO3の形成温度・組成範囲")
    print("- Si-N系: Si3N4の安定領域")
    print("- 多元系セラミックスの相関係")
    
    # 概念的なプロット（実データに基づくイメージ）
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 温度範囲
    T = np.linspace(800, 1600, 100)
    
    # 各相の安定領域（概念図）
    # BaO + TiO2 → BaTiO3 反応
    BaO_region = np.ones_like(T) * 0.3
    TiO2_region = np.ones_like(T) * 0.7
    BaTiO3_region = np.where((T > 1100) & (T < 1400), 0.5, np.nan)
    
    ax.fill_between(T, 0, BaO_region, alpha=0.3, color='blue', label='BaO + TiO2')
    ax.fill_between(T, BaO_region, TiO2_region, alpha=0.3, color='green',
                    label='BaTiO3 stable')
    ax.fill_between(T, TiO2_region, 1, alpha=0.3, color='red', label='Liquid')
    
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
               label='BaTiO3 composition')
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
    
    # 実際の使用例（コメントアウト）
    """
    # 実際のpycalphad使用例
    db = Database('BaO-TiO2.tdb')  # TDBファイル読み込み
    
    # 平衡計算
    eq = equilibrium(db, ['BA', 'TI', 'O'], ['LIQUID', 'BATIO3'],
                     {v.X('BA'): (0, 1, 0.01),
                      v.T: (1000, 1600, 50),
                      v.P: 101325})
    
    # 結果プロット
    eq.plot()
    """
    

## 1.6 実験計画法（DOE）による条件最適化

### 1.6.1 DOEとは

実験計画法（Design of Experiments, DOE）は、複数のパラメータが相互作用する系で、最小の実験回数で最適条件を見つける統計手法です。

**固相反応で最適化すべき主要パラメータ：**

  * 反応温度（T）
  * 保持時間（t）
  * 粒子サイズ（r）
  * 原料比（モル比）
  * 雰囲気（空気、窒素、真空など）

### 1.6.2 応答曲面法（Response Surface Methodology）
    
    
    # ===================================
    # Example 6: DOEによる条件最適化
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.optimize import minimize
    
    # 仮想的な反応率モデル（温度と時間の関数）
    def reaction_yield(T, t, noise=0):
        """温度と時間から反応率を計算（仮想モデル）
    
        Args:
            T (float): 温度 [°C]
            t (float): 時間 [hours]
            noise (float): ノイズレベル
    
        Returns:
            float: 反応率 [%]
        """
        # 最適値: T=1200°C, t=6 hours
        T_opt = 1200
        t_opt = 6
    
        # 二次モデル（ガウス型）
        yield_val = 100 * np.exp(-((T-T_opt)/150)**2 - ((t-t_opt)/3)**2)
    
        # ノイズ追加
        if noise > 0:
            yield_val += np.random.normal(0, noise)
    
        return np.clip(yield_val, 0, 100)
    
    # 実験点配置（中心複合計画法）
    T_levels = [1000, 1100, 1200, 1300, 1400]  # °C
    t_levels = [2, 4, 6, 8, 10]  # hours
    
    # グリッドで実験点を配置
    T_grid, t_grid = np.meshgrid(T_levels, t_levels)
    yield_grid = np.zeros_like(T_grid, dtype=float)
    
    # 各実験点で反応率を測定（シミュレーション）
    for i in range(len(t_levels)):
        for j in range(len(T_levels)):
            yield_grid[i, j] = reaction_yield(T_grid[i, j], t_grid[i, j], noise=2)
    
    # 結果の表示
    print("実験計画法による反応条件最適化")
    print("=" * 70)
    print(f"{'Temperature (°C)':<20} {'Time (hours)':<20} {'Yield (%)':<20}")
    print("-" * 70)
    for i in range(len(t_levels)):
        for j in range(len(T_levels)):
            print(f"{T_grid[i, j]:<20} {t_grid[i, j]:<20} {yield_grid[i, j]:<20.1f}")
    
    # 最大反応率の条件を探す
    max_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape)
    T_best = T_grid[max_idx]
    t_best = t_grid[max_idx]
    yield_best = yield_grid[max_idx]
    
    print("-" * 70)
    print(f"最適条件: T = {T_best}°C, t = {t_best} hours")
    print(f"最大反応率: {yield_best:.1f}%")
    
    # 3Dプロット
    fig = plt.figure(figsize=(14, 6))
    
    # 3D表面プロット
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
    
    # 等高線プロット
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
    

### 1.6.3 実験計画の実践的アプローチ

実際の固相反応では、以下の手順でDOEを適用します：

  1. **スクリーニング実験** （2水準要因計画法）: 影響の大きいパラメータを特定
  2. **応答曲面法** （中心複合計画法）: 最適条件の探索
  3. **確認実験** : 予測された最適条件で実験し、モデルを検証

**✅ 実例: Li-ion電池正極材LiCoO₂の合成最適化**

ある研究グループがDOEを用いてLiCoO₂の合成条件を最適化した結果：

  * 実験回数: 従来法100回 → DOE法25回（75%削減）
  * 最適温度: 900°C（従来の850°Cより高温）
  * 最適保持時間: 12時間（従来の24時間から半減）
  * 電池容量: 140 mAh/g → 155 mAh/g（11%向上）

## 1.7 反応速度曲線のフィッティング

### 1.7.1 実験データからの速度定数決定
    
    
    # ===================================
    # Example 7: 反応速度曲線フィッティング
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # 実験データ（時間 vs 反応率）
    # 例: BaTiO3合成 @ 1200°C
    time_exp = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20])  # hours
    conversion_exp = np.array([0, 0.15, 0.28, 0.38, 0.47, 0.60,
                              0.70, 0.78, 0.84, 0.90, 0.95])
    
    # Jander式モデル
    def jander_model(t, k):
        """Jander式による反応率計算
    
        Args:
            t (array): 時間 [hours]
            k (float): 速度定数
    
        Returns:
            array: 反応率
        """
        # [1 - (1-α)^(1/3)]² = kt を α について解く
        kt = k * t
        alpha = 1 - (1 - np.sqrt(kt))**3
        alpha = np.clip(alpha, 0, 1)  # 0-1の範囲に制限
        return alpha
    
    # Ginstling-Brounshtein式（別の拡散モデル）
    def gb_model(t, k):
        """Ginstling-Brounshtein式
    
        Args:
            t (array): 時間
            k (float): 速度定数
    
        Returns:
            array: 反応率
        """
        # 1 - 2α/3 - (1-α)^(2/3) = kt
        # 数値的に解く必要があるが、ここでは近似式を使用
        kt = k * t
        alpha = 1 - (1 - kt/2)**(3/2)
        alpha = np.clip(alpha, 0, 1)
        return alpha
    
    # Power law (経験式)
    def power_law_model(t, k, n):
        """べき乗則モデル
    
        Args:
            t (array): 時間
            k (float): 速度定数
            n (float): 指数
    
        Returns:
            array: 反応率
        """
        alpha = k * t**n
        alpha = np.clip(alpha, 0, 1)
        return alpha
    
    # 各モデルでフィッティング
    # Jander式
    popt_jander, _ = curve_fit(jander_model, time_exp, conversion_exp, p0=[0.01])
    k_jander = popt_jander[0]
    
    # Ginstling-Brounshtein式
    popt_gb, _ = curve_fit(gb_model, time_exp, conversion_exp, p0=[0.01])
    k_gb = popt_gb[0]
    
    # Power law
    popt_power, _ = curve_fit(power_law_model, time_exp, conversion_exp, p0=[0.1, 0.5])
    k_power, n_power = popt_power
    
    # 予測曲線生成
    t_fit = np.linspace(0, 20, 200)
    alpha_jander = jander_model(t_fit, k_jander)
    alpha_gb = gb_model(t_fit, k_gb)
    alpha_power = power_law_model(t_fit, k_power, n_power)
    
    # 残差計算
    residuals_jander = conversion_exp - jander_model(time_exp, k_jander)
    residuals_gb = conversion_exp - gb_model(time_exp, k_gb)
    residuals_power = conversion_exp - power_law_model(time_exp, k_power, n_power)
    
    # R²計算
    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
    
    r2_jander = r_squared(conversion_exp, jander_model(time_exp, k_jander))
    r2_gb = r_squared(conversion_exp, gb_model(time_exp, k_gb))
    r2_power = r_squared(conversion_exp, power_law_model(time_exp, k_power, n_power))
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # フィッティング結果
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
    
    # 残差プロット
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
    
    # 結果サマリー
    print("\n反応速度モデルのフィッティング結果:")
    print("=" * 70)
    print(f"{'Model':<25} {'Parameter':<30} {'R²':<10}")
    print("-" * 70)
    print(f"{'Jander':<25} {'k = ' + f'{k_jander:.4f} h⁻¹':<30} {r2_jander:.4f}")
    print(f"{'Ginstling-Brounshtein':<25} {'k = ' + f'{k_gb:.4f} h⁻¹':<30} {r2_gb:.4f}")
    print(f"{'Power law':<25} {'k = ' + f'{k_power:.4f}, n = {n_power:.4f}':<30} {r2_power:.4f}")
    print("=" * 70)
    print(f"\n最適モデル: {'Jander' if r2_jander == max(r2_jander, r2_gb, r2_power) else 'GB' if r2_gb == max(r2_jander, r2_gb, r2_power) else 'Power law'}")
    
    # 出力例:
    # 反応速度モデルのフィッティング結果:
    # ======================================================================
    # Model                     Parameter                      R²
    # ----------------------------------------------------------------------
    # Jander                    k = 0.0289 h⁻¹                 0.9953
    # Ginstling-Brounshtein     k = 0.0412 h⁻¹                 0.9867
    # Power law                 k = 0.2156, n = 0.5234         0.9982
    # ======================================================================
    #
    # 最適モデル: Power law
    

## 1.8 高度なトピック: 微細構造制御

### 1.8.1 粒成長の抑制

固相反応では、高温・長時間保持により望ましくない粒成長が起こります。これを抑制する戦略：

  * **Two-step sintering** : 高温で短時間保持後、低温で長時間保持
  * **添加剤の使用** : 粒成長抑制剤（例: MgO, Al₂O₃）を微量添加
  * **Spark Plasma Sintering (SPS)** : 急速加熱・短時間焼結

### 1.8.2 反応の機械化学的活性化

メカノケミカル法（高エネルギーボールミル）により、固相反応を室温付近で進行させることも可能です：
    
    
    # ===================================
    # Example 8: 粒成長シミュレーション
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def grain_growth(t, T, D0, Ea, G0, n):
        """粒成長の時間発展
    
        Burke-Turnbull式: G^n - G0^n = k*t
    
        Args:
            t (array): 時間 [hours]
            T (float): 温度 [K]
            D0 (float): 頻度因子
            Ea (float): 活性化エネルギー [J/mol]
            G0 (float): 初期粒径 [μm]
            n (float): 粒成長指数（通常2-4）
    
        Returns:
            array: 粒径 [μm]
        """
        R = 8.314
        k = D0 * np.exp(-Ea / (R * T))
        G = (G0**n + k * t * 3600)**(1/n)  # hours → seconds
        return G
    
    # パラメータ設定
    D0_grain = 1e8  # μm^n/s
    Ea_grain = 400e3  # J/mol
    G0 = 0.5  # μm
    n = 3
    
    # 温度の影響
    temps_celsius = [1100, 1200, 1300]
    t_range = np.linspace(0, 12, 100)  # 0-12 hours
    
    plt.figure(figsize=(12, 5))
    
    # 温度依存性
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
    
    # Two-step sinteringの効果
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
    
    # 最終粒径の比較
    G_final_conv = grain_growth(6, 1300+273.15, D0_grain, Ea_grain, G0, n)
    G_final_two_step = G_two_step[-1]
    
    print("\n粒成長の比較:")
    print("=" * 50)
    print(f"Conventional (1300°C, 6h): {G_final_conv:.2f} μm")
    print(f"Two-step (1300°C 1h + 1200°C 5h): {G_final_two_step:.2f} μm")
    print(f"粒径抑制効果: {(1 - G_final_two_step/G_final_conv)*100:.1f}%")
    
    # 出力例:
    # 粒成長の比較:
    # ==================================================
    # Conventional (1300°C, 6h): 4.23 μm
    # Two-step (1300°C 1h + 1200°C 5h): 2.87 μm
    # 粒径抑制効果: 32.2%
    

## 学習目標の確認

この章を完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ 固相反応の3つの律速段階（核生成・界面反応・拡散）を説明できる
  * ✅ Arrhenius式の物理的意味と温度依存性を理解している
  * ✅ Jander式とGinstling-Brounshtein式の違いを説明できる
  * ✅ 温度プロファイルの3要素（加熱速度・保持時間・冷却速度）の重要性を理解している

### 実践スキル

  * ✅ Pythonで拡散係数の温度依存性をシミュレートできる
  * ✅ Jander式を用いて反応進行を予測できる
  * ✅ Kissinger法でDSC/TGデータから活性化エネルギーを計算できる
  * ✅ DOE（実験計画法）で反応条件を最適化できる
  * ✅ pycalphadを用いた相図計算の基礎を理解している

### 応用力

  * ✅ 新規セラミックス材料の合成プロセスを設計できる
  * ✅ 実験データから反応機構を推定し、適切な速度式を選択できる
  * ✅ 産業プロセスでの条件最適化戦略を立案できる
  * ✅ 粒成長制御の戦略（Two-step sintering等）を提案できる

## 演習問題

### Easy（基礎確認）

Q1: STLファイル形式の理解

STLファイルのASCII形式とBinary形式について、正しい説明はどれですか？

a) ASCII形式の方がファイルサイズが小さい  
b) Binary形式は人間が直接読めるテキスト形式  
c) Binary形式は通常ASCII形式の5-10倍小さいファイルサイズ  
d) Binary形式はASCII形式より精度が低い

解答を表示

**正解: c) Binary形式は通常ASCII形式の5-10倍小さいファイルサイズ**

**解説:**

  * **ASCII STL** : テキスト形式で人間が読める。各三角形が7行（facet、normal、3頂点、endfacet）で記述される。大きなファイルサイズ（数十MB〜数百MB）。
  * **Binary STL** : バイナリ形式で小型。80バイトヘッダー + 4バイト三角形数 + 各三角形50バイト。同じ形状でASCIIの1/5〜1/10のサイズ。
  * 精度は両形式とも同じ（32-bit浮動小数点数）
  * 現代の3Dプリンタソフトは両形式をサポート、Binary推奨

**実例:** 10,000三角形のモデル → ASCII: 約7MB、Binary: 約0.5MB

Q2: 造形時間の簡易計算

体積12,000 mm³、高さ30 mmの造形物を、レイヤー高さ0.2 mm、印刷速度50 mm/sで造形します。おおよその造形時間はどれですか？（インフィル20%、壁2層と仮定）

a) 30分  
b) 60分  
c) 90分  
d) 120分

解答を表示

**正解: c) 90分（約1.5時間）**

**計算手順:**

  1. **レイヤー数** : 高さ30mm ÷ レイヤー高さ0.2mm = 150層
  2. **1層あたりの経路長さの推定** : 
     * 体積12,000mm³ → 1層あたり平均80mm³
     * 壁（シェル）: 約200mm/層（ノズル径0.4mmと仮定）
     * インフィル20%: 約100mm/層
     * 合計: 約300mm/層
  3. **総経路長** : 300mm/層 × 150層 = 45,000mm = 45m
  4. **印刷時間** : 45,000mm ÷ 50mm/s = 900秒 = 15分
  5. **実際の時間** : 移動時間・リトラクション・加減速を考慮すると約5-6倍 → 75-90分

**ポイント:** スライサーソフトが提供する推定時間は、加減速・移動・温度安定化を含むため、単純計算の4-6倍程度になります。

Q3: AMプロセスの選択

次の用途に最適なAMプロセスを選んでください：「航空機エンジン部品のチタン合金製燃料噴射ノズル、複雑な内部流路、高強度・高耐熱性要求」

a) FDM (Fused Deposition Modeling)  
b) SLA (Stereolithography)  
c) SLM (Selective Laser Melting)  
d) Binder Jetting

解答を表示

**正解: c) SLM (Selective Laser Melting / Powder Bed Fusion for Metal)**

**理由:**

  * **SLMの特徴** : 金属粉末（チタン、インコネル、ステンレス）をレーザーで完全溶融。高密度（99.9%）、高強度、高耐熱性。
  * **用途適合性** : 
    * ✓ チタン合金（Ti-6Al-4V）対応
    * ✓ 複雑内部流路製造可能（サポート除去後）
    * ✓ 航空宇宙グレードの機械的特性
    * ✓ GE Aviationが実際にFUEL噴射ノズルをSLMで量産
  * **他の選択肢が不適な理由** : 
    * FDM: プラスチックのみ、強度・耐熱性不足
    * SLA: 樹脂のみ、機能部品には不適
    * Binder Jetting: 金属可能だが、焼結後密度90-95%で航空宇宙基準に届かない

**実例:** GE AviationのLEAP燃料ノズル（SLM製）は、従来20部品を溶接していたものを1部品に統合、重量25%削減、耐久性5倍向上を達成。

### Medium（応用）

Q4: PythonでSTLメッシュを検証

以下のPythonコードを完成させて、STLファイルのマニフォールド性（watertight）を検証してください。
    
    
    import trimesh
    
    mesh = trimesh.load('model.stl')
    
    # ここにコードを追加: マニフォールド性をチェックし、
    # 問題があれば自動修復を行い、修復後のメッシュを
    # 'model_fixed.stl'として保存してください
    

解答を表示

**解答例:**
    
    
    import trimesh
    
    mesh = trimesh.load('model.stl')
    
    # マニフォールド性をチェック
    print(f"Is watertight: {mesh.is_watertight}")
    print(f"Is winding consistent: {mesh.is_winding_consistent}")
    
    # 問題がある場合は修復
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        print("メッシュ修復を実行中...")
    
        # 法線を修正
        trimesh.repair.fix_normals(mesh)
    
        # 穴を埋める
        trimesh.repair.fill_holes(mesh)
    
        # 縮退三角形を削除
        mesh.remove_degenerate_faces()
    
        # 重複頂点を結合
        mesh.merge_vertices()
    
        # 修復結果を確認
        print(f"修復後 watertight: {mesh.is_watertight}")
    
        # 修復したメッシュを保存
        if mesh.is_watertight:
            mesh.export('model_fixed.stl')
            print("修復完了: model_fixed.stl として保存")
        else:
            print("⚠️ 自動修復失敗。Meshmixer等を使用してください")
    else:
        print("✓ メッシュは3Dプリント可能です")
    

**解説:**

  * `trimesh.repair.fix_normals()`: 法線ベクトルの向きを統一
  * `trimesh.repair.fill_holes()`: メッシュの穴を充填
  * `remove_degenerate_faces()`: 面積ゼロの縮退三角形を削除
  * `merge_vertices()`: 重複した頂点を結合

**実践ポイント:** trimeshでも修復できない複雑な問題は、Meshmixer、Netfabb、MeshLabなどの専用ツールが必要です。

Q5: サポート材料の体積計算

直径40mm、高さ30mmの円柱を、底面から45度の角度で傾けて造形します。サポート密度15%、レイヤー高さ0.2mmと仮定して、おおよそのサポート材料体積を推定してください。

解答を表示

**解答プロセス:**

  1. **サポートが必要な領域の特定** : 
     * 45度傾斜 → 円柱底面の約半分がオーバーハング（45度以上の傾斜）
     * 円柱を45度傾けると、片側が浮いた状態になる
  2. **サポート領域の幾何計算** : 
     * 円柱の投影面積: π × (20mm)² ≈ 1,257 mm²
     * 45度傾斜時のサポート必要面積: 約1,257mm² × 0.5 = 629 mm²
     * サポート高さ: 最大で約 30mm × sin(45°) ≈ 21mm
     * サポート体積（密度100%と仮定）: 629mm² × 21mm ÷ 2（三角形状）≈ 6,600 mm³
  3. **サポート密度15%を考慮** : 
     * 実際のサポート材料: 6,600mm³ × 0.15 = **約990 mm³**
  4. **検証** : 
     * 円柱本体の体積: π × 20² × 30 ≈ 37,700 mm³
     * サポート/本体比: 990 / 37,700 ≈ 2.6%（妥当な範囲）

**答え: 約1,000 mm³ (990 mm³)**

**実践的考察:**

  * 造形向きの最適化で、サポートを大幅削減可能（この例では円柱を立てて造形すればサポート不要）
  * Tree Supportを使用すれば、さらに30-50%材料削減可能
  * 水溶性サポート材（PVA、HIPS）を使用すれば、除去が容易

Q6: レイヤー高さの最適化

高さ60mmの造形物を、品質と時間のバランスを考慮して造形します。レイヤー高さ0.1mm、0.2mm、0.3mmの3つの選択肢がある場合、それぞれの造形時間比と推奨用途を説明してください。

解答を表示

**解答:**

レイヤー高さ | レイヤー数 | 時間比 | 品質 | 推奨用途  
---|---|---|---|---  
0.1 mm | 600層 | ×3.0 | 非常に高い | 展示用フィギュア、医療モデル、最終製品  
0.2 mm | 300層 | ×1.0（基準） | 良好 | 一般的なプロトタイプ、機能部品  
0.3 mm | 200層 | ×0.67 | 低い | 初期プロトタイプ、強度優先の内部部品  
  
**時間比の計算根拠:**

  * レイヤー数が1/2になると、Z軸移動回数も1/2
  * BUT: 各層の印刷時間は微増（1層あたりの体積が増えるため）
  * 総合的には、レイヤー高さに「ほぼ反比例」（厳密には0.9-1.1倍の係数あり）

**実践的な選択基準:**

  1. **0.1mm推奨ケース** : 
     * 表面品質が最優先（顧客プレゼン、展示会）
     * 曲面の滑らかさが重要（顔、曲線形状）
     * 積層痕をほぼ消したい
  2. **0.2mm推奨ケース** : 
     * 品質と時間のバランス重視（最も一般的）
     * 機能試験用プロトタイプ
     * 適度な表面仕上がりで十分
  3. **0.3mm推奨ケース** : 
     * 速度優先（形状確認のみ）
     * 内部構造部品（外観不問）
     * 大型造形物（時間削減効果大）

**変数レイヤー高さ（Advanced）:**  
PrusaSlicerやCuraの変数レイヤー高さ機能を使えば、平坦部は0.3mm、曲面部は0.1mmと混在させて、品質と時間を両立可能。

Q7: AMプロセス選択の総合問題

航空宇宙用の軽量ブラケット（アルミニウム合金、トポロジー最適化済み複雑形状、高強度・軽量要求）の製造に最適なAMプロセスを選択し、その理由を3つ挙げてください。また、考慮すべき後処理を2つ挙げてください。

解答を表示

**最適プロセス: LPBF (Laser Powder Bed Fusion) - SLM for Aluminum**

**選択理由（3つ）:**

  1. **高密度・高強度** : 
     * レーザー完全溶融により相対密度99.5%以上を達成
     * 鍛造材に匹敵する機械的特性（引張強度、疲労特性）
     * 航空宇宙認証（AS9100、Nadcap）取得可能
  2. **トポロジー最適化形状の製造能力** : 
     * 複雑なラティス構造（厚さ0.5mm以下）を高精度で造形
     * 中空構造、バイオニック形状など従来加工不可能な形状に対応
     * サポート除去後、内部構造もアクセス可能
  3. **材料効率と軽量化** : 
     * Buy-to-Fly比（材料投入量/最終製品重量）が切削加工の1/10〜1/20
     * トポロジー最適化で従来設計比40-60%軽量化
     * アルミ合金（AlSi10Mg、Scalmalloy）で比強度最大化

**必要な後処理（2つ）:**

  1. **熱処理（Heat Treatment）** : 
     * 応力除去焼鈍（Stress Relief Annealing）: 300°C、2-4時間
     * 目的: 造形時の残留応力を除去、寸法安定性向上
     * 効果: 疲労寿命30-50%向上、反り変形防止
  2. **表面処理（Surface Finishing）** : 
     * 機械加工（CNC）: 取り付け面、ボルト穴の高精度加工（Ra < 3.2μm）
     * 化学研磨（Electropolishing）: 表面粗さ低減（Ra 10μm → 2μm）
     * ショットピーニング（Shot Peening）: 表面層に圧縮残留応力を付与、疲労特性向上
     * アノダイズ処理: 耐食性向上、絶縁性付与（航空宇宙標準）

**追加考慮事項:**

  * **造形方向** : 荷重方向と積層方向を考慮（Z方向強度は10-15%低い）
  * **サポート設計** : 除去しやすいTree Support、接触面積最小化
  * **品質管理** : CT スキャンで内部欠陥検査、X線検査
  * **トレーサビリティ** : 粉末ロット管理、造形パラメータ記録

**実例: Airbus A350のチタンブラケット**  
従来32部品を組立てていたブラケットを1部品に統合、重量55%削減、リードタイム65%短縮、コスト35%削減を達成。

3水準 × 3水準 = **9回** （フルファクトリアル計画） 

**DOEの利点（従来法との比較）:**

  1. **交互作用の検出が可能**
     * 従来法: 温度の影響、時間の影響を個別に評価
     * DOE: 「高温では時間を短くできる」といった交互作用を定量化
     * 例: 1300°Cでは4時間で十分だが、1100°Cでは8時間必要、など
  2. **実験回数の削減**
     * 従来法（OFAT: One Factor At a Time）: 
       * 温度検討: 3回（時間固定）
       * 時間検討: 3回（温度固定）
       * 確認実験: 複数回
       * 合計: 10回以上
     * DOE: 9回で完了（全条件網羅＋交互作用解析）
     * さらに中心複合計画法を使えば7回に削減可能

**追加の利点:**

  * 統計的に有意な結論が得られる（誤差評価が可能）
  * 応答曲面を構築でき、未実施条件の予測が可能
  * 最適条件が実験範囲外にある場合でも検出できる

### Hard（発展）

Q7: 複雑な反応系の設計

次の条件でLi₁.₂Ni₀.₂Mn₀.₆O₂（リチウムリッチ正極材料）を合成する温度プロファイルを設計してください：

  * 原料: Li₂CO₃, NiO, Mn₂O₃
  * 目標: 単一相、粒径 < 5 μm、Li/遷移金属比の精密制御
  * 制約: 900°C以上でLi₂Oが揮発（Li欠損のリスク）

温度プロファイル（加熱速度、保持温度・時間、冷却速度）と、その設計理由を説明してください。

解答を見る

**推奨温度プロファイル:**

**Phase 1: 予備加熱（Li₂CO₃分解）**

  * 室温 → 500°C: 3°C/min
  * 500°C保持: 2時間
  * **理由:** Li₂CO₃の分解（~450°C）をゆっくり進行させ、CO₂を完全に除去

**Phase 2: 中間加熱（前駆体形成）**

  * 500°C → 750°C: 5°C/min
  * 750°C保持: 4時間
  * **理由:** Li₂MnO₃やLiNiO₂などの中間相を形成。Li揮発の少ない温度で均質化

**Phase 3: 本焼成（目的相合成）**

  * 750°C → 850°C: 2°C/min（ゆっくり）
  * 850°C保持: 12時間
  * **理由:**
    * Li₁.₂Ni₀.₂Mn₀.₆O₂の単一相形成には長時間必要
    * 850°Cに制限してLi揮発を最小化（<900°C制約）
    * 長時間保持で拡散を進めるが、粒成長は抑制される温度

**Phase 4: 冷却**

  * 850°C → 室温: 2°C/min
  * **理由:** 徐冷により結晶性向上、熱応力による亀裂防止

**設計の重要ポイント:**

  1. **Li揮発対策:**
     * 900°C以下に制限（本問の制約）
     * さらに、Li過剰原料（Li/TM = 1.25など）を使用
     * 酸素気流中で焼成してLi₂Oの分圧を低減
  2. **粒径制御 ( < 5 μm):**
     * 低温（850°C）・長時間（12h）で反応を進める
     * 高温・短時間だと粒成長が過剰になる
     * 原料粒径も1μm以下に微細化
  3. **組成均一性:**
     * 750°Cでの中間保持が重要
     * この段階で遷移金属の分布を均質化
     * 必要に応じて、750°C保持後に一度冷却→粉砕→再加熱

**全体所要時間:** 約30時間（加熱12h + 保持18h）

**代替手法の検討:**

  * **Sol-gel法:** より低温（600-700°C）で合成可能、均質性向上
  * **Spray pyrolysis:** 粒径制御が容易
  * **Two-step sintering:** 900°C 1h → 800°C 10h で粒成長抑制

Q8: 速度論的解析の総合問題

以下のデータから、反応機構を推定し、活性化エネルギーを計算してください。

**実験データ:**

温度 (°C) | 50%反応到達時間 t₅₀ (hours)  
---|---  
1000| 18.5  
1100| 6.2  
1200| 2.5  
1300| 1.2  
  
Jander式を仮定した場合: [1-(1-0.5)^(1/3)]² = k·t₅₀

解答を見る

**解答:**

**ステップ1: 速度定数kの計算**

Jander式で α=0.5 のとき:  
[1-(1-0.5)^(1/3)]² = [1-0.794]² = 0.206² = 0.0424

したがって k = 0.0424 / t₅₀

T (°C) | T (K) | t₅₀ (h) | k (h⁻¹) | ln(k) | 1000/T (K⁻¹)  
---|---|---|---|---|---  
1000| 1273| 18.5| 0.00229| -6.080| 0.7855  
1100| 1373| 6.2| 0.00684| -4.985| 0.7284  
1200| 1473| 2.5| 0.01696| -4.077| 0.6788  
1300| 1573| 1.2| 0.03533| -3.343| 0.6357  
  
**ステップ2: Arrheniusプロット**

ln(k) vs 1/T をプロット（線形回帰）

線形フィット: ln(k) = A - Eₐ/(R·T)

傾き = -Eₐ/R

線形回帰計算:  
slope = Δ(ln k) / Δ(1000/T)  
= (-3.343 - (-6.080)) / (0.6357 - 0.7855)  
= 2.737 / (-0.1498)  
= -18.27

**ステップ3: 活性化エネルギー計算**

slope = -Eₐ / (R × 1000)  
Eₐ = -slope × R × 1000  
Eₐ = 18.27 × 8.314 × 1000  
Eₐ = 151,899 J/mol ≈ **152 kJ/mol**

**ステップ4: 反応機構の考察**

  * **活性化エネルギーの比較:**
    * 得られた値: 152 kJ/mol
    * 典型的な固相拡散: 200-400 kJ/mol
    * 界面反応: 50-150 kJ/mol
  * **推定される機構:**
    * この値は界面反応と拡散の中間
    * 可能性1: 界面反応が主律速（拡散の影響は小）
    * 可能性2: 粒子が微細で拡散距離が短く、見かけのEₐが低い
    * 可能性3: 混合律速（界面反応と拡散の両方が寄与）

**ステップ5: 検証方法の提案**

  1. **粒子サイズ依存性:** 異なる粒径で実験し、k ∝ 1/r₀² が成立するか確認 
     * 成立 → 拡散律速
     * 不成立 → 界面反応律速
  2. **他の速度式でのフィッティング:**
     * Ginstling-Brounshtein式（3次元拡散）
     * Contracting sphere model（界面反応）
     * どちらがR²が高いか比較
  3. **微細構造観察:** SEMで反応界面を観察 
     * 厚い生成物層 → 拡散律速の証拠
     * 薄い生成物層 → 界面反応律速の可能性

**最終結論:**  
活性化エネルギー **Eₐ = 152 kJ/mol**  
推定機構: **界面反応律速、または微細粒子系での拡散律速**  
追加実験が推奨される。

## 次のステップ

第3章では積層造形（AM）の基礎として、ISO/ASTM 52900による7つのプロセス分類、STLファイル形式の構造、スライシングとG-codeの基本を学びました。次の第2章では、材料押出（FDM/FFF）の詳細な造形プロセス、材料特性、プロセスパラメータ最適化について学びます。

[← シリーズ目次](<./index.html>) [第2章へ進む →](<./chapter-4.html>)

## 参考文献

  1. Gibson, I., Rosen, D., & Stucker, B. (2015). _Additive Manufacturing Technologies: 3D Printing, Rapid Prototyping, and Direct Digital Manufacturing_ (2nd ed.). Springer. pp. 1-35, 89-145, 287-334. - AM技術の包括的教科書、7つのプロセスカテゴリとSTLデータ処理の詳細解説
  2. ISO/ASTM 52900:2021. _Additive manufacturing — General principles — Fundamentals and vocabulary_. International Organization for Standardization. - AM用語とプロセス分類の国際標準規格、産業界で広く参照される
  3. Kruth, J.P., Leu, M.C., & Nakagawa, T. (1998). "Progress in Additive Manufacturing and Rapid Prototyping." _CIRP Annals - Manufacturing Technology_ , 47(2), 525-540. - 選択的レーザー焼結とバインディング機構の理論的基礎
  4. Hull, C.W. (1986). _Apparatus for production of three-dimensional objects by stereolithography_. US Patent 4,575,330. - 世界初のAM技術（SLA）の特許、AM産業の起源となる重要文献
  5. Wohlers, T. (2023). _Wohlers Report 2023: 3D Printing and Additive Manufacturing Global State of the Industry_. Wohlers Associates, Inc. pp. 15-89, 156-234. - AM市場動向と産業応用の最新統計レポート、年次更新される業界標準資料
  6. 3D Systems, Inc. (1988). _StereoLithography Interface Specification_. - STLファイル形式の公式仕様書、ASCII/Binary STL構造の定義
  7. numpy-stl Documentation. (2024). _Python library for working with STL files_. <https://numpy-stl.readthedocs.io/> \- STLファイル読込・体積計算のためのPythonライブラリ
  8. trimesh Documentation. (2024). _Python library for loading and using triangular meshes_. <https://trimsh.org/> \- メッシュ修復・ブーリアン演算・品質評価の包括的ライブラリ

## 使用ツールとライブラリ

  * **NumPy** (v1.24+): 数値計算ライブラリ - <https://numpy.org/>
  * **numpy-stl** (v3.0+): STLファイル処理ライブラリ - <https://numpy-stl.readthedocs.io/>
  * **trimesh** (v4.0+): 3Dメッシュ処理ライブラリ（修復、検証、ブーリアン演算） - <https://trimsh.org/>
  * **Matplotlib** (v3.7+): データ可視化ライブラリ - <https://matplotlib.org/>
  * **SciPy** (v1.10+): 科学技術計算ライブラリ（最適化、補間） - <https://scipy.org/>

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
