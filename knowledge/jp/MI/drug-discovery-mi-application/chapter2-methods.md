---
title: 創薬に特化したMI手法
chapter_title: 創薬に特化したMI手法
subtitle: 分子記述子から生成モデルまで
---

# 第2章：創薬に特化したMI手法

**分子表現から生成モデルまで - AI創薬の技術基盤**

## 2.1 分子表現と記述子

創薬MIの第一歩は、化学構造をコンピュータが理解できる形式に変換することです。この「分子表現」の選択が、モデルの性能を大きく左右します。

### 2.1.1 SMILES表現

**SMILES（Simplified Molecular Input Line Entry System）** は、分子構造を文字列で表現する最も普及した形式です。

**基本ルール:**
    
    
    原子: C（炭素）、N（窒素）、O（酸素）、S（硫黄）等
    結合: 単結合（省略可）、二重結合（=）、三重結合（#）
    環構造: 数字でマーキング（例: C1CCCCC1 = シクロヘキサン）
    分岐: 括弧で表現（例: CC(C)C = イソブタン）
    芳香族: 小文字（例: c1ccccc1 = ベンゼン）
    

**実例:**

化合物 | SMILES | 構造の特徴  
---|---|---  
エタノール | `CCO` | 2炭素 + ヒドロキシ基  
アスピリン | `CC(=O)OC1=CC=CC=C1C(=O)O` | エステル + カルボン酸  
カフェイン | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | プリン骨格 + メチル基  
イブプロフェン | `CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O` | 芳香環 + キラル中心  
ペニシリンG | `CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O` | β-ラクタム環 + チアゾリジン環  
  
**利点:** \- ✅ コンパクト（数十文字で複雑な分子を表現） \- ✅ 人間が読み書き可能 \- ✅ データベース検索に適している \- ✅ 一意性（Canonical SMILES）

**欠点:** \- ❌ 3D構造情報なし（立体配座が失われる） \- ❌ 互変異性体の区別困難 \- ❌ 無効なSMILESの存在（構文エラー）

### 2.1.2 分子指紋（Molecular Fingerprints）

分子指紋は、分子をビットベクトル（0/1の配列）で表現する手法です。類似性検索やQSARに広く使われます。

#### ECFP（Extended Connectivity Fingerprints）

**原理:** 1\. 各原子から開始 2\. 半径R（通常2-4）の近傍構造をハッシュ化 3\. ビットベクトル（長さ1024-4096）に変換

**例: ECFP4（半径2、4ステップの近傍）**
    
    
    分子: CCO（エタノール）
    
    原子1（C）:
      - 半径0: C
      - 半径1: C-C
      - 半径2: C-C-O
    
    原子2（C）:
      - 半径0: C
      - 半径1: C-C, C-O
      - 半径2: C-C-O, C-O
    
    原子3（O）:
      - 半径0: O
      - 半径1: O-C
      - 半径2: O-C-C
    
    → これらをハッシュ化してビット位置を決定
    → 該当ビットを1にセット
    

**種類:**

指紋 | サイズ | 特徴 | 用途  
---|---|---|---  
ECFP4 | 1024-4096 bit | 半径2の環境 | 類似性検索、QSAR  
ECFP6 | 1024-4096 bit | 半径3の環境 | より精密な構造認識  
MACCS keys | 166 bit | 固定の部分構造 | 高速検索、多様性分析  
RDKit Fingerprint | 2048 bit | パス（経路）ベース | 汎用的なQSAR  
Morgan Fingerprint | 可変 | ECFPの実装 | RDKitでの標準  
  
**Tanimoto係数による類似度:**
    
    
    Tanimoto係数 = (A ∩ B) / (A ∪ B)
    
    A, B: 2つの分子の指紋ビットベクトル
    ∩: ビットAND（両方1のビット数）
    ∪: ビットOR（少なくとも片方が1のビット数）
    
    範囲: 0（完全に異なる）〜 1（完全一致）
    

**実用的な閾値:** \- Tanimoto > 0.85: 非常に類似（同じ化合物クラス） \- 0.70-0.85: 類似（類似の薬理活性の可能性） \- 0.50-0.70: やや類似 \- < 0.50: 異なる

#### MACCS keys

**特徴:** \- 166個の固定された部分構造の有無を表現 \- 例: ベンゼン環、カルボン酸、アミノ基、ハロゲン等

**利点:** \- 解釈可能（どの部分構造があるかわかる） \- 高速計算 \- 化学的に意味のある類似性

**欠点:** \- 情報量が少ない（166ビットのみ） \- 新規骨格には対応しにくい

### 2.1.3 3D記述子

3D記述子は、分子の立体構造を数値化します。

**主要な3D記述子:**

  1. **分子表面積（Molecular Surface Area）** \- TPSA（Topological Polar Surface Area）: 極性表面積 \- 予測: 経口吸収性（TPSA < 140 Å²で良好）

  2. **体積と形状** \- 分子体積（Molecular Volume） \- 球形度（Sphericity） \- アスペクト比（Aspect Ratio）

  3. **電荷分布** \- 部分電荷（Partial Charges）: Gasteiger, MMFF \- 双極子モーメント（Dipole Moment） \- 四重極モーメント（Quadrupole Moment）

  4. **薬理活性座標（Pharmacophore）** \- 水素結合ドナー/アクセプター位置 \- 疎水性領域 \- 正/負電荷中心 \- 芳香環の向き

**例: Lipinski's Rule of Fiveで使用される記述子**
    
    
    分子量（MW）: < 500 Da
    LogP（脂溶性）: < 5
    水素結合ドナー（HBD）: < 5
    水素結合アクセプター（HBA）: < 10
    
    これらを満たす化合物は経口吸収性が高い傾向
    

### 2.1.4 Graph表現（グラフニューラルネットワーク用）

分子を数学的なグラフ（Graph）として表現します。

**定義:**
    
    
    G = (V, E)
    
    V: 頂点（Vertices）= 原子
    E: 辺（Edges）= 結合
    
    各頂点vには特徴ベクトル h_v
    各辺eには特徴ベクトル h_e
    

**原子（頂点）の特徴:** \- 原子番号（C=6, N=7, O=8, etc.） \- 原子タイプ（C, N, O, S, F, Cl, Br, I） \- ハイブリダイゼーション（sp, sp2, sp3） \- 形式電荷（Formal Charge） \- 芳香族性（Aromatic or not） \- 水素数（Hydrogen Count） \- 孤立電子対数 \- キラリティ（R/S）

**結合（辺）の特徴:** \- 結合次数（Single=1, Double=2, Triple=3, Aromatic=1.5） \- 結合タイプ（Covalent, Ionic, etc.） \- 環の一部かどうか \- 立体配置（E/Z）

**グラフの利点:** \- 分子の構造を直接表現 \- 回転・並進不変性 \- Graph Neural Networksで学習可能

* * *

## 2.2 QSAR（定量的構造活性相関）

### 2.2.1 QSARの基本原理

**QSAR（Quantitative Structure-Activity Relationship）** は、分子構造と生物活性の定量的関係を数式で表現します。

**基本仮説:**

> 類似の分子構造は類似の生物活性を示す（Similar Property Principle）

**QSARの一般式:**
    
    
    Activity = f(Descriptors)
    
    Activity: 生物活性（IC50, EC50, Ki等）
    Descriptors: 分子記述子（MW, LogP, TPSA等）
    f: 数学的関数（線形回帰、Random Forest、NN等）
    

**歴史的なQSAR式（Hansch-Fujita式, 1962）:**
    
    
    log(1/C) = a * logP + b * σ + c * Es + d
    
    C: 生物活性濃度（低いほど活性が高い）
    logP: 分配係数（脂溶性）
    σ: Hammett定数（電子効果）
    Es: 立体パラメータ
    a, b, c, d: 回帰係数
    

### 2.2.2 QSARワークフロー
    
    
    ```mermaid
    flowchart TD
        A[化合物ライブラリ\nSMILES + 活性データ] --> B[分子記述子計算\nMW, LogP, ECFP等]
        B --> C[データ分割\nTrain 80% / Test 20%]
        C --> D[モデル訓練\nRF, SVM, NN]
        D --> E[性能評価\nR^2, MAE, ROC-AUC]
        E --> F{性能OK?}
        F -->|No| G[ハイパーパラメータ\nチューニング]
        G --> D
        F -->|Yes| H[新規化合物予測\nVirtual Screening]
        H --> I[実験検証\nTop候補のみ]
    
        style A fill:#e3f2fd
        style E fill:#fff3e0
        style H fill:#e8f5e9
        style I fill:#ffebee
    ```

### 2.2.3 QSARモデルの種類

#### 分類モデル（活性/非活性の予測）

**目的:** 化合物が活性（Active）か非活性（Inactive）かを予測

**評価指標:**
    
    
    ROC-AUC: 0.5（ランダム）〜 1.0（完璧）
    目標: > 0.80
    
    Precision（精度） = TP / (TP + FP)
    Recall（再現率） = TP / (TP + FN)
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    
    TP: True Positive（活性を正しく予測）
    FP: False Positive（非活性を活性と誤予測）
    FN: False Negative（活性を非活性と誤予測）
    

**典型的な問題設定:** \- IC50 < 1 μM → Active (1) \- IC50 ≥ 1 μM → Inactive (0)

#### 回帰モデル（活性値の予測）

**目的:** IC50、Ki、EC50等の連続値を予測

**評価指標:**
    
    
    R²（決定係数）: 0（無相関）〜 1（完璧）
    目標: > 0.70
    
    MAE（平均絶対誤差） = Σ|y_pred - y_true| / n
    RMSE（二乗平均平方根誤差） = √(Σ(y_pred - y_true)² / n)
    
    y_pred: 予測値
    y_true: 実測値
    n: サンプル数
    

**対数変換:** 多くの場合、活性値（IC50等）は対数変換してから回帰
    
    
    pIC50 = -log10(IC50[M])
    
    例:
    IC50 = 1 nM = 10^-9 M → pIC50 = 9.0
    IC50 = 100 nM = 10^-7 M → pIC50 = 7.0
    
    範囲: 通常4-10（10 μM 〜 0.1 nM）
    

### 2.2.4 機械学習手法の比較

手法 | 精度 | 速度 | 解釈性 | データ量 | 推奨ケース  
---|---|---|---|---|---  
Linear Regression | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 100+ | ベースライン、線形関係  
Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 1K-100K | 中規模データ、非線形  
SVM (RBF kernel) | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 1K-10K | 小〜中規模、高次元  
Neural Network | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | 10K+ | 大規模データ、複雑関係  
LightGBM/XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 1K-1M | Kaggle優勝、高速  
Graph Neural Network | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 10K+ | 分子グラフ、SOTA  
  
**実世界での性能（ChEMBLデータ）:**
    
    
    タスク: キナーゼ阻害活性予測（分類）
    
    Linear Regression: ROC-AUC = 0.72
    Random Forest:      ROC-AUC = 0.87
    SVM (RBF):          ROC-AUC = 0.85
    Neural Network:     ROC-AUC = 0.89
    LightGBM:           ROC-AUC = 0.90
    GNN (MPNN):         ROC-AUC = 0.92
    
    訓練データ: 10,000化合物
    テストデータ: 2,000化合物
    

### 2.2.5 QSARの適用限界（Applicability Domain）

**問題:** 訓練データと大きく異なる化合物の予測は信頼できない

**Applicability Domainの定義:**
    
    
    AD = {x | 訓練データの化学空間に含まれる}
    
    判定方法:
    1. Tanimoto距離: 最近傍化合物との類似度 > 0.3
    2. レバレッジ: h_i < 3p/n （h: hat matrix, p: 特徴数, n: サンプル数）
    3. 標準化距離: d_i < 3σ （σ: 標準偏差）
    

**AD外の化合物:** \- 予測の信頼性が低い \- 実験検証必須 \- モデルの再訓練を検討

**例:**
    
    
    訓練データ: キナーゼ阻害剤（主にATP競合型）
    新規化合物: アロステリック阻害剤（結合サイトが異なる）
    
    → AD外の可能性が高い
    → 予測精度は保証されない
    

* * *

## 2.3 ADMET予測

### 2.3.1 ADMETの重要性

**臨床試験失敗の30%はADMET問題**

Phase | 主な失敗原因 | ADMET関連  
---|---|---  
Phase I | 毒性（肝毒性、心毒性） | 30%  
Phase II | 有効性不足、PK不良 | 20%  
Phase III | 長期毒性、副作用 | 15%  
  
**ADMET予測による失敗削減:** \- 早期（リード発見段階）でADMET評価 \- 問題のある化合物を排除 \- 開発コスト$100M-500M削減

### 2.3.2 吸収（Absorption）

#### Caco-2透過性

**Caco-2細胞:** 大腸がん細胞株、腸上皮細胞のモデル

**測定:** 細胞層を透過する速度（Papp: 見かけの透過係数）
    
    
    Papp [cm/s]
    
    Papp > 10^-6 cm/s: 高透過性（良好な吸収）
    10^-7 < Papp < 10^-6: 中程度
    Papp < 10^-7 cm/s: 低透過性（吸収不良）
    

**予測モデル:**
    
    
    # 簡易予測式（R² = 0.75）
    log Papp = 0.5 * logP - 0.01 * TPSA + 0.3 * HBD - 2.5
    
    logP: 分配係数（脂溶性）
    TPSA: 極性表面積
    HBD: 水素結合ドナー数
    

**機械学習予測精度:** \- Random Forest: R² = 0.80-0.85 \- Neural Network: R² = 0.82-0.88 \- Graph NN: R² = 0.85-0.90

#### 経口バイオアベイラビリティ（F%）

**定義:** 投与量のうち全身循環に到達する割合
    
    
    F% = (AUC_oral / AUC_iv) * (Dose_iv / Dose_oral) * 100
    
    AUC: 血中濃度-時間曲線下面積
    

**目標値:** \- F% > 30%: 許容範囲 \- F% > 50%: 良好 \- F% > 70%: 優秀

**予測因子:** 1\. 溶解度（Solubility） 2\. 透過性（Permeability） 3\. First-pass代謝（肝臓での初回通過代謝） 4\. P-糖タンパク質による排出

**BCS分類（Biopharmaceutics Classification System）:**

Class | 溶解度 | 透過性 | 例 | F%  
---|---|---|---|---  
I | 高 | 高 | メトプロロール | > 80%  
II | 低 | 高 | イブプロフェン | 50-80%  
III | 高 | 低 | アテノロール | 30-50%  
IV | 低 | 低 | タクロリムス | < 30%  
  
### 2.3.3 分布（Distribution）

#### 血漿タンパク結合率

**定義:** 薬物が血漿タンパク質（アルブミン、α1-酸性糖タンパク質）に結合する割合
    
    
    結合率 = (Bound / Total) * 100%
    
    Bound: 結合型薬物濃度
    Total: 総薬物濃度
    

**臨床的意義:** \- 高結合率（> 90%）: 遊離型薬物（活性体）が少ない \- 薬物間相互作用のリスク（結合サイト競合） \- 分布容積に影響

**予測モデル:** \- Random Forest: R² = 0.65-0.75 \- Deep Learning: R² = 0.70-0.80

#### 脳血液関門（BBB）透過性

**LogBB（Brain/Blood比）:**
    
    
    LogBB = log10(C_brain / C_blood)
    
    LogBB > 0: 脳に濃縮（CNS薬に好ましい）
    LogBB < -1: 脳に透過しない（CNS副作用回避）
    

**予測因子:** \- 分子量（< 400 Da が好ましい） \- TPSA（< 60 Å²が好ましい） \- LogP（2-5が最適） \- 水素結合数（少ないほど良い）

**機械学習予測:** \- Random Forest: R² = 0.70-0.80 \- Neural Network: R² = 0.75-0.85

### 2.3.4 代謝（Metabolism）

#### CYP450阻害

**CYP450（シトクロムP450）** : 肝臓の主要な代謝酵素

**主要アイソフォーム:**

CYP | 基質薬物の割合 | 例  
---|---|---  
CYP3A4 | 50% | スタチン、免疫抑制剤  
CYP2D6 | 25% | β遮断薬、抗うつ薬  
CYP2C9 | 15% | ワルファリン、NSAIDs  
CYP2C19 | 10% | プロトンポンプ阻害薬  
  
**阻害の問題:** \- 薬物間相互作用（DDI: Drug-Drug Interaction） \- 併用薬の血中濃度上昇 → 毒性リスク

**予測（分類問題）:**
    
    
    阻害剤: IC50 < 10 μM
    非阻害剤: IC50 ≥ 10 μM
    
    予測精度:
    Random Forest: ROC-AUC = 0.85-0.90
    Neural Network: ROC-AUC = 0.87-0.92
    

#### 代謝安定性

**測定:** 肝ミクロソームとのインキュベーション後の残存率
    
    
    t1/2 (半減期) [min]
    
    t1/2 > 60 min: 安定（代謝が遅い）
    30 < t1/2 < 60: 中程度
    t1/2 < 30 min: 不安定（代謝が速い）
    

**予測:** \- Random Forest: R² = 0.55-0.65（やや困難） \- Graph NN: R² = 0.60-0.70

### 2.3.5 排泄（Excretion）

#### 腎クリアランス

**定義:** 腎臓による薬物除去速度
    
    
    CL_renal [mL/min/kg]
    
    CL_renal > 10: 高クリアランス（速やかに排泄）
    1 < CL_renal < 10: 中程度
    CL_renal < 1: 低クリアランス
    

**影響因子:** \- 分子量（小さいほど排泄されやすい） \- 極性（高いほど排泄されやすい） \- 腎輸送体の基質性

#### 半減期（t1/2）

**定義:** 血中濃度が半分になるまでの時間
    
    
    t1/2 = 0.693 / (CL / Vd)
    
    CL: クリアランス（全身）
    Vd: 分布容積
    

**臨床的意義:** \- t1/2 < 2 h: 頻回投与必要（1日3-4回） \- 2 < t1/2 < 8 h: 1日2回投与 \- t1/2 > 8 h: 1日1回投与可能

**予測精度:** \- Random Forest: R² = 0.55-0.65（困難） \- 複雑な薬物動態パラメータの組み合わせ

### 2.3.6 毒性（Toxicity）

#### hERG阻害（心毒性）

**hERG（human Ether-à-go-go-Related Gene）:** 心臓カリウムチャネル

**阻害の結果:** QT延長 → 致死性不整脈（Torsades de pointes）

**リスク評価:**
    
    
    IC50 < 1 μM: 高リスク（開発中止）
    1 < IC50 < 10 μM: 中リスク（慎重な評価必要）
    IC50 > 10 μM: 低リスク（安全）
    

**予測精度（最も精度が高いADMET予測）:** \- Random Forest: ROC-AUC = 0.85-0.90 \- Deep Learning: ROC-AUC = 0.90-0.95 \- Graph NN: ROC-AUC = 0.92-0.97

**構造アラート（hERG阻害しやすい構造）:** \- 塩基性窒素原子（pKa > 7） \- 疎水性芳香環 \- 柔軟なリンカー

#### 肝毒性

**DILI（Drug-Induced Liver Injury）:** 薬物誘発性肝障害

**メカニズム:** 1\. 反応性代謝物の生成 2\. ミトコンドリア毒性 3\. 胆汁酸輸送阻害

**予測（分類問題）:**
    
    
    肝毒性薬物の判定:
    
    Random Forest: ROC-AUC = 0.75-0.85
    Neural Network: ROC-AUC = 0.78-0.88
    
    課題: データ不足（承認後に判明するケースも多い）
    

#### 変異原性（Ames test）

**Ames test:** 細菌を使った変異原性試験

**予測:**
    
    
    陽性: 変異原性あり（発がん性リスク）
    陰性: 変異原性なし
    
    Random Forest: ROC-AUC = 0.80-0.90
    Deep Learning: ROC-AUC = 0.85-0.93
    

**構造アラート:** \- ニトロ基（-NO2） \- アゾ基（-N=N-） \- エポキシド \- アルキル化剤

* * *

## 2.4 分子生成モデル

### 2.4.1 生成モデルの必要性

**従来のVirtual Screening:** \- 既存の化合物ライブラリ（10^6-10^9化合物）から選択 \- 制約: ライブラリに含まれない化合物は発見できない

**生成モデルのアプローチ:** \- 新規分子を直接生成 \- 化学空間（10^60分子）を自由に探索 \- 所望の特性を持つ分子を設計

**生成モデルの種類:**
    
    
    ```mermaid
    flowchart TD
        A[分子生成モデル] --> B[VAE\n変分オートエンコーダ]
        A --> C[GAN\n敵対的生成ネットワーク]
        A --> D[Transformer\nGPT-like]
        A --> E[強化学習\nRL]
        A --> F[Graph生成\nGNN-based]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fff9c4
        style F fill:#fce4ec
    ```

### 2.4.2 VAE（Variational Autoencoder）

**アーキテクチャ:**
    
    
    SMILES → Encoder → 潜在空間（z） → Decoder → SMILES'
    
    Encoder: SMILES文字列を低次元ベクトル（z）に圧縮
    潜在空間: 連続的な化学空間（通常50-500次元）
    Decoder: ベクトル（z）からSMILES文字列を復元
    

**学習目標:**
    
    
    Loss = Reconstruction Loss + KL Divergence
    
    Reconstruction Loss: 入力SMILESの再構築誤差
    KL Divergence: 潜在空間の正則化（正規分布に近づける）
    

**分子生成のワークフロー:** 1\. 既知分子（訓練データ）でVAEを訓練 2\. 潜在空間でサンプリング（z ~ N(0, I)） 3\. Decoderで新規SMILESを生成 4\. 有効性チェック（RDKitでパース可能か） 5\. 特性予測（QSARモデル、ADMETモデル） 6\. 良好な分子を選択

**利点:** \- ✅ 連続的な潜在空間での最適化が可能 \- ✅ 既知分子の周辺を探索（類似分子生成） \- ✅ 補間（Interpolation）で中間的な分子を生成

**欠点:** \- ❌ 無効なSMILESを生成する割合が高い（30-50%） \- ❌ 新規骨格の生成は困難（訓練データに依存）

**実例: ChemVAE（Gómez-Bombarelli et al., 2018）**
    
    
    訓練データ: ZINC 250K化合物
    潜在空間: 196次元
    有効SMILES生成率: 68%
    応用: ペナルティ法で特性最適化（LogP, QED）
    

### 2.4.3 GAN（Generative Adversarial Network）

**2つのネットワーク:**
    
    
    Generator（生成器）: ノイズ → 分子SMILES
    Discriminator（識別器）: SMILES → Real or Fake?
    
    敵対的学習:
    - Generator: Discriminatorを騙す分子を生成
    - Discriminator: Real（訓練データ）とFake（生成）を見分ける
    

**学習プロセス:**
    
    
    1. Generator: ランダムノイズ → SMILES生成
    2. Discriminator: Real/Fake判定
    3. Generator: Discriminatorの勾配で更新（Fakeを"Realっぽく"）
    4. Discriminator: 判定精度向上
    5. 繰り返し → Generatorが訓練データに類似した分子を生成
    

**利点:** \- ✅ 多様な分子生成（Mode collapse対策が必要） \- ✅ 高品質な分子（Discriminatorによるフィルタリング効果）

**欠点:** \- ❌ 訓練が不安定（Mode collapse, Gradient vanishing） \- ❌ 評価が難しい（生成品質の定量化）

**実例: ORGANIC（Guimaraes et al., 2017）**
    
    
    Generator: LSTM（SMILESを逐次生成）
    Discriminator: CNN（SMILES文字列を分類）
    応用: 強化学習と組み合わせて特性最適化
    

**Insilico MedicineのChemistry42:** \- GANベース \- 特発性肺線維症（IPF）治療薬候補を18ヶ月で発見 \- 技術: WGAN-GP（Wasserstein GAN with Gradient Penalty）

### 2.4.4 Transformer（GPT-like Models）

**原理:** \- SMILESを自然言語のように扱う \- Attention機構で文脈を捉える \- 大規模事前学習（10^6-10^7分子）

**アーキテクチャ:**
    
    
    SMILES: C C ( = O ) O [EOS]
    ↓
    Tokenization: [C] [C] [(] [=] [O] [)] [O] [EOS]
    ↓
    Transformer Encoder/Decoder
    ↓
    次トークン予測: P(token_i+1 | token_1, ..., token_i)
    

**生成方法:** 1\. 開始トークン（[SOS]）から開始 2\. 次のトークンを確率的にサンプリング 3\. [EOS]まで繰り返し 4\. SMILESとして妥当性チェック

**利点:** \- ✅ 高い有効SMILES生成率（> 90%） \- ✅ 大規模事前学習の恩恵（転移学習） \- ✅ 制御可能な生成（条件付き生成）

**欠点:** \- ❌ 計算コスト（大規模モデル） \- ❌ 新規性の制御が難しい

**実例:** 1\. **ChemBERTa（HuggingFace, 2020）** \- 事前学習: 1000万SMILES（PubChem） \- Fine-tuning: 100サンプルで高精度QSAR

  2. **MolGPT（Bagal et al., 2021）** \- GPT-2アーキテクチャ \- 条件付き生成（特性値を指定）

  3. **SMILES-BERT（Wang et al., 2019）** \- Masked Language Model（MLMタスク） \- 転移学習で少量データでも高精度

### 2.4.5 強化学習（Reinforcement Learning）

**設定:**
    
    
    Agent: 分子生成モデル
    Environment: 化学空間
    Action: 次のトークン選択（SMILES生成）
    State: 現在のSMILES prefix
    Reward: 生成分子の特性（QED, LogP, ADMET等）
    

**RL + 生成モデルのフロー:**
    
    
    ```mermaid
    flowchart LR
        A[Agent\n生成モデル] -->|Action| B[SMILES生成]
        B -->|State| C[特性予測\nQSAR, ADMET]
        C -->|Reward| D[報酬計算]
        D -->|Update| A
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#e8f5e9
    ```

**報酬関数の設計:**
    
    
    Reward = w1 * QED + w2 * LogP_score + w3 * SA_score + w4 * ADMET_score
    
    QED: Quantitative Estimate of Druglikeness（薬物らしさ）
    LogP_score: 脂溶性スコア（2-5が最適）
    SA_score: Synthetic Accessibility（合成容易性）
    ADMET_score: ADMETモデルの予測値
    
    w1, w2, w3, w4: 重み（多目的最適化）
    

**学習アルゴリズム:** 1\. **Policy Gradient（REINFORCE）** ``` ∇J(θ) = E[∇log π(a|s) * R]

π: 方策（生成確率） R: 累積報酬 ```

  2. **Proximal Policy Optimization（PPO）** \- より安定した学習 \- Clip勾配で大きな更新を防ぐ

**利点:** \- ✅ 明示的な最適化（報酬関数で制御） \- ✅ 多目的最適化が可能 \- ✅ 探索と活用のバランス

**欠点:** \- ❌ 報酬関数の設計が難しい \- ❌ 訓練に時間がかかる

**実例: ReLeaSE（Popova et al., 2018）**
    
    
    技術: RL + LSTM
    目標: LogP、QEDの最適化
    結果: 既知薬物より優れた特性の分子生成
    

### 2.4.6 Graph生成モデル

**原理:** 分子グラフを直接生成

**生成プロセス:**
    
    
    1. 初期状態: 空グラフ
    2. 原子追加: どの原子タイプを追加するか（C, N, O, etc.）
    3. 結合追加: どの原子間に結合を追加するか
    4. 繰り返し: 所望のサイズに到達まで
    

**主要手法:**

  1. **GraphRNN（You et al., 2018）** \- RNNで逐次的にグラフ生成

  2. **Junction Tree VAE（Jin et al., 2018）** \- 分子骨格（Junction Tree）を生成 \- 有効SMILES生成率100%（理論的）

  3. **MoFlow（Zang & Wang, 2020）** \- Normalizing Flowsベース \- 可逆変換で正確な確率密度

**利点:** \- ✅ 有効な分子のみ生成（グラフの化学的制約を組み込み） \- ✅ 3D構造も考慮可能

**欠点:** \- ❌ 計算コストが高い \- ❌ 実装が複雑

* * *

## 2.5 主要データベースとツール

### 2.5.1 ChEMBL

**概要:** \- 生物活性データベース（欧州バイオインフォマティクス研究所 EBI） \- 200万+ 化合物 \- 1,500万+ 生物活性データポイント \- 14,000+ ターゲット

**データ構造:**
    
    
    Compound: ChEMBL ID, SMILES, InChI, 分子量等
    Assay: アッセイタイプ、ターゲット、細胞株等
    Activity: IC50, Ki, EC50, Kd等
    Target: タンパク質、遺伝子、細胞等
    

**API アクセス:**
    
    
    from chembl_webresource_client.new_client import new_client
    
    # ターゲット検索
    target = new_client.target
    kinases = target.filter(target_type='PROTEIN KINASE')
    
    # 化合物活性データ取得
    activity = new_client.activity
    egfr_data = activity.filter(target_chembl_id='CHEMBL203', pchembl_value__gte=6)
    # pchembl_value ≥ 6 → IC50 ≤ 1 μM
    

**使い分け:** \- **ChEMBL** : 生物活性データ、QSAR訓練 \- **PubChem** : 化学構造、文献検索 \- **DrugBank** : 承認薬、臨床情報 \- **BindingDB** : タンパク質-リガンド結合親和性

### 2.5.2 PubChem

**概要:** \- 化学情報データベース（米国NIH） \- 1億+ 化合物 \- 生物活性アッセイデータ（PubChem BioAssay）

**特徴:** \- 無料アクセス \- REST API、FTP提供 \- 2D/3D構造データ \- 文献リンク（PubMed）

**用途:** \- 化合物ライブラリ構築 \- 構造検索（類似性、部分構造） \- 特性データ収集

### 2.5.3 DrugBank

**概要:** \- 承認薬・臨床試験薬データベース（カナダ） \- 14,000+ 薬物 \- 詳細な薬物動態、薬力学データ

**情報:** \- ADMET特性 \- 薬物間相互作用 \- ターゲット情報 \- 臨床試験ステータス

**用途:** \- Drug repurposing（既承認薬の新適応症探索） \- ADMET訓練データ \- ベンチマーク

### 2.5.4 BindingDB

**概要:** \- タンパク質-リガンド結合親和性データベース \- 250万+ 結合データ \- 9,000+ タンパク質

**データタイプ:** \- Ki（阻害定数） \- Kd（解離定数） \- IC50（50%阻害濃度） \- EC50（50%有効濃度）

**用途:** \- ドッキング検証 \- 結合親和性予測モデル訓練

### 2.5.5 RDKit

**概要:** \- オープンソース化学情報処理ライブラリ \- Python, C++, Java対応 \- 創薬MIの標準ツール

**主要機能:** 1\. **分子I/O** : SMILES, MOL, SDF読み書き 2\. **記述子計算** : 200+ 物理化学的特性 3\. **分子指紋** : ECFP, MACCS, RDKit FP 4\. **部分構造検索** : SMARTS, MCS（最大共通部分構造） 5\. **2D描画** : 分子構造の可視化 6\. **3D構造生成** : ETKDG（距離幾何学法）

**使用例:**
    
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    
    # SMILES読み込み
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # アスピリン
    
    # 記述子計算
    mw = Descriptors.MolWt(mol)  # 180.16
    logp = Descriptors.MolLogP(mol)  # 1.19
    
    # 分子指紋
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    # 類似度計算
    mol2 = Chem.MolFromSmiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')  # イブプロフェン
    similarity = DataStructs.TanimotoSimilarity(fp, fp2)  # 0.31
    

* * *

## 2.6 創薬MIワークフローの全体像
    
    
    ```mermaid
    flowchart TB
        A[ターゲット同定\n疾患関連タンパク質] --> B[化合物ライブラリ構築\nChEMBL, PubChem, In-house]
        B --> C[Virtual Screening\nQSAR, Docking]
        C --> D[ヒット化合物\n上位0.1-1%]
        D --> E[ADMET予測\nin silico評価]
        E --> F{ADMET OK?}
        F -->|No| G[構造最適化\nScaffold hopping]
        G --> E
        F -->|Yes| H[in vitro検証\n実験確認]
        H --> I{活性OK?}
        I -->|No| J[Active Learning\nモデル更新]
        J --> C
        I -->|Yes| K[リード化合物]
        K --> L[リード最適化\n多目的最適化]
        L --> M[前臨床試験]
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style K fill:#e8f5e9
        style M fill:#fce4ec
    ```

**各ステップの時間とコスト:**

ステージ | 従来 | AI活用 | 削減率  
---|---|---|---  
ターゲット同定 | 1-2年 | 0.5-1年 | 50%  
リード発見 | 2-3年 | 0.5-1年 | 75%  
リード最適化 | 2-3年 | 1-1.5年 | 50%  
前臨床 | 1-2年 | 0.5-1年 | 50%  
**合計** | **6-10年** | **2.5-4.5年** | **60-70%**  
  
**コスト削減:** \- Virtual Screening: $500M → $50M（90%削減） \- ADMET失敗削減: $200M → $50M（75%削減） \- 実験回数削減: 1,000実験 → 100実験（90%削減）

* * *

## 学習目標の確認

このchapterを完了すると、以下を説明できるようになります：

### 基本理解（Remember & Understand）

  * ✅ 4種類の分子表現法（SMILES、分子指紋、3D記述子、Graph）を説明できる
  * ✅ QSARの原理と5ステップワークフローを理解している
  * ✅ ADMET5項目を具体的に説明できる
  * ✅ 4種類の分子生成モデル（VAE、GAN、Transformer、RL）の違いを理解している
  * ✅ 主要データベース（ChEMBL、PubChem、DrugBank、BindingDB）の特徴を把握している

### 実践スキル（Apply & Analyze）

  * ✅ Lipinski's Rule of Fiveを計算し、薬物らしさを評価できる
  * ✅ Tanimoto係数で分子類似度を計算できる
  * ✅ QSARモデルの性能指標（R²、ROC-AUC）を解釈できる
  * ✅ ADMET予測結果から開発可能性を判断できる
  * ✅ 機械学習手法（RF、SVM、NN、GNN）を適切に選択できる

### 応用力（Evaluate & Create）

  * ✅ 新規創薬プロジェクトに適したワークフローを設計できる
  * ✅ QSARモデルのApplicability Domainを評価できる
  * ✅ 分子生成モデルの適用場面を判断できる
  * ✅ ChEMBL APIを使ってデータセットを構築できる
  * ✅ 創薬MIの全体フローを最適化できる

* * *

## 演習問題

### Easy（基礎確認）

**Q1** : 以下のSMILES文字列が表す化合物は何ですか？
    
    
    CCO
    

a) メタノール b) エタノール c) プロパノール d) ブタノール

解答を見る **正解**: b) エタノール **解説**: \- `C`: 炭素（メチル基） \- `C`: 炭素（メチレン基） \- `O`: 酸素（ヒドロキシ基） 構造: CH3-CH2-OH = エタノール **他の選択肢:** \- a) メタノール: `CO` \- c) プロパノール: `CCCO` \- d) ブタノール: `CCCCO` 

**Q2** : Lipinski's Rule of Fiveで、分子量の上限は何Daですか？

解答を見る **正解**: 500 Da **Lipinski's Rule of Five（再掲）:** 1\. 分子量（MW）: **< 500 Da** 2\. LogP（脂溶性）: < 5 3\. 水素結合ドナー（HBD）: < 5 4\. 水素結合アクセプター（HBA）: < 10 これらを満たす化合物は経口吸収性が高い傾向があります。 

**Q3** : hERG阻害のIC50が0.5 μMの化合物は、リスク評価としてどれに分類されますか？ a) 低リスク b) 中リスク c) 高リスク

解答を見る **正解**: c) 高リスク **hERGリスク評価基準:** \- IC50 < 1 μM: **高リスク**（開発中止を検討） \- 1 < IC50 < 10 μM: 中リスク（慎重な評価必要） \- IC50 > 10 μM: 低リスク（安全） IC50 = 0.5 μM < 1 μM → 高リスク hERG阻害は致死性不整脈（Torsades de pointes）を引き起こす可能性があるため、 創薬において最も重要な毒性評価項目の一つです。 

### Medium（応用）

**Q4** : 2つの分子のECFP4指紋（2048ビット）を比較したところ、以下の結果が得られました。Tanimoto係数を計算してください。
    
    
    分子A: 1ビットが立っている位置数 = 250
    分子B: 1ビットが立っている位置数 = 280
    両方で1ビットが立っている位置数 = 120
    

解答を見る **正解**: Tanimoto = 0.293 **計算:** 
    
    
    Tanimoto = (A ∩ B) / (A ∪ B)
    
    A ∩ B = 120（両方1のビット数）
    A ∪ B = 250 + 280 - 120 = 410
    
    Tanimoto = 120 / 410 = 0.293
    

**解釈:** Tanimoto = 0.293 < 0.50 → 構造的に異なる分子 類似性の目安: \- > 0.85: 非常に類似 \- 0.70-0.85: 類似 \- 0.50-0.70: やや類似 \- **< 0.50: 異なる**（今回のケース） 

**Q5** : QSARモデルを構築し、以下の性能が得られました。このモデルは実用的ですか？
    
    
    訓練データ: R² = 0.92, MAE = 0.3
    テストデータ: R² = 0.58, MAE = 1.2
    

解答を見る **正解**: 実用的ではない（過学習している） **分析:** \- 訓練データ: R² = 0.92（優秀） \- テストデータ: R² = 0.58（不十分） \- **差**: 0.92 - 0.58 = 0.34（大きすぎる） **問題: 過学習（Overfitting）** \- モデルが訓練データに過剰適合 \- 未知データへの汎化性能が低い \- テストR² < 0.70は実用的でない **対策:** 1\. 正則化（L1/L2、Dropout） 2\. 訓練データ追加 3\. 特徴量削減（重要な記述子のみ使用） 4\. より単純なモデル（Random Forestの木の深さ制限等） 5\. 交差検証でハイパーパラメータチューニング 

**Q6** : ある化合物のCaco-2透過性がPapp = 5 × 10^-7 cm/sでした。この化合物の経口吸収性を評価してください。

解答を見る **正解**: 低〜中程度の透過性（吸収性やや不良） **Caco-2透過性の基準:** \- Papp > 10^-6 cm/s: **高透過性**（良好な吸収） \- 10^-7 < Papp < 10^-6 cm/s: **中程度**（吸収は可能だが最適ではない） \- Papp < 10^-7 cm/s: 低透過性（吸収不良） **今回のケース:** Papp = 5 × 10^-7 cm/s 10^-7 < 5 × 10^-7 < 10^-6 → 中程度 **改善戦略:** 1\. 脂溶性の調整（LogP最適化） 2\. TPSA削減（極性表面積を小さく） 3\. 水素結合数の削減 4\. 製剤技術（ナノ粒子、リポソーム） ただし、この値でも経口薬として開発可能なケースはあります（例: アテノロール）。 

### Hard（発展）

**Q7** : ChemVAE（潜在空間196次元）で新規分子を10,000個生成したところ、6,800個が有効なSMILESでした。生成分子のうち、訓練データ（ZINC 250K）と最も類似度が高い分子のTanimoto係数が0.95以上の割合が40%でした。この結果をどう評価しますか？また、どのような改善策がありますか？

解答を見る **評価:** **ポジティブな側面:** \- 有効SMILES生成率 = 6,800 / 10,000 = 68% → ChemVAEの典型的な性能（論文値68%と一致） → 技術的には成功 **ネガティブな側面:** \- Tanimoto > 0.95が40% = 4,000分子 → 訓練データと非常に類似（新規性低い） → 「ほぼコピー」の分子が多すぎる **結論:** 新規性が不十分。既知化合物の再生成に近く、創薬的価値は限定的。 **改善策:** 1\. **潜在空間のサンプリング戦略変更** ```python # 現在: 標準正規分布からサンプリング z = np.random.randn(196) # 改善: 訓練データから遠い領域をサンプリング z = np.random.randn(196) * 2.0 # 分散を大きく ``` 2\. **ペナルティ項の追加** ``` Loss = Reconstruction + KL + λ * Novelty Penalty Novelty Penalty = -log(1 - max_tanimoto) （訓練データとの最大類似度が高いほどペナルティ） ``` 3\. **Conditional VAEの使用** \- 所望の特性（LogP, MW等）を条件として与える \- 特性空間で訓練データから離れた領域を探索 4\. **強化学習との組み合わせ** \- VAE生成分子をRLで最適化 \- 報酬関数に新規性項を追加 ``` Reward = Activity + λ1 * Novelty + λ2 * Druglikeness ``` 5\. **Junction Tree VAEの検討** \- 新規骨格（Scaffold）の生成に優れる \- 有効SMILES生成率100% **実践例:** Insilico MedicineはGANとRLを組み合わせることで、 訓練データと異なる新規骨格を持つ分子を生成し、IPF治療薬候補を発見しました。 

**Q8** : 以下の創薬プロジェクトで、どの機械学習手法を選択すべきか、理由とともに説明してください。
    
    
    プロジェクト: EGFR（上皮成長因子受容体）キナーゼ阻害剤の開発
    データ: ChEMBLから取得したEGFR活性データ 15,000化合物
    タスク: IC50予測（回帰）
    目標: R² > 0.80、予測時間 < 1秒/化合物
    追加要件: モデルの解釈性が欲しい（どの構造が活性に寄与するか）
    

解答を見る **推奨手法: Random Forest** **理由:** 1\. **データサイズとの適合性** \- 15,000化合物 = 中規模データ \- Random Forestは1K-100Kで最適性能 \- Neural Networkは10K+で本領発揮だが、15Kは境界線 \- Random Forestの方が安定（過学習しにくい） 2\. **性能目標の達成可能性** \- ChEMBL EGFR活性予測でのベンチマーク: \- Random Forest: R² = 0.82-0.88（目標R² > 0.80をクリア） \- SVM: R² = 0.78-0.85（やや不安定） \- Neural Network: R² = 0.85-0.90（高いがオーバーキル） \- LightGBM: R² = 0.85-0.92（最高性能だが解釈性低い） 3\. **予測速度** \- Random Forest: < 0.1秒/化合物（目標の10倍高速） \- Neural Network: 0.5-2秒/化合物（GPU不使用時） \- 100万化合物のVirtual Screeningでも現実的 4\. **解釈性（最重要要件）** \- **Feature Importance（特徴量重要度）**: ```python importances = model.feature_importances_ # どの記述子が予測に重要かランキング # 例: LogP, TPSA, 芳香環数, etc. ``` \- **SHAP（SHapley Additive exPlanations）**: ```python import shap explainer = shap.TreeExplainer(model) shap_values = explainer.shap_values(X_test) # 各化合物の各特徴の寄与度を可視化 ``` \- これにより「ATP結合ポケットに適合する疎水性部位が重要」等の知見が得られる **実装例:** 
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import shap
    
    # モデル訓練
    rf = RandomForestRegressor(
        n_estimators=500,  # 木の数（多いほど安定）
        max_depth=20,      # 深さ制限（過学習防止）
        min_samples_leaf=5,
        n_jobs=-1          # 並列化
    )
    
    rf.fit(X_train, y_train)
    
    # 性能評価
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 特徴量重要度
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 important features:")
    print(importances.head(10))
    
    # SHAP解釈
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test[:100])  # サンプル100化合物
    
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
    # → どの特徴が高活性/低活性に寄与するか可視化
    

**代替案（状況に応じて）:** \- **データが10万+に増えた場合**: LightGBM \- より高精度（R² > 0.90可能） \- 高速（Random Forestより2-5倍速） \- Feature Importanceも利用可能 \- **解釈性が最優先の場合**: Linear Regression with feature selection \- 係数が直接解釈可能 \- ただし性能はR² = 0.70程度（目標未達） \- **最高精度が必要な場合**: Graph Neural Network (MPNN) \- R² = 0.90-0.95 \- ただし解釈性低い、訓練時間長い（数時間-数日） \- Attention weightsで部分的な解釈は可能 **結論:** Random Forestが性能・速度・解釈性のバランスに最適。 

* * *

## 次のステップ

第2章で創薬特化のMI手法を理解しました。次の第3章では、これらの手法をPythonで実装し、実際に動かしてみます。RDKitとChEMBLを使った30個のコード例を通じて、実践的なスキルを習得しましょう。

**[第3章: Pythonで実装する創薬MI - RDKit& ChEMBL実践 →](<./chapter3-hands-on.html>)**

* * *

## 参考文献

  1. Gómez-Bombarelli, R., et al. (2018). "Automatic chemical design using a data-driven continuous representation of molecules." _ACS Central Science_ , 4(2), 268-276.

  2. Guimaraes, G. L., et al. (2017). "Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models." _arXiv:1705.10843_.

  3. Jin, W., Barzilay, R., & Jaakkola, T. (2018). "Junction tree variational autoencoder for molecular graph generation." _ICML 2018_.

  4. Lipinski, C. A., et al. (2001). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." _Advanced Drug Delivery Reviews_ , 46(1-3), 3-26.

  5. Popova, M., et al. (2018). "Deep reinforcement learning for de novo drug design." _Science Advances_ , 4(7), eaap7885.

  6. Rogers, D., & Hahn, M. (2010). "Extended-connectivity fingerprints." _Journal of Chemical Information and Modeling_ , 50(5), 742-754.

  7. Wang, S., et al. (2019). "SMILES-BERT: Large scale unsupervised pre-training for molecular property prediction." _BCB 2019_.

  8. Zhavoronkov, A., et al. (2019). "Deep learning enables rapid identification of potent DDR1 kinase inhibitors." _Nature Biotechnology_ , 37(9), 1038-1040.

  9. Gaulton, A., et al. (2017). "The ChEMBL database in 2017." _Nucleic Acids Research_ , 45(D1), D945-D954.

  10. Landrum, G. (2023). RDKit: Open-source cheminformatics. https://www.rdkit.org

* * *

[シリーズ目次に戻る](<./index.html>) | [第1章へ戻る](<./chapter1-background.html>) | [第3章へ進む →](<./chapter3-hands-on.html>)
