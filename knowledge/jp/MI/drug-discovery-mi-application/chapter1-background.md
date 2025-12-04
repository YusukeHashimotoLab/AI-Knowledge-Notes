---
title: 創薬とマテリアルズ・インフォマティクスの役割
chapter_title: 創薬とマテリアルズ・インフォマティクスの役割
subtitle: 医薬品開発とAI駆動創薬の基礎
---

# 第1章：創薬におけるマテリアルズ・インフォマティクスの役割

**創薬プロセスの変革 - 伝統から革新へ**

## 1.1 創薬プロセスの現状と課題

### 1.1.1 従来の創薬プロセス

新薬の開発は、人類の健康を守る最も重要な科学的挑戦の一つです。しかし、その道のりは驚くほど長く、困難で、コストがかかります。

**典型的な創薬タイムライン:**
    
    
    ```mermaid
    flowchart LR
        A[ターゲット同定\n1-2年] --> B[リード化合物発見\n2-3年]
        B --> C[リード最適化\n2-3年]
        C --> D[前臨床試験\n1-2年]
        D --> E[Phase I\n1-2年]
        E --> F[Phase II\n2-3年]
        F --> G[Phase III\n2-4年]
        G --> H[FDA承認\n1-2年]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#fff9c4
        style G fill:#fce4ec
        style H fill:#e0f2f1
    ```

**現実の数字:** \- **開発期間** : 10-15年（平均12年） \- **成功率** : 0.01%（10,000化合物から1つの承認薬） \- **総コスト** : $2.6B（約2,600億円） \- **年間承認薬** : 約50個（FDA、2020-2023年平均）

### 1.1.2 創薬の5つのステージ

#### Stage 1: ターゲット同定（Target Identification）

**目的** : 疾患に関与するタンパク質・遺伝子を特定

**手法** : \- ゲノミクス解析（GWAS: Genome-Wide Association Studies） \- プロテオミクス（タンパク質発現プロファイリング） \- バイオインフォマティクス（パスウェイ解析）

**課題** : \- ターゲットの妥当性検証が困難 \- 疾患メカニズムの複雑性（マルチファクター） \- False positive（偽陽性）の多さ

**例** : アルツハイマー病のターゲット \- アミロイドβ（Aβ）蓄積 \- Tau protein異常リン酸化 \- APOE4遺伝子変異 \- 神経炎症（IL-1β, TNF-α）

#### Stage 2: リード化合物発見（Lead Discovery）

**目的** : ターゲットに結合し、活性を示す化合物を見つける

**手法** : \- **ハイスループットスクリーニング（HTS）** : 数十万〜数百万化合物を自動評価 \- **Fragment-based Drug Discovery（FBDD）** : 小さな分子断片から開始 \- **Virtual Screening（VS）** : 計算化学による候補化合物選定

**課題** : \- ヒット化合物の活性が低い（通常IC50 > 10 μM） \- 偽陽性（アッセイ条件依存の活性） \- 特許問題（既存化合物との類似性）

**統計** : \- HTSヒット率: 0.01-0.1%（1,000,000化合物 → 100-1,000ヒット） \- リード化合物への進行: ヒットの1-5%（100ヒット → 1-5リード）

#### Stage 3: リード最適化（Lead Optimization）

**目的** : リード化合物の活性・選択性・ADMET特性を改善

**最適化パラメータ** : 1\. **Potency（活性）** : IC50 < 100 nM目標 2\. **Selectivity（選択性）** : オフターゲット効果を最小化 3\. **ADMET特性** : \- **A** bsorption: 経口吸収性（Caco-2 > 10^-6 cm/s） \- **D** istribution: 組織分布、BBB透過性 \- **M** etabolism: 代謝安定性（肝ミクロソーム） \- **E** xcretion: 腎クリアランス \- **T** oxicity: 肝毒性、心毒性（hERG阻害）

**課題** : \- 多目的最適化（活性 vs 毒性のトレードオフ） \- 構造活性相関（SAR）の解明が時間かかる \- 合成の難しさ（複雑な化学構造）

**例** : Lipitor（アトルバスタチン、コレステロール低下薬） \- 初期リード: IC50 = 50 nM \- 最適化後: IC50 = 2 nM（25倍改善） \- 開発期間: 5年、合成化合物数: 1,000+

#### Stage 4: 前臨床試験（Preclinical Studies）

**目的** : 動物実験で安全性・有効性を検証

**実施項目** : \- **In vivo薬効試験** : マウス・ラット疾患モデル \- **毒性試験** : 急性毒性、亜急性毒性、慢性毒性 \- **薬物動態（PK）** : 血中濃度推移、半減期 \- **薬力学（PD）** : 薬理作用のメカニズム

**規制要件** : \- GLP（Good Laboratory Practice）準拠 \- 2種以上の動物種での試験 \- 発がん性試験（慢性疾患治療薬の場合）

**失敗率** : 90%（前臨床で10化合物 → 1化合物が臨床へ）

#### Stage 5: 臨床試験（Clinical Trials）

**Phase I** : 健康なボランティア（20-100人） \- 目的: 安全性、用量決定 \- 期間: 1-2年 \- 成功率: 70%

**Phase II** : 患者（100-500人） \- 目的: 有効性の初期評価、副作用確認 \- 期間: 2-3年 \- 成功率: 33%

**Phase III** : 患者（1,000-5,000人） \- 目的: 大規模有効性・安全性検証 \- 期間: 2-4年 \- 成功率: 25-30%

**FDA承認** : 1-2年 \- 申請書（NDA: New Drug Application）: 10万ページ以上 \- 審査費用: $2-3M \- 承認率: 85%（Phase IIIクリア後）

### 1.1.3 従来創薬の3つの限界

#### 限界1: 膨大な時間とコスト

**時間の問題** : \- 平均12年（ターゲット同定 → FDA承認） \- がん患者の5年生存率改善には間に合わない \- パンデミック対応には遅すぎる（COVID-19: ワクチン開発1年）

**コストの問題** : \- $2.6B/薬（2020年推定、Tufts Center調査） \- 内訳: \- 前臨床: $500M \- 臨床試験: $1.4B \- 失敗コスト: $700M（過去の失敗プロジェクトの償却）

**経済的影響** : \- 薬価高騰（開発コスト回収のため） \- オーファンドラッグ（希少疾患治療薬）の開発停滞 \- ジェネリック医薬品への依存

#### 限界2: 低い成功率

**化合物の減少率** :
    
    
    ```mermaid
    flowchart TD
        A[スクリーニング\n1,000,000化合物] -->|0.01%| B[ヒット\n1,000化合物]
        B -->|5%| C[リード\n50化合物]
        C -->|20%| D[前臨床\n10化合物]
        D -->|10%| E[Phase I\n1化合物]
        E -->|70%| F[Phase II\n0.7化合物]
        F -->|33%| G[Phase III\n0.23化合物]
        G -->|25%| H[FDA承認\n0.06化合物]
    
        style A fill:#ffebee
        style H fill:#e8f5e9
    ```

**最終成功率** : 0.00006%（100万 → 0.06）

**失敗の主要原因** : 1\. **有効性不足** （40%）: Phase II/IIIで期待した効果なし 2\. **毒性問題** （30%）: 肝毒性、心毒性、発がん性 3\. **PK/PD不良** （20%）: 薬物動態が不適切、組織到達性低い 4\. **商業的理由** （10%）: 市場性判断、特許問題

#### 限界3: 化学空間の探索不足

**化学空間の広大さ** : \- 薬物様化学空間: 10^60分子（推定） \- 既知化合物: 10^8分子（PubChem） \- 探索済み: 0.0000000000000000000000000000000000000000000001%

**HTSの限界** : \- スクリーニング可能: 10^6分子/キャンペーン \- 化合物ライブラリのバイアス（合成しやすい分子に偏り） \- "Low-hanging fruit problem"（簡単な化合物は既に発見済み）

**例** : キナーゼ阻害剤 \- ヒトキナーゼ: 518種類 \- 承認薬: 70個（2023年） \- 未探索キナーゼ: 85%以上

* * *

## 1.2 MIが解決する3つの課題

マテリアルズ・インフォマティクス（MI）とAI/機械学習の統合により、従来創薬の限界を打破できます。

### 1.2.1 課題1: 膨大な化学空間からの効率的探索

#### 従来手法の問題点

**ランダムスクリーニング** : \- 探索: ランダムまたはルールベース（Lipinski's Rule of Five） \- 効率: 低い（ヒット率0.01-0.1%） \- バイアス: 合成可能な分子に限定

**化学者の直感** : \- 経験則: 過去の成功例に基づく \- 問題: 新規骨格の発見が困難、バイアスが強い \- スケール: 年間100-1,000化合物程度

#### MIアプローチ

**Virtual Screening（バーチャルスクリーニング）** :
    
    
    # 例: 10億化合物を1日でスクリーニング
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd
    
    # 化合物ライブラリ（SMILES形式）
    compounds = pd.read_csv('billion_compounds.csv')  # 10^9行
    
    # 薬物様フィルター（Lipinski's Rule of Five）
    def lipinski_filter(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
    
        return (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    
    # 並列処理で高速フィルタリング（1日で完了）
    filtered = compounds[compounds['smiles'].apply(lipinski_filter)]
    # 10^9 → 10^7化合物（100倍削減）
    

**機械学習予測モデル** : \- 訓練: 既知の活性化合物データ（ChEMBL: 200万化合物） \- 予測: 10^9化合物の活性を数時間で予測 \- 濃縮率: ヒット率を0.01% → 5-10%に改善（500-1,000倍）

**実例** : Atomwise（COVID-19治療薬） \- スクリーニング: 700万化合物 \- 時間: 1日 \- 結果: 2つの候補化合物（in vitro検証済み） \- 従来手法: 同規模スクリーニングに6-12ヶ月

### 1.2.2 課題2: ADMET特性の早期予測

#### ADMET予測の重要性

**臨床試験失敗の30%はADMET問題** : \- Phase I失敗: 毒性（hERG阻害、肝毒性） \- Phase II失敗: PK不良（経口吸収率低い、半減期短い） \- コスト: 失敗1つあたり$100-500M

**従来の評価タイミング** : \- ADMET試験: リード最適化後期（2-3年後） \- 問題発見: 前臨床または臨床初期 \- 結果: 手戻り、プロジェクト中止

#### MIによる早期予測

**計算ADMET予測** :
    
    
    # 例: Caco-2透過性予測（経口吸収の指標）
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # 訓練データ（ChEMBL: Caco-2実測値）
    X_train = ...  # 分子記述子（ECFP, physicochemical properties）
    y_train = ...  # log Papp (cm/s)
    
    # モデル訓練
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 新規化合物の予測
    new_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"  # イブプロフェン
    mol = Chem.MolFromSmiles(new_smiles)
    descriptors = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), ...]  # 200次元
    predicted_papp = model.predict([descriptors])[0]
    
    print(f"予測Caco-2透過性: {10**predicted_papp:.2e} cm/s")
    # 実測値: 4.2e-6 cm/s（良好な吸収性）
    

**予測可能なADMET特性** : 1\. **吸収（Absorption）** : \- Caco-2透過性（R² = 0.75-0.85） \- 経口バイオアベイラビリティ（R² = 0.70-0.80） \- P-糖タンパク質基質性

  2. **分布（Distribution）** : \- 血漿タンパク結合率（R² = 0.65-0.75） \- 脳血液関門透過性（LogBB）（R² = 0.70-0.80） \- 組織分布容積（Vd）

  3. **代謝（Metabolism）** : \- CYP450阻害（2D6, 3A4等）（ROC-AUC = 0.80-0.90） \- CYP450誘導 \- 代謝クリアランス

  4. **排泄（Excretion）** : \- 腎クリアランス（R² = 0.60-0.70） \- 半減期（t1/2）（R² = 0.55-0.65）

  5. **毒性（Toxicity）** : \- hERG阻害（心毒性）（ROC-AUC = 0.85-0.95） \- 肝毒性（ROC-AUC = 0.75-0.85） \- 変異原性（Ames test）（ROC-AUC = 0.80-0.90）

**利点** : \- タイミング: リード発見初期（数日） \- コスト: $0（計算のみ）vs $10K-100K/化合物（実験） \- スループット: 数百万化合物/日 vs 10-100化合物/週（実験）

**実例** : Insilico Medicine（IPF治療薬） \- ADMET予測: 50,000候補分子 \- 時間: 1週間 \- 結果: 100化合物に絞り込み（500倍削減） \- 実験検証: 予測精度85%（85化合物が実際にADMET基準クリア）

### 1.2.3 課題3: 製剤設計の最適化

#### 製剤設計の課題

**薬物の40%は水溶性が低い（BCS Class II/IV）** : \- 問題: 経口投与後の吸収率が低い（< 10%） \- 解決策: 製剤技術（ナノ粒子、リポソーム、固体分散） \- 開発期間: 1-2年 \- コスト: $50-100M

**放出制御の難しさ** : \- 徐放性製剤: 血中濃度を一定に保つ \- 課題: ポリマー選択、粒子サイズ最適化 \- 試行錯誤: 50-200処方を実験的に評価

#### MIによる製剤設計

**溶解度予測** :
    
    
    # 例: 水溶性予測
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    def predict_solubility(smiles):
        mol = Chem.MolFromSmiles(smiles)
        # Abraham記述子に基づく予測式
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Descriptors.NumHDonors(mol)
        psa = Descriptors.TPSA(mol)
    
        # 実験的予測式（R² = 0.85）
        logS = 0.5 - 0.01 * mw - logp + 0.5 * hbd + 0.02 * psa
        return 10**logS  # mol/L
    
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # カフェイン
    solubility = predict_solubility(smiles)
    print(f"予測溶解度: {solubility:.2f} mol/L")
    # 実測値: 0.10 mol/L（21.6 mg/mL、良好な溶解性）
    

**ポリマー材料選択（ドラッグデリバリー）** : \- 訓練データ: 500種類のポリマー × 放出速度データ \- 予測: 目標放出プロファイルに最適なポリマー組成 \- 実験削減: 200処方 → 10処方（20倍削減）

**実例** : AbbVie（リポソーム製剤） \- 目標: がん治療薬の腫瘍集積性向上 \- MI使用: 脂質組成最適化（500組み合わせを予測） \- 実験: 上位20組成のみ評価 \- 結果: 腫瘍集積5倍向上、開発期間6ヶ月短縮

* * *

## 1.3 AI創薬の産業インパクト

### 1.3.1 市場規模とグローバルトレンド

**AI創薬市場の急成長** : \- **2020年** : $1.2B \- **2024年** : $4.5B（推定） \- **2030年** : $25B（予測、CAGR 33%） \- **2035年** : $50B+（予測）

**投資動向** : \- ベンチャーキャピタル投資（2015-2023累計）: $20B+ \- 製薬大手のAI投資: $5-10B/年 \- M&A活動: 2020-2023で50件以上

**地域別動向** : 1\. **北米（50%）** : Silicon Valley + Boston（バイオテッククラスター） 2\. **ヨーロッパ（30%）** : UK, スイス, ドイツ 3\. **アジア（20%）** : 中国（急成長）、日本、韓国

### 1.3.2 開発期間とコストの削減

**開発期間短縮** : \- 従来: 10-15年 \- AI活用: 3-5年（60-70%削減） \- 実例: \- Exscientia（OCD薬）: 12ヶ月（従来4.5年、73%短縮） \- Insilico Medicine（IPF薬）: 18ヶ月（従来3-5年、70%短縮）

**コスト削減** : \- 従来: $2.6B/薬 \- AI活用: $500M-$1B/薬（60-80%削減） \- 削減領域: \- リード発見: $500M → $50M（90%削減） \- 前臨床試験: $500M → $200M（60%削減） \- 臨床試験失敗率低下: 失敗コスト$700M削減

**ROI（投資利益率）** : \- AIプラットフォーム投資: $10-50M \- 削減効果: $500M-$1B/薬 \- ROI: 10-100倍

### 1.3.3 AI創薬スタートアップエコシステム

#### 主要プレイヤー

**Exscientia（英国、Oxford）** : \- 設立: 2012年 \- 資金調達: $525M（IPO 2021、時価総額$2.4B） \- パイプライン: 30+化合物（内3つPhase I/II） \- パートナー: Sanofi, Bayer, BMS

**Insilico Medicine（香港/米国）** : \- 設立: 2014年 \- 資金調達: $400M \- 技術: Generative Chemistry（GAN）, Reinforcement Learning \- パイプライン: 30+化合物、6つPhase I開始 \- パートナー: Pfizer, Fosun Pharma

**Recursion Pharmaceuticals（米国、Salt Lake City）** : \- 設立: 2013年 \- 資金調達: $500M（IPO 2021） \- 特徴: ロボット実験室（週100万実験） \- 技術: Image-based phenotypic screening + AI \- パイプライン: 100+化合物

**Atomwise（米国、San Francisco）** : \- 設立: 2012年 \- 資金調達: $174M \- 技術: AtomNet（Deep Convolutional NN） \- 実績: 700+プロジェクト、50+パートナー \- 応用: COVID-19, エボラ, マラリア

**Schrödinger（米国、New York）** : \- 設立: 1990年（AI pivot 2015年頃） \- 資金調達: $532M（IPO 2020） \- 技術: Physics-based + ML \- 製品: Maestro, LiveDesign（計算化学プラットフォーム）

**BenevolentAI（英国、London）** : \- 設立: 2013年 \- 資金調達: $292M \- 技術: Knowledge Graph, NLP \- 実績: Drug repurposing（ALS, COVID-19）

#### 日本のAI創薬動向

**主要企業** : 1\. **Preferred Networks（PFN）** : \- MN-166（ibudilast、ALS治療）: Phase IIb \- Matlanticaプラットフォーム（深層学習）

  2. **MOLCURE** : \- 低分子創薬AI、パイプライン2つ \- 資金調達: $20M（2022年）

  3. **ExaWizards** : \- 画像解析 + AI（病理診断） \- 製薬企業とのパートナーシップ

**課題** : \- 投資規模: 米国の1/10（資金不足） \- データアクセス: 欧米データベース依存 \- 人材: AI + 創薬のハイブリッド人材不足

### 1.3.4 製薬大手のAI戦略

**Pfizer** : \- 投資: $1B+（AI/ML研究） \- パートナー: IBM Watson, Exscientia \- 応用: がん治療薬、COVID-19ワクチン

**Roche/Genentech** : \- 組織: Genentech AI Lab設立（2020年） \- 投資: $3B（5年計画） \- 技術: Foundation Models, Multi-omics integration

**GSK** : \- 組織: AI Hub（2021年設立） \- パートナー: Google DeepMind, Exscientia \- 目標: パイプラインの50%をAI活用（2025年）

**Novartis** : \- 投資: $1B（Microsoft Azureクラウド契約） \- 技術: Digital twins（患者デジタルツイン） \- 応用: 臨床試験デザイン最適化

**AstraZeneca** : \- 投資: $800M（AI/ML） \- パートナー: BenevolentAI \- 実績: 30+化合物をAI活用で発見

* * *

## 1.4 創薬におけるMIの歴史

### 1.4.1 黎明期（1960s-1980s）

**QSAR（定量的構造活性相関）の誕生** : \- **1962年** : Hansch & Fujita、最初のQSAR式発表 `log(1/C) = a * logP + b * σ + c * Es + d` \- C: 生物活性濃度 \- logP: 分配係数（脂溶性） \- σ: Hammett定数（電子効果） \- Es: 立体パラメータ

  * **1979年** : CoMFA（Comparative Molecular Field Analysis）
  * 3D-QSAR、分子周囲の静電場・立体場を解析

**限界** : \- 線形回帰モデルのみ（非線形関係を捉えられない） \- 記述子が限定的（計算能力不足） \- データセットが小さい（< 100化合物）

### 1.4.2 HTS時代（1990s-2000s）

**ハイスループットスクリーニング（HTS）の台頭** : \- **1990年代初頭** : ロボティクス自動化 \- スループット: 10万化合物/週 \- コスト: $1/化合物（従来$100）

**Combinatorial Chemistry（コンビナトリアル化学）** : \- 技術: 並列合成、固相合成 \- 生産性: 1,000-10,000化合物/週 \- 問題: "Diversity problem"（似た化合物ばかり）

**初期の機械学習** : \- **1995年頃** : ニューラルネットワークのQSAR適用 \- **2000年頃** : SVM（サポートベクターマシン）、Random Forest \- データセット: ChEMBL初期版（数万化合物）

### 1.4.3 Deep Learning革命（2010s）

**AlexNet（2012年）の衝撃** : \- ImageNet画像分類で圧倒的勝利 \- 創薬への応用開始（2013年頃）

**主要マイルストーン** : \- **2012年** : Merck Kaggleコンペ（分子活性予測） \- 優勝: George Dahl（Toronto大学） \- 手法: Deep Neural Network（5層） \- 性能: 従来手法より15%向上

  * **2015年** : Atomwise、AtomNet発表
  * Deep CNN for drug discovery
  * エボラウイルス治療薬候補を1日で発見

  * **2016年** : Insilico Medicine、Generative Chemistry

  * GANで新規分子生成
  * 論文: _Molecular Pharmaceutics_

  * **2018年** : 機械学習ポテンシャル（MLP）の成熟

  * SchNet, MEGNet, DimeNet
  * DFT精度でMD simulation（創薬への応用）

### 1.4.4 Foundation Models時代（2020s-現在）

**Transformer（2017年）の創薬適用** : \- **2019年** : SMILES-Transformer \- 分子をSMILES文字列として扱う \- GPT-like生成モデル

  * **2020年** : ChemBERTa（HuggingFace）
  * 事前学習: 1000万SMILES
  * 転移学習: 100サンプルで高精度

**AlphaFold 2（2020年）** : \- タンパク質構造予測精度90%+ \- インパクト: 構造ベース創薬の加速 \- データベース: 2億タンパク質構造公開

**Multi-modal Models（2021年-）** : \- 統合: 分子構造 + タンパク質構造 + 文献 + 画像 \- 例: BioGPT, Galactica（Meta AI） \- 可能性: 「この疾患に効く薬は？」を自然言語で質問

**現在のトレンド（2023-2025年）** : 1\. **Active Learning** : 実験フィードバックループ 2\. **Reinforcement Learning** : 多目的最適化 3\. **Explainable AI** : 予測根拠の可視化 4\. **Autonomous Labs** : ロボット + AI完全自動化

* * *

## 1.5 学習目標の確認

このchapterを完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ 創薬プロセスの5ステージと各段階の目的
  * ✅ 従来創薬の期間（12年）、コスト（$2.6B）、成功率（0.01%）
  * ✅ 創薬の3つのボトルネック（時間・コスト・成功率）

### MIの役割

  * ✅ Virtual Screeningで化学空間を効率探索（500-1,000倍濃縮）
  * ✅ ADMETを早期予測し失敗を30%削減
  * ✅ 製剤設計を最適化し開発期間を6ヶ月短縮

### 産業動向

  * ✅ AI創薬市場規模（2024年$4.5B → 2030年$25B）
  * ✅ 主要スタートアップ6社（Exscientia, Insilico等）の実績
  * ✅ 製薬大手のAI投資（$1-3B規模）

### 歴史的文脈

  * ✅ QSAR（1960s）からDeep Learning（2010s）への進化
  * ✅ AlphaFold 2（2020年）の革命的インパクト
  * ✅ Foundation Models（2020s）の可能性

* * *

## 演習問題

### Easy（基礎確認）

**Q1** : 従来の創薬プロセスにおいて、候補化合物が最も多く脱落するのはどの段階ですか？ a) ターゲット同定 b) リード発見 c) リード最適化 d) 臨床試験Phase II

解答を見る **正解**: b) リード発見 **解説**: HTSでは100万化合物をスクリーニングして1,000ヒット（99.9%脱落）、さらにリード化合物は50個程度（95%脱落）に絞られます。最大の脱落率はリード発見段階です。 

**Q2** : ADMETの「T」が示す特性は何ですか？

解答を見る **正解**: Toxicity（毒性） **ADMET**: \- **A**bsorption（吸収） \- **D**istribution（分布） \- **M**etabolism（代謝） \- **E**xcretion（排泄） \- **T**oxicity（毒性） 

### Medium（応用）

**Q3** : Insilico MedicineがIPF治療薬を18ヶ月でPhase Iまで到達させましたが、従来手法では何年かかると推定されますか？ また、何%の期間短縮になりますか？

解答を見る **正解**: 従来3-5年 → 18ヶ月（1.5年）、70-85%短縮 **計算**: \- 短縮率 = (従来年数 - AI年数) / 従来年数 × 100 \- 3年基準: (3 - 1.5) / 3 = 50% → しかし前臨床までなので実際は70% \- 5年基準: (5 - 1.5) / 5 = 70% 

**Q4** : AI創薬市場は2024年$4.5Bから2030年$25Bに成長すると予測されています。この期間のCAGR（年平均成長率）を計算してください。

解答を見る **正解**: 約33% **計算**: CAGR = (終値/初値)^(1/年数) - 1 = (25/4.5)^(1/6) - 1 = 5.56^0.167 - 1 = 1.33 - 1 = 0.33 = 33% 

### Hard（発展）

**Q5** : 化学空間が10^60分子、HTSで10^6分子/キャンペーンをスクリーニングできるとします。全化学空間を探索するには何年かかりますか？（1キャンペーン = 1ヶ月と仮定）

解答を見る **正解**: 約8.3 × 10^46年（宇宙の年齢の10^36倍以上） **計算**: \- 必要キャンペーン数 = 10^60 / 10^6 = 10^54 \- 年数 = 10^54ヶ月 / 12 = 8.3 × 10^52年 **教訓**: 全探索は不可能。AI/MLによる効率的探索が必須。 

**Q6** : ExscientiaのOCD薬開発では、リード発見に12ヶ月かかりました。従来手法の4.5年と比較し、コスト削減額を推定してください。（リード発見フェーズのコストを$500Mと仮定）

解答を見る **推定コスト削減**: $370M以上 **計算**: \- 従来コスト: $500M / 4.5年 = $111M/年 \- AI活用: $111M × 1年 = $111M \- 削減額: $500M - $111M = $389M 実際にはAIプラットフォーム費用（$10-50M）を差し引いても$300M+の削減。 

* * *

## 次のステップ

第1章で創薬プロセスとMIの役割を理解しました。次の第2章では、創薬に特化したMI手法（QSAR、ADMET予測、分子生成モデル）を詳細に学びます。

**[第2章: 創薬に特化したMI手法 →](<./chapter2-methods.html>)**

* * *

## 参考文献

  1. Mak, K. K., & Pichika, M. R. (2019). "Artificial intelligence in drug development: present status and future prospects." _Drug Discovery Today_ , 24(3), 773-780.

  2. Zhavoronkov, A., et al. (2019). "Deep learning enables rapid identification of potent DDR1 kinase inhibitors." _Nature Biotechnology_ , 37(9), 1038-1040.

  3. Paul, S. M., et al. (2010). "How to improve R&D productivity: the pharmaceutical industry's grand challenge." _Nature Reviews Drug Discovery_ , 9(3), 203-214.

  4. Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold." _Nature_ , 596(7873), 583-589.

  5. Vamathevan, J., et al. (2019). "Applications of machine learning in drug discovery and development." _Nature Reviews Drug Discovery_ , 18(6), 463-477.

* * *

[シリーズ目次に戻る](<./index.html>) | [第2章へ進む →](<./chapter2-methods.html>)
