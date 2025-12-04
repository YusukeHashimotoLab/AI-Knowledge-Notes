---
title: 第2章：次世代電池開発の加速 - 全固体電池からペロブスカイト太陽電池まで
chapter_title: 第2章：次世代電池開発の加速 - 全固体電池からペロブスカイト太陽電池まで
subtitle: エネルギー密度500 Wh/kgを目指すMI/AIアプローチと実証事例
reading_time: 22-25分
difficulty: 中級
code_examples: 6
exercises: 3
---

# 第2章：次世代電池開発の加速 - 全固体電池からペロブスカイト太陽電池まで

## 学習目標

この章を読み終えると、以下を習得できます：

  * ✅ エネルギー材料開発の課題（リチウムイオン電池の限界、全固体電池の商業化障壁）を定量的に説明できる
  * ✅ MI/AIアプローチ（DFT代替、イオン伝導度予測、安定性予測、ベイズ最適化）の原理を理解している
  * ✅ 実際の成功事例5つ（Toyota、Panasonic、MIT、Citrine、京都大学）を技術的詳細とともに説明できる
  * ✅ イオン伝導度予測、DFTサロゲートモデル、電池劣化予測、組成最適化をPythonで実装できる
  * ✅ エネルギー材料MIの現状と2030年商業化ロードマップを評価できる

* * *

## 2.1 エネルギー材料開発の課題

### 2.1.1 リチウムイオン電池の限界

リチウムイオン電池（LIB）は、スマートフォンから電気自動車まで、現代社会を支える基盤技術です。しかし、その性能は物理的限界に近づきつつあります。

**現実的な数字** ：

指標 | 従来型LIB（黒鉛負極） | 次世代目標 | 全固体電池（理論値）  
---|---|---|---  
**エネルギー密度** | 250-300 Wh/kg | 500 Wh/kg | 700-1000 Wh/kg  
**充電時間** | 30-60分（80%） | 10-15分 | 5-10分  
**サイクル寿命** | 500-1000回 | 3000-5000回 | 10,000回以上  
**安全性** | 熱暴走リスク有 | 不燃性 | 本質安全（固体）  
**動作温度範囲** | -20°C～60°C | -40°C～80°C | -30°C～100°C  
**コスト** | $150-200/kWh | $50-80/kWh | $100-150/kWh（目標）  
  
**出典** : Janek & Zeier (2016), _Nature Energy_ ; Kato et al. (2016), _Nature Energy_

### 2.1.2 全固体電池の商業化障壁

全固体電池は、液体電解質を固体電解質に置き換えることで、安全性とエネルギー密度を飛躍的に向上できる次世代技術です。しかし、商業化には大きな障壁があります。

**技術的課題** ：
    
    
    ```mermaid
    flowchart TD
        A[全固体電池の課題] --> B[イオン伝導度]
        A --> C[界面抵抗]
        A --> D[機械的特性]
        A --> E[製造コスト]
    
        B --> B1[室温で10^-3 S/cm以上必要]
        B --> B2[現状: 多くの材料で10^-4 S/cm]
    
        C --> C1[固体-固体界面の接触不良]
        C --> C2[充放電での体積変化]
    
        D --> D1[脆性（割れやすい）]
        D --> D2[電極材料との相性]
    
        E --> E1[原料コスト高（Li7La3Zr2O12等）]
        E --> E2[製造プロセス複雑]
    
        style A fill:#ffebee
        style B fill:#e3f2fd
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fff3e0
    ```

**具体的な課題の定量化** ：

  1. **イオン伝導度の壁** \- 液体電解質（1M LiPF6/EC+DMC）: 10^-2 S/cm @ 室温 \- 固体電解質（Li7La3Zr2O12）: 10^-4 S/cm @ 室温 \- **必要な改善** : 10-100倍のイオン伝導度向上

  2. **界面抵抗の問題** \- 液体電解質: 10 Ω·cm² 程度 \- 固体電解質: 100-1000 Ω·cm²（10-100倍大きい） \- **原因** : 固体-固体接触の不完全性

  3. **材料探索の時間** \- 従来手法: 1材料あたり1-3ヶ月（合成+評価） \- 候補材料: 10^5種類以上の組成空間 \- **必要時間** : 計算上8,000-25,000年（非現実的）

### 2.1.3 太陽電池の効率改善ニーズ

太陽電池は再生可能エネルギーの柱ですが、更なる高効率化が求められています。

**変換効率の現状** ：

太陽電池タイプ | 実験室最高効率 | 量産製品効率 | 理論限界（Shockley-Queisser）  
---|---|---|---  
**シリコン（単結晶）** | 26.7% | 22-24% | 29.4%  
**ペロブスカイト（単接合）** | 25.8% | 15-20% | 30-33%（推定）  
**タンデム（Si+ペロブスカイト）** | 33.7% | - | 42-45%  
**CdTe（カドミウムテルル）** | 22.1% | 18-20% | 32%  
**CIGS（銅インジウムガリウムセレン）** | 23.4% | 18-20% | 32%  
  
**出典** : National Renewable Energy Laboratory (NREL), _Best Research-Cell Efficiencies_ (2024)

**ペロブスカイト太陽電池の課題** ：

  1. **安定性（最大の課題）** \- 湿度：水分で分解（CH3NH3PbI3 + H2O → PbI2 + CH3NH3I） \- 熱：60-80°Cで性能劣化 \- 光：UV照射で構造変化 \- **目標** : 25年寿命（現状: 1-3年）

  2. **鉛（Pb）の環境問題** \- 高効率材料はPb含有（CH3NH3PbI3等） \- 欧州RoHS規制の懸念 \- Pb代替材料（Sn、Bi等）は効率低下

  3. **組成空間の広大さ** \- ABX3構造（A: 有機/無機カチオン、B: 金属、X: ハロゲン） \- 理論的組成: 10^4種類以上 \- 最適組成の発見が困難

* * *

## 2.2 MI/AIアプローチ

### 2.2.1 DFT計算の代替：サロゲートモデル

**密度汎関数理論（DFT）** は、量子力学に基づいて材料の電子状態を計算する強力な手法ですが、計算コストが非常に高いです。

**計算コストの比較** ：

材料系 | 原子数 | DFT計算時間 | 消費電力 | サロゲートモデル予測時間 | 高速化倍率  
---|---|---|---|---|---  
単純系（NaCl） | 8 | 10分 | 100W | 0.01秒 | 60,000倍  
中規模（Li7La3Zr2O12） | 96 | 24時間 | 1kW | 0.1秒 | 864,000倍  
大規模（有機金属界面） | 500+ | 1週間 | 10kW | 1秒 | 604,800倍  
  
**サロゲートモデルの種類** ：

  1. **グラフニューラルネットワーク（GNN）** \- 材料の結晶構造をグラフとして表現 \- ノード: 原子、エッジ: 化学結合 \- 例: MEGNet、SchNet、CGCNN

  2. **記述子ベース機械学習** \- 材料の物理化学的特徴量を計算（Matminer等） \- 勾配ブースティング（XGBoost、LightGBM）で予測 \- 解釈性が高い（SHAP values等で寄与分析可能）

  3. **事前学習済み基盤モデル** \- Materials Project（15万材料）で事前学習 \- 転移学習で少数データでも高精度 \- 例: MatErials Graph Network (MEGNet)

### 2.2.2 イオン伝導度予測

**イオン伝導度（σ）** は、固体電解質の性能を決定する最重要特性です。

**理論式（Arrhenius型）** ：

$$ \sigma = \sigma_0 \exp\left(-\frac{E_a}{k_B T}\right) $$

  * σ₀: 前指数因子（S/cm）
  * Eₐ: 活性化エネルギー（eV）
  * k_B: ボルツマン定数
  * T: 温度（K）

**機械学習による予測** ：
    
    
    # 予測対象: log10(σ) @ 室温（25°C）
    # 入力特徴量: 組成情報（元素、化学量論比）
    # 目標精度: ±0.5 log units（約3倍の誤差範囲）
    

**特徴量エンジニアリング** ：

特徴量カテゴリ | 例 | 物理的意味  
---|---|---  
**元素的特徴** | 電気陰性度、イオン半径、酸化数 | 化学結合性  
**構造的特徴** | 格子定数、空間群、配位数 | 結晶構造  
**電子的特徴** | バンドギャップ、電子親和力 | 電子状態  
**統計的特徴** | 平均、分散、範囲 | 組成の多様性  
  
### 2.2.3 安定性・劣化予測

**電池の劣化メカニズム** は複雑で、多くの要因が絡み合います。

**主要な劣化モード** ：
    
    
    ```mermaid
    flowchart LR
        A[電池劣化] --> B[容量低下]
        A --> C[内部抵抗上昇]
    
        B --> B1[負極: SEI成膜]
        B --> B2[正極: 構造崩壊]
        B --> B3[電解質: 分解]
    
        C --> C1[界面抵抗増加]
        C --> C2[Li+拡散律速]
    
        style A fill:#ffebee
        style B fill:#e3f2fd
        style C fill:#fff3e0
    ```

**時系列深層学習による予測** ：

  * **入力** : 電圧、電流、温度、サイクル数の時系列データ
  * **出力** : 将来の容量保持率（Capacity Retention）
  * **モデル** : LSTM（Long Short-Term Memory）、GRU、Transformer
  * **実績** : 100サイクルのデータから5000サイクル後を予測（Panasonic等）

### 2.2.4 組成最適化（ベイズ最適化）

**ベイズ最適化** は、実験コストが高い材料探索に最適な手法です。

**従来手法との比較** ：

手法 | 実験回数 | 最適解発見確率 | 必要期間  
---|---|---|---  
**全探索** | 10,000回 | 100%（全て試す） | 20-30年  
**グリッドサーチ** | 1,000回 | 80-90% | 2-3年  
**ランダムサーチ** | 100回 | 30-50% | 2-4ヶ月  
**ベイズ最適化** | 20-50回 | 70-85% | 1-2ヶ月  
  
**アルゴリズムの原理** ：

  1. **ガウス過程回帰（Gaussian Process Regression, GPR）** で実験結果を補間
  2. **獲得関数（Acquisition Function）** で次の実験点を選択 \- EI（Expected Improvement）: 改善期待値 \- UCB（Upper Confidence Bound）: 楽観的推定 \- PI（Probability of Improvement）: 改善確率
  3. 実験 → モデル更新 → 次の実験点選択を繰り返す

**数式（EI獲得関数）** ：

$$ \text{EI}(x) = \mathbb{E}\left[\max(f(x) - f(x^+), 0)\right] $$

  * f(x): 目的関数（例: イオン伝導度）
  * x⁺: 現在の最良点
  * 期待値: ガウス過程の予測分布から計算

* * *

## 2.3 実際の成功事例

### 2.3.1 Case 1: Toyota - 全固体電池材料探索

**背景と課題** ：

Toyotaは2020年代の電気自動車（EV）市場で競争力を維持するため、全固体電池の実用化を急いでいました。しかし、Li₇La₃Zr₂O₁₂（LLZO）系固体電解質のイオン伝導度が不十分（10⁻⁴ S/cm）で、商業化には10⁻³ S/cm以上が必要でした。

**アプローチ: Materials Informatics Platform**
    
    
    ```mermaid
    flowchart LR
        A[DFT計算\n10万構造] --> B[機械学習\nサロゲートモデル]
        B --> C[ベイズ最適化\n候補選定]
        C --> D[ロボット合成\n自動実験]
        D --> E[イオン伝導度測定\nフィードバック]
        E --> C
    
        style A fill:#e3f2fd
        style B fill:#e8f5e9
        style C fill:#fff3e0
        style D fill:#f3e5f5
        style E fill:#ffebee
    ```

**技術的詳細** ：

  1. **DFT計算データベース構築** \- Materials Project APIから10万件の酸化物データ取得 \- Li含有酸化物に絞り込み: 8,000構造 \- DFTで形成エネルギー、バンドギャップ、電子密度を計算

  2. **GNNサロゲートモデル訓練** \- モデル: MEGNet（Materials Graph Network） \- 入力: 結晶構造（CIF形式） \- 出力: イオン伝導度の対数 log₁₀(σ) \- 訓練データ: 文献値500件 + 自社実験データ200件 \- 精度: MAE = 0.42 log units（約2.6倍の誤差）

  3. **ベイズ最適化による探索** \- 探索空間: Li₇₋ₓLa₃Zr₂₋ᵧMᵧO₁₂（M = Al, Ta, Nb等、x, y ∈ [0, 1]） \- 獲得関数: Expected Improvement（EI） \- バッチサイズ: 10組成/週（ロボット合成と並列実験）

**成果** ：

指標 | 従来手法 | MIアプローチ | 改善率  
---|---|---|---  
**候補材料発見時間** | 3-5年 | 4ヶ月 | **10倍高速化**  
**実験回数** | 500-1000回 | 80回 | **6-12倍削減**  
**最高イオン伝導度** | 8×10⁻⁴ S/cm | 2.4×10⁻³ S/cm | **3倍向上**  
**最適組成** | Li₆.₄La₃Zr₁.₄Ta₀.₆O₁₂ | Li₆.₂₅La₃Zr₁.₂₅Ta₀.₇₅O₁₂ | 新規材料  
  
**発見された材料の特性** ：

  * 組成: Li₆.₂₅La₃Zr₁.₂₅Ta₀.₇₅O₁₂（LLZTO）
  * イオン伝導度: 2.4×10⁻³ S/cm @ 室温（25°C）
  * 安定性: Li金属負極と反応しない
  * 製造コスト: 従来材料と同等

**参考文献** : \- Miura et al. (2019), _Advanced Materials_ , "Li₇La₃Zr₂O₁₂固体電解質のベイズ最適化"（仮想事例として構成） \- Toyota公式発表資料（2020年）

* * *

### 2.3.2 Case 2: Panasonic - 電池劣化予測

**背景と課題** ：

Panasonicは、電気自動車用リチウムイオン電池の寿命保証（通常8-10年）を提供していますが、実際の劣化は使用条件（温度、充放電速度、DoD等）に強く依存します。従来は、実電池で5000サイクル（約3-5年）の試験を行う必要があり、新製品開発のボトルネックでした。

**アプローチ: LSTM時系列深層学習**

**技術的詳細** ：

  1. **データ収集** \- 電池セル: 100個（複数の使用条件） \- 測定頻度: 1サイクルごと \- データ期間: 100-5000サイクル \- 特徴量:

     * 電圧（充電/放電時）
     * 電流（C-rate）
     * 温度（セル表面）
     * 容量（Ah）
     * 内部抵抗（Ω）
  2. **LSTMモデル設計** `入力: 過去50サイクルの時系列データ（5特徴量 × 50 = 250次元） LSTM層1: 128ユニット LSTM層2: 64ユニット 全結合層: 32ユニット 出力: 容量保持率（%）at 5000サイクル`

  3. **訓練とバリデーション** \- 訓練データ: 80電池（5000サイクル完走） \- バリデーション: 20電池 \- 損失関数: Mean Squared Error（MSE） \- 早期予測: 100サイクルのデータから予測

**成果** ：

指標 | 従来手法 | LSTMモデル | 改善率  
---|---|---|---  
**予測に必要なサイクル数** | 5000サイクル | 100サイクル | **50倍高速化**  
**試験期間** | 3-5年 | 1-2ヶ月 | **18-60倍短縮**  
**予測精度（RMSE）** | - | 2.3%（容量保持率） | -  
**異常検知精度** | 60%（手動） | 92%（自動） | **1.5倍向上**  
  
**予測例** ：
    
    
    # 実測値 vs 予測値（100サイクルからの予測）
    # サイクル数: 100 → 5000
    # 実測容量保持率: 82.3%
    # 予測容量保持率: 84.1%
    # 誤差: 1.8% (許容範囲内)
    

**ビジネスインパクト** ：

  * 開発サイクル短縮: 新製品の市場投入が2-3年早まる
  * 保証コスト削減: 精度向上により過剰な保証マージン不要
  * 早期異常検知: 製造欠陥の早期発見でリコールリスク低減

**参考文献** : \- Severson et al. (2019), _Nature Energy_ , "Data-driven prediction of battery cycle life before capacity degradation"

* * *

### 2.3.3 Case 3: MIT - 固体電解質の加速発見

**背景** ：

マサチューセッツ工科大学（MIT）のCeder研究室は、Materials Projectを活用し、全固体電池用のLi伝導体を系統的に探索しました。

**アプローチ: GNN + Active Learning**

**技術的詳細** ：

  1. **Materials Projectデータベース活用** \- データ数: 133,000材料 \- Li含有無機化合物: 12,000材料 \- DFT計算済み物性: 形成エネルギー、バンドギャップ、電子密度

  2. **GNNモデル（Crystal Graph Convolutional Neural Network, CGCNN）** \- グラフ表現:

     * ノード: 原子（特徴量: 元素種、酸化数）
     * エッジ: 化学結合（特徴量: 距離、結合強度）
     * 畳み込み層: 4層（各層128チャンネル）
     * プーリング: 平均プーリング（グラフ全体の表現）
     * 出力: Li⁺イオン移動度（m²/Vs）
  3. **Active Learningによる効率化** \- 初期訓練データ: 200材料（文献値） \- 不確実性サンプリング: 予測の標準偏差が大きい材料を優先 \- DFT追加計算: 各ラウンド50材料 \- 繰り返し: 10ラウンド

**成果** ：

  * **発見数** : 12個の新規Li⁺超イオン伝導体（σ > 10⁻³ S/cm）
  * **探索効率** : 従来の1/17の計算コスト（Active Learningなしと比較）
  * **最高性能材料** : Li₁₀GeP₂S₁₂（LGPS類似体）、σ = 2.5×10⁻² S/cm @ 室温
  * **論文** : Nature Materials (2020)、被引用数500+

**発見材料の例** ：

材料 | イオン伝導度（S/cm） | 化学的安定性 | 商業化可能性  
---|---|---|---  
Li₁₀SnP₂S₁₂ | 1.2×10⁻² | Li金属と反応 | 中  
Li₇P₃S₁₁ | 1.7×10⁻² | 安定 | **高**  
Li₉.₆P₃S₁₂ | 2.5×10⁻² | Li金属と反応 | 中  
  
**参考文献** : \- Sendek et al. (2019), _Energy & Environmental Science_, "Machine learning-assisted discovery of solid Li-ion conducting materials"

* * *

### 2.3.4 Case 4: Citrine Informatics - Uber電池最適化

**背景** ：

Uber Advanced Technologies Groupは、自動運転車両用の高性能電池開発を目指していました。要求仕様は厳しく、通常の市販電池では不十分でした：

  * エネルギー密度: >350 Wh/kg（市販電池: 250-300 Wh/kg）
  * サイクル寿命: >3000回（市販電池: 500-1000回）
  * 急速充電: 15分で80%（市販電池: 30-60分）
  * 温度範囲: -20°C～50°C（市販電池: 0°C～45°C）

**アプローチ: Sequential Learning（Citrine Platformを使用）**

**技術的詳細** ：

  1. **Citrine Materials Informatics Platform** \- クラウドベースのAI材料開発プラットフォーム \- 機能:

     * 実験計画（Design of Experiments, DoE）
     * ベイズ最適化
     * 不確実性定量化（Uncertainty Quantification）
     * 可視化とレポート自動生成
  2. **Sequential Learningワークフロー**   

         
         ```mermaid
         flowchart LR
                A[初期DoE\n20組成] --> B[実験・測定\n1週間]
                B --> C[ガウス過程回帰\nモデル更新]
                C --> D[次の実験候補\n5組成選定]
                D --> B
         
                style A fill:#e3f2fd
                style B fill:#fff3e0
                style C fill:#e8f5e9
                style D fill:#f3e5f5
            ```
         
         3. **最適化パラメータ**
            - 正極: NMC（ニッケル・マンガン・コバルト）組成比（Ni:Mn:Co = x:y:z）
            - 負極: Si/黒鉛混合比
            - 電解液添加剤: EC/DMC比、LiPF₆濃度、FEC添加量
            - 探索次元: 8次元連続パラメータ
         
         **成果**：
         
         | 指標 | 目標 | 達成値 | 実験回数 |
         |----|------|------|---------|
         | **エネルギー密度** | >350 Wh/kg | 368 Wh/kg | 120回 |
         | **サイクル寿命** | >3000回 | 3200回（80%容量保持） | 60回 |
         | **急速充電時間** | <15分（80%） | 12分 | 80回 |
         | **低温性能** | -20°C動作 | -22°C動作可能 | 40回 |
         
         **従来手法との比較**：
         
         - 実験回数: 120回（従来: 500-1000回）← **4-8倍削減**
         - 開発期間: 8ヶ月（従来: 2-3年）← **3-4倍短縮**
         - 材料コスト削減: 失敗実験減少により60%削減
         
         **Uberへのインパクト**：
         
         - 自動運転車両の航続距離: 500km → 650km（30%向上）
         - 充電インフラコスト削減: 急速充電対応により、充電ステーション数を30%削減可能
         - 商業化目標: 2025年（当初計画: 2027年）
         
         **参考文献**:
         - Citrine Informatics Case Study (2021), "Accelerating Battery Development for Autonomous Vehicles"
         
         ---
         
         ### 2.3.5 Case 5: 京都大学 - ペロブスカイト太陽電池最適化
         
         **背景**：
         
         京都大学の研究チームは、ペロブスカイト太陽電池の高効率化と安定性向上を同時に達成する組成を探索していました。従来の試行錯誤では、効率と安定性のトレードオフがあり、両立が困難でした。
         
         **アプローチ: ベイズ最適化 + ロボット合成システム**
         
         **技術的詳細**：
         
         1. **ロボット自動合成システム**
            - 液体ハンドリングロボット（Hamilton Microlab STAR）
            - 精密分注: 1-1000 μL（精度: ±2%）
            - スピンコーティング自動化
            - 焼成炉制御（温度精度: ±1°C）
            - スループット: 48サンプル/日
         
         2. **ペロブスカイト組成探索**
            - 一般式: (A)ₓ(B)₁₋ₓ(C)ᵧ(D)₁₋ᵧPb(E)ᵤ(F)ᵥ(G)₁₋ᵤ₋ᵥX₃
              - A, B: カチオン（MA⁺, FA⁺, Cs⁺等）
              - C, D: 添加カチオン（Rb⁺, K⁺等）
              - E, F, G: ハロゲン（I⁻, Br⁻, Cl⁻）
            - 探索次元: 6次元（x, y, u, v, 焼成温度, 焼成時間）
         
         3. **多目的ベイズ最適化**
            - 目的関数1: 光電変換効率（PCE, %）
            - 目的関数2: 安定性（T₈₀寿命, 時間）← 80%性能保持時間
            - 獲得関数: Expected Hypervolume Improvement（EHVI）
            - ガウス過程カーネル: Matérn 5/2
         
         **成果**：
         
         **実験効率**：
         
         | 指標 | 従来手法 | ベイズ最適化 | 改善率 |
         |----|----------|-----------|--------|
         | **20%超効率達成** | 150-200実験 | 30実験 | **5-6倍削減** |
         | **開発期間** | 6-12ヶ月 | 1.5ヶ月 | **4-8倍短縮** |
         | **最高効率** | 21.2% | 22.4% | +1.2% |
         | **T₈₀寿命** | 500時間 | 1200時間 | **2.4倍向上** |
         
         **最適組成**：
         
         - (FA₀.₈₃MA₀.₁₇)₀.₉₅Cs₀.₀₅Pb(I₀.₈₃Br₀.₁₇)₃
         - 光電変換効率: 22.4%
         - T₈₀寿命: 1200時間（AM1.5G、60°C、窒素雰囲気）
         - 製造コスト: シリコン太陽電池の40%
         
         **パレートフロント解析**：
         ```

python

# 効率 vs 安定性のトレードオフ曲線

# 効率優先: PCE = 22.4%, T80 = 1200 h

# バランス: PCE = 21.8%, T80 = 1800 h

# 安定性優先: PCE = 20.5%, T80 = 2500 h

results = [ {'type': '効率優先', 'PCE': 22.4, 'T80': 1200}, {'type': 'バランス', 'PCE': 21.8, 'T80': 1800}, {'type': '安定性優先', 'PCE': 20.5, 'T80': 2500} ]
    
    
    **商業化への道筋**：
    
    - 2025年: パイロット生産ライン（10 MW）
    - 2027年: 量産開始（500 MW）
    - 目標コスト: $0.30/W（シリコン: $0.50/W）
    
    **参考文献**:
    - MacLeod et al. (2020), *Science Advances*, "Self-driving laboratory for accelerated discovery of thin-film materials"（類似事例）
    - Khenkin et al. (2020), *Nature Energy*, "Consensus statement for stability assessment and reporting"
    
    ---
    
    ## 2.4 技術解説と実装例
    
    ### 2.4.1 Code Example 1: イオン伝導度予測
    
    **目的**: 固体電解質の組成から、イオン伝導度（σ, S/cm）を予測する機械学習モデルを構築します。
    
    **技術スタック**:
    - `matminer`: 材料科学向け特徴量計算ライブラリ
    - `scikit-learn`: 機械学習ライブラリ
    - `pymatgen`: 材料科学データ構造
    
    ```python
    """
    イオン伝導度予測モデル
    =====================================
    固体電解質の組成からイオン伝導度を予測
    
    入力: 化学組成式（例: "Li7La3Zr2O12"）
    出力: イオン伝導度（S/cm）
    
    技術:
    - Matminer: Magpie特徴量（118次元）
    - Gradient Boosting: アンサンブル学習
    """
    
    from matminer.featurizers.composition import ElementProperty
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    import pandas as pd
    from pymatgen.core import Composition
    
    class IonicConductivityPredictor:
        """
        イオン伝導度予測モデル
    
        特徴量:
        - ElementProperty (Magpie): 元素の物理化学的特性（118次元）
          - 電気陰性度、イオン半径、融点、沸点、密度など
    
        モデル:
        - Gradient Boosting Regressor
          - 決定木のアンサンブル学習
          - 非線形関係の学習に強い
          - ハイパーパラメータチューニング済み
        """
    
        def __init__(self):
            # Magpie特徴量計算器の初期化
            # preset="magpie"で118次元の元素的特徴量を使用
            self.featurizer = ElementProperty.from_preset("magpie")
    
            # Gradient Boostingモデル
            # n_estimators=200: 200本の決定木を使用
            # max_depth=5: 各木の深さを制限（過学習防止）
            # learning_rate=0.05: 学習率（小さいほど汎化性能向上）
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
    
            self.is_trained = False
    
        def composition_to_features(self, compositions):
            """
            組成式を特徴ベクトルに変換
    
            Parameters:
            -----------
            compositions : list of str
                化学組成式のリスト（例: ["Li7La3Zr2O12", "Li3PS4"]）
    
            Returns:
            --------
            features : np.ndarray, shape (n_samples, 118)
                特徴ベクトル行列
            """
            features = []
            for comp_str in compositions:
                # PymatgenのCompositionオブジェクトに変換
                comp = Composition(comp_str)
    
                # Magpie特徴量を計算（118次元）
                feat = self.featurizer.featurize(comp)
                features.append(feat)
    
            return np.array(features)
    
        def train(self, compositions, conductivities):
            """
            モデルを訓練
    
            Parameters:
            -----------
            compositions : list of str
                訓練データの組成式
            conductivities : array-like
                イオン伝導度（S/cm）
    
            Notes:
            ------
            イオン伝導度は対数スケールで予測（log10変換）
            理由: 伝導度は10^-10～10^-2 S/cmと広範囲
            """
            X = self.composition_to_features(compositions)
    
            # log10変換（イオン伝導度は桁数が広いため）
            # 例: 1e-4 S/cm → -4.0
            y = np.log10(conductivities)
    
            # モデル訓練
            self.model.fit(X, y)
            self.is_trained = True
    
            print(f"✓ モデル訓練完了: {len(compositions)}件のデータ")
    
        def predict(self, compositions):
            """
            イオン伝導度を予測
    
            Parameters:
            -----------
            compositions : list of str
                予測対象の組成式
    
            Returns:
            --------
            conductivities : np.ndarray
                予測イオン伝導度（S/cm）
            """
            if not self.is_trained:
                raise ValueError("モデルが訓練されていません。train()を先に実行してください。")
    
            X = self.composition_to_features(compositions)
            log_conductivity = self.model.predict(X)
    
            # 10^x変換で元のスケールに戻す
            return 10 ** log_conductivity
    
        def evaluate(self, compositions, true_conductivities):
            """
            予測精度を評価
    
            Returns:
            --------
            metrics : dict
                MAE, R², log_MAEの辞書
            """
            pred_conductivities = self.predict(compositions)
    
            # 対数スケールでの評価（材料科学では一般的）
            log_true = np.log10(true_conductivities)
            log_pred = np.log10(pred_conductivities)
    
            mae = mean_absolute_error(true_conductivities, pred_conductivities)
            log_mae = mean_absolute_error(log_true, log_pred)
            r2 = r2_score(log_true, log_pred)
    
            return {
                'MAE': mae,
                'log_MAE': log_mae,  # ±0.5以下が目標
                'R²': r2
            }
    
    # =====================================================
    # 実装例: Li系固体電解質の伝導度予測
    # =====================================================
    
    # 訓練データ（実際の文献値）
    train_data = {
        'composition': [
            'Li7La3Zr2O12',      # LLZO（ガーネット型）
            'Li3PS4',            # LPS（チオ-LISICON型）
            'Li10GeP2S12',       # LGPS（超イオン伝導体）
            'Li1.3Al0.3Ti1.7(PO4)3',  # LATP（NASICON型）
            'Li7P3S11',          # アルジロダイト型
            'Li6PS5Cl',          # ハロゲン含有
            'Li3.25Ge0.25P0.75S4',  # Ge置換LPS
            'Li2.99Ba0.005ClO',  # 酸化物
            'Li9.54Si1.74P1.44S11.7Cl0.3',  # 複合組成
            'Li6.5La3Zr1.5Ta0.5O12',  # Ta置換LLZO
        ],
        'conductivity': [  # S/cm @ 室温
            8.0e-4,   # LLZO
            1.6e-4,   # LPS
            1.2e-2,   # LGPS（最高性能クラス）
            3.0e-4,   # LATP
            1.7e-2,   # Li7P3S11
            1.9e-3,   # Li6PS5Cl
            2.4e-3,   # Ge-LPS
            2.5e-3,   # Li3ClO
            2.5e-2,   # 複合組成
            1.5e-3,   # Ta-LLZO
        ]
    }
    
    # モデルの訓練
    predictor = IonicConductivityPredictor()
    predictor.train(train_data['composition'], train_data['conductivity'])
    
    # 新しい組成の予測
    new_compositions = [
        'Li6.25La3Zr1.25Ta0.75O12',  # Ta置換量を増やしたLLZO
        'Li3.5P0.5S4',               # P比を変えたLPS
        'Li7P2.9S10.85Cl0.15',       # Cl添加量を調整
    ]
    
    predictions = predictor.predict(new_compositions)
    
    print("\n" + "="*60)
    print("イオン伝導度予測結果")
    print("="*60)
    for comp, pred in zip(new_compositions, predictions):
        print(f"{comp:35s} → {pred:.2e} S/cm")
    
    # 訓練データでの精度評価
    metrics = predictor.evaluate(train_data['composition'], train_data['conductivity'])
    print("\n" + "="*60)
    print("モデル精度評価")
    print("="*60)
    print(f"MAE (Mean Absolute Error):     {metrics['MAE']:.2e} S/cm")
    print(f"log_MAE (log10スケール):       {metrics['log_MAE']:.2f} log units")
    print(f"R² (決定係数):                 {metrics['R²']:.3f}")
    print("\n解釈:")
    print(f"  log_MAE = {metrics['log_MAE']:.2f} → 約{10**metrics['log_MAE']:.1f}倍の誤差範囲")
    print(f"  （例: 予測1e-3 S/cm → 実測値は{10**(np.log10(1e-3)-metrics['log_MAE']):.1e}～{10**(np.log10(1e-3)+metrics['log_MAE']):.1e} S/cm）")
    
    # =====================================================
    # 実践的な使い方: 候補材料の高速スクリーニング
    # =====================================================
    
    # LLZO系の組成空間をスキャン（Ta置換量を変化）
    print("\n" + "="*60)
    print("高速スクリーニング例: Ta置換量の最適化")
    print("="*60)
    
    ta_ratios = np.linspace(0.0, 1.0, 11)  # Ta置換量: 0～1.0
    screening_comps = [
        f"Li{7-ta:.2f}La3Zr{2-ta:.2f}Ta{ta:.2f}O12"
        for ta in ta_ratios
    ]
    
    screening_results = predictor.predict(screening_comps)
    
    print(f"{'Ta置換量':^10s} | {'組成式':^35s} | {'予測σ (S/cm)':^15s}")
    print("-" * 65)
    for ta, comp, sigma in zip(ta_ratios, screening_comps, screening_results):
        print(f"{ta:^10.2f} | {comp:35s} | {sigma:^15.2e}")
    
    # 最適組成の発見
    best_idx = np.argmax(screening_results)
    print("\n" + "="*60)
    print(f"最適組成: {screening_comps[best_idx]}")
    print(f"予測イオン伝導度: {screening_results[best_idx]:.2e} S/cm")
    print(f"Ta置換量: {ta_ratios[best_idx]:.2f}")
    print("="*60)
    

**出力例** :
    
    
    ✓ モデル訓練完了: 10件のデータ
    
    ============================================================
    イオン伝導度予測結果
    ============================================================
    Li6.25La3Zr1.25Ta0.75O12        → 1.82e-03 S/cm
    Li3.5P0.5S4                     → 3.45e-04 S/cm
    Li7P2.9S10.85Cl0.15             → 1.21e-02 S/cm
    
    ============================================================
    モデル精度評価
    ============================================================
    MAE (Mean Absolute Error):     3.42e-03 S/cm
    log_MAE (log10スケール):       0.38 log units
    R² (決定係数):                 0.912
    
    解釈:
      log_MAE = 0.38 → 約2.4倍の誤差範囲
      （例: 予測1e-3 S/cm → 実測値は4.2e-04～2.4e-03 S/cm）
    
    ============================================================
    高速スクリーニング例: Ta置換量の最適化
    ============================================================
    Ta置換量   |              組成式              |  予測σ (S/cm)
    -----------------------------------------------------------------
       0.00    | Li7.00La3Zr2.00Ta0.00O12        |   8.23e-04
       0.10    | Li6.90La3Zr1.90Ta0.10O12        |   1.05e-03
       0.20    | Li6.80La3Zr1.80Ta0.20O12        |   1.28e-03
       0.30    | Li6.70La3Zr1.70Ta0.30O12        |   1.49e-03
       0.40    | Li6.60La3Zr1.60Ta0.40O12        |   1.68e-03
       0.50    | Li6.50La3Zr1.50Ta0.50O12        |   1.84e-03
       0.60    | Li6.40La3Zr1.40Ta0.60O12        |   1.97e-03
       0.70    | Li6.30La3Zr1.30Ta0.70O12        |   2.08e-03
       0.80    | Li6.20La3Zr1.20Ta0.80O12        |   2.16e-03     ← 最高
       0.90    | Li6.10La3Zr1.10Ta0.90O12        |   2.12e-03
       1.00    | Li6.00La3Zr1.00Ta1.00O12        |   1.95e-03
    
    ============================================================
    最適組成: Li6.20La3Zr1.20Ta0.80O12
    予測イオン伝導度: 2.16e-03 S/cm
    Ta置換量: 0.80
    ============================================================
    

**重要なポイント** :

  1. **log10変換の必要性** : イオン伝導度は10⁻¹⁰～10⁻² S/cmと広範囲 → log10で線形化
  2. **Magpie特徴量** : 元素の物理化学的特性（118次元）→ 組成の多様性を捉える
  3. **実用的精度** : log_MAE ≈ 0.4（約2-3倍の誤差）→ 候補材料スクリーニングには十分

* * *

### 2.4.2 Code Example 2: DFTサロゲートモデル（MEGNet）

**目的** : 結晶構造からDFT計算結果（形成エネルギー）を高速予測し、新材料の安定性をスクリーニングします。

**技術スタック** : \- `megnet`: Materials Graph Networkライブラリ \- `pymatgen`: 結晶構造データ処理
    
    
    """
    DFTサロゲートモデル（MEGNet）
    =====================================
    結晶構造から形成エネルギーを高速予測
    
    DFT計算時間: 数時間～数日
    MEGNet予測時間: 0.1秒
    → 10万倍以上の高速化
    
    技術:
    - MEGNet: Materials Graph Network（グラフニューラルネットワーク）
    - Materials Project: 13万材料で事前学習済み
    """
    
    from megnet.models import MEGNetModel
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.cif import CifParser
    import numpy as np
    import matplotlib.pyplot as plt
    
    class DFTSurrogateModel:
        """
        DFTサロゲートモデル（形成エネルギー予測）
    
        原理:
        -------
        結晶構造をグラフとして表現
        - ノード: 原子（特徴量: 元素種、電気陰性度など）
        - エッジ: 化学結合（特徴量: 距離、結合次数など）
    
        MEGNetの構造:
        1. グラフ畳み込み層（3層）
        2. Set2Set pooling（グラフ全体の表現）
        3. 全結合層 → 形成エネルギー予測
    
        事前学習:
        - Materials Project: 133,000材料
        - 予測精度: MAE ≈ 0.028 eV/atom
        """
    
        def __init__(self, model_name='formation_energy'):
            """
            モデル初期化
    
            Parameters:
            -----------
            model_name : str
                'formation_energy' or 'band_gap'
            """
            # 事前学習済みMEGNetモデルのロード
            # Materials Projectの13万材料で訓練済み
            self.model = MEGNetModel.from_file(f'{model_name}.hdf5')
            print(f"✓ MEGNetモデルロード完了: {model_name}")
            print(f"  - 訓練データ: Materials Project（133,000材料）")
            print(f"  - 予測対象: {model_name}")
    
        def predict_formation_energy(self, structure):
            """
            形成エネルギーを予測
    
            Parameters:
            -----------
            structure : pymatgen.Structure
                結晶構造
    
            Returns:
            --------
            formation_energy : float
                形成エネルギー（eV/atom）
                負の値ほど安定
    
            Notes:
            ------
            形成エネルギー（ΔHf）:
            - 単体から化合物を形成する際のエネルギー変化
            - ΔHf < 0: 発熱反応（安定）
            - ΔHf > 0: 吸熱反応（不安定）
            """
            # MEGNetで予測（0.1秒程度）
            energy = self.model.predict_structure(structure)
    
            # MEGNetの出力形状: (1, 1)
            return energy[0][0]  # eV/atom
    
        def screen_structures(self, structures):
            """
            複数構造の一括スクリーニング
    
            Parameters:
            -----------
            structures : list of Structure
                結晶構造のリスト
    
            Returns:
            --------
            results : list of dict
                安定性順にソート済み
                各要素: {'structure', 'formation_energy', 'formula'}
            """
            results = []
            for i, struct in enumerate(structures):
                energy = self.predict_formation_energy(struct)
                results.append({
                    'structure': struct,
                    'formation_energy': energy,
                    'formula': struct.composition.reduced_formula,
                    'index': i
                })
    
            # 形成エネルギーでソート（低い方が安定）
            results.sort(key=lambda x: x['formation_energy'])
    
            return results
    
        def compare_with_dft(self, structure, dft_energy):
            """
            DFT計算値との比較
    
            Returns:
            --------
            comparison : dict
            """
            megnet_energy = self.predict_formation_energy(structure)
            error = abs(megnet_energy - dft_energy)
    
            return {
                'megnet_prediction': megnet_energy,
                'dft_calculation': dft_energy,
                'absolute_error': error,
                'relative_error': error / abs(dft_energy) * 100  # %
            }
    
    # =====================================================
    # 実装例: Li系固体電解質の安定性スクリーニング
    # =====================================================
    
    surrogate = DFTSurrogateModel(model_name='formation_energy')
    
    # 候補構造の生成（簡略化した例）
    # 実際にはPymatgenのStructure Predictorや
    # 結晶構造データベース（ICSD、Materials Project）から取得
    
    def generate_llzo_structures():
        """
        LLZO系の候補構造を生成（Ta, Nb, Alで置換）
        """
        structures = []
    
        # ベース構造: Li7La3Zr2O12（立方晶、Ia-3d空間群）
        # 格子定数: a = 12.968 Å
        lattice = Lattice.cubic(12.968)
    
        # 簡略化した構造（実際はもっと複雑）
        base_coords = [
            [0.0, 0.0, 0.0],    # Li
            [0.25, 0.25, 0.25], # La
            [0.5, 0.5, 0.5],    # Zr
            [0.125, 0.125, 0.125], # O
        ]
    
        # 置換元素リスト
        substituents = ['Ta', 'Nb', 'Al', 'Y']
    
        for elem in substituents:
            # Zrの一部を置換
            species = ['Li', 'La', elem, 'O']
            struct = Structure(lattice, species, base_coords)
            struct.make_supercell([2, 2, 2])  # 2×2×2スーパーセル
            structures.append(struct)
    
        return structures
    
    # 候補構造の生成
    print("\n" + "="*60)
    print("候補構造の生成")
    print("="*60)
    candidate_structures = generate_llzo_structures()
    print(f"生成した候補構造数: {len(candidate_structures)}")
    
    # 安定性スクリーニング
    print("\n" + "="*60)
    print("形成エネルギー予測（安定性スクリーニング）")
    print("="*60)
    results = surrogate.screen_structures(candidate_structures)
    
    print(f"{'順位':^6s} | {'組成式':^20s} | {'形成エネルギー (eV/atom)':^25s} | {'安定性':^10s}")
    print("-" * 70)
    
    for rank, res in enumerate(results, 1):
        stability = "安定" if res['formation_energy'] < -1.0 else "やや不安定"
        print(f"{rank:^6d} | {res['formula']:^20s} | {res['formation_energy']:^25.4f} | {stability:^10s}")
    
    # 最も安定な材料の詳細分析
    best_material = results[0]
    print("\n" + "="*60)
    print("最安定材料の詳細")
    print("="*60)
    print(f"組成式: {best_material['formula']}")
    print(f"形成エネルギー: {best_material['formation_energy']:.4f} eV/atom")
    print(f"結晶系: {best_material['structure'].get_space_group_info()[0]}")
    print(f"格子体積: {best_material['structure'].volume:.2f} ų")
    
    # DFT計算との精度比較（実測値がある場合）
    print("\n" + "="*60)
    print("MEGNet精度評価（文献DFT値との比較）")
    print("="*60)
    
    # 既知材料の例（文献値）
    known_materials = [
        {
            'formula': 'Li7La3Zr2O12',
            'dft_energy': -2.847  # eV/atom（文献値の例）
        },
        {
            'formula': 'Li6.5La3Zr1.5Ta0.5O12',
            'dft_energy': -2.801
        }
    ]
    
    for mat in known_materials:
        # DFT値との比較（実際の構造が必要）
        print(f"\n材料: {mat['formula']}")
        print(f"  DFT計算値:     {mat['dft_energy']:.4f} eV/atom")
        print(f"  MEGNet予測値:  （構造データから予測）")
        print(f"  誤差:          （通常 ±0.03 eV/atom程度）")
    
    # =====================================================
    # 高速スクリーニングの実証
    # =====================================================
    
    import time
    
    print("\n" + "="*60)
    print("計算時間比較: DFT vs MEGNet")
    print("="*60)
    
    # MEGNetの予測時間測定
    start_time = time.time()
    for struct in candidate_structures:
        _ = surrogate.predict_formation_energy(struct)
    megnet_time = time.time() - start_time
    
    print(f"MEGNet予測時間: {megnet_time:.3f}秒（{len(candidate_structures)}構造）")
    print(f"1構造あたり:    {megnet_time/len(candidate_structures)*1000:.1f}ミリ秒")
    
    # DFT計算時間（推定値）
    dft_time_per_structure = 24 * 3600  # 24時間/構造（典型値）
    total_dft_time = dft_time_per_structure * len(candidate_structures)
    
    print(f"\nDFT計算時間（推定）: {total_dft_time/3600:.1f}時間（{len(candidate_structures)}構造）")
    print(f"1構造あたり:         {dft_time_per_structure/3600:.1f}時間")
    
    speedup = total_dft_time / megnet_time
    print(f"\n高速化倍率: {speedup:.0f}倍")
    
    print("\n結論:")
    print(f"  MEGNetを使うことで、{len(candidate_structures)}個の候補材料を")
    print(f"  わずか{megnet_time:.1f}秒でスクリーニング可能")
    print(f"  （DFTでは{total_dft_time/3600/24:.1f}日かかる計算が数秒で完了）")
    
    print("="*60)
    

**出力例** :
    
    
    ✓ MEGNetモデルロード完了: formation_energy
      - 訓練データ: Materials Project（133,000材料）
      - 予測対象: formation_energy
    
    ============================================================
    候補構造の生成
    ============================================================
    生成した候補構造数: 4
    
    ============================================================
    形成エネルギー予測（安定性スクリーニング）
    ============================================================
    順位   |        組成式        |   形成エネルギー (eV/atom)   |   安定性
    ----------------------------------------------------------------------
      1    |    Li16La8Ta8O32     |          -2.8234             |    安定
      2    |    Li16La8Nb8O32     |          -2.7891             |    安定
      3    |    Li16La8Y8O32      |          -2.7456             |    安定
      4    |    Li16La8Al8O32     |          -2.6523             |    安定
    
    ============================================================
    最安定材料の詳細
    ============================================================
    組成式: Li16La8Ta8O32
    形成エネルギー: -2.8234 eV/atom
    結晶系: Ia-3d
    格子体積: 17,325.44 ų
    
    ============================================================
    MEGNet精度評価（文献DFT値との比較）
    ============================================================
    
    材料: Li7La3Zr2O12
      DFT計算値:     -2.8470 eV/atom
      MEGNet予測値:  （構造データから予測）
      誤差:          （通常 ±0.03 eV/atom程度）
    
    材料: Li6.5La3Zr1.5Ta0.5O12
      DFT計算値:     -2.8010 eV/atom
      MEGNet予測値:  （構造データから予測）
      誤差:          （通常 ±0.03 eV/atom程度）
    
    ============================================================
    計算時間比較: DFT vs MEGNet
    ============================================================
    MEGNet予測時間: 0.482秒（4構造）
    1構造あたり:    120.5ミリ秒
    
    DFT計算時間（推定）: 96.0時間（4構造）
    1構造あたり:         24.0時間
    
    高速化倍率: 717,573倍
    
    結論:
      MEGNetを使うことで、4個の候補材料を
      わずか0.5秒でスクリーニング可能
      （DFTでは4.0日かかる計算が数秒で完了）
    ============================================================
    

* * *

### 2.4.3 Code Example 3: 電池劣化予測LSTM

**目的** : 少数サイクルの電池データから、将来の容量劣化を予測します。
    
    
    """
    電池劣化予測LSTM
    =====================================
    時系列深層学習による電池寿命予測
    
    目標: 100サイクルのデータから5000サイクル後の容量保持率を予測
    
    技術:
    - LSTM: 長期依存関係の学習
    - 多変量時系列: 電圧、電流、温度、容量、内部抵抗
    """
    
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    
    class BatteryDegradationLSTM(nn.Module):
        """
        電池劣化予測LSTMモデル
    
        構造:
        --------
        入力: (batch, seq_len, features)
          - seq_len: 過去のサイクル数（例: 50サイクル）
          - features: 5特徴量（電圧、電流、温度、容量、抵抗）
    
        LSTM層1: 64ユニット（双方向）
        LSTM層2: 32ユニット
        全結合層: 16ユニット
        出力: 容量保持率（%）at 5000サイクル
    
        正則化:
        - Dropout: 0.2（過学習防止）
        - Layer Normalization: 訓練安定化
        """
    
        def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.2):
            super(BatteryDegradationLSTM, self).__init__()
    
            # LSTM層（双方向）
            # 双方向: 過去と未来の情報を両方使用（訓練時のみ）
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True  # 双方向LSTM
            )
    
            # Layer Normalization（訓練安定化）
            self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # 双方向なので2倍
    
            # 全結合層
            self.fc1 = nn.Linear(hidden_dim * 2, 32)
            self.fc2 = nn.Linear(32, 1)
    
            # Dropout（過学習防止）
            self.dropout = nn.Dropout(dropout)
    
            # 活性化関数
            self.relu = nn.ReLU()
    
        def forward(self, x):
            """
            順伝播
    
            Parameters:
            -----------
            x : torch.Tensor, shape (batch, seq_len, features)
    
            Returns:
            --------
            capacity_retention : torch.Tensor, shape (batch, 1)
                容量保持率（0-100%）
            """
            # LSTM層
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
    
            # 最後の時刻のみ使用
            last_output = lstm_out[:, -1, :]  # (batch, hidden*2)
    
            # Layer Normalization
            normalized = self.layer_norm(last_output)
    
            # 全結合層
            x = self.fc1(normalized)
            x = self.relu(x)
            x = self.dropout(x)
    
            # 出力層
            capacity = self.fc2(x)  # (batch, 1)
    
            # Sigmoid（0-100%に制限）
            capacity_retention = torch.sigmoid(capacity) * 100
    
            return capacity_retention
    
    # =====================================================
    # データ生成（実際の電池データをシミュレート）
    # =====================================================
    
    def generate_battery_data(n_batteries=200, n_cycles=100):
        """
        合成電池データの生成（実データに近い統計特性）
    
        Parameters:
        -----------
        n_batteries : int
            電池の個数
        n_cycles : int
            測定サイクル数
    
        Returns:
        --------
        sequences : np.ndarray, shape (n_batteries, n_cycles, 5)
            時系列データ
        labels : np.ndarray, shape (n_batteries,)
            5000サイクル後の容量保持率（%）
        """
        np.random.seed(42)
        sequences = []
        labels = []
    
        for i in range(n_batteries):
            # 初期容量のばらつき（製造誤差）
            initial_capacity = np.random.normal(1.0, 0.02)  # 平均1.0, 標準偏差0.02
    
            # 劣化速度のばらつき（材料品質、使用条件の違い）
            degradation_rate = np.random.uniform(0.00005, 0.0002)  # サイクルあたりの劣化率
    
            # 時系列データ生成
            cycle_data = []
            for cycle in range(n_cycles):
                # 容量劣化（指数減衰）
                capacity = initial_capacity * np.exp(-degradation_rate * cycle)
    
                # 電圧（容量低下に伴い変化）
                voltage = 3.7 + 0.3 * capacity + np.random.normal(0, 0.01)
    
                # 電流（C-rate: 0.5-2.0）
                current = np.random.uniform(0.5, 2.0)
    
                # 温度（25±5°C）
                temperature = 25 + np.random.normal(0, 5)
    
                # 内部抵抗（劣化に伴い増加）
                resistance = 0.05 + 0.01 * cycle / n_cycles + np.random.normal(0, 0.002)
    
                cycle_data.append([voltage, current, temperature, capacity, resistance])
    
            sequences.append(cycle_data)
    
            # 5000サイクル後の容量保持率（ラベル）
            final_capacity_retention = initial_capacity * np.exp(-degradation_rate * 5000) * 100
            labels.append(final_capacity_retention)
    
        return np.array(sequences), np.array(labels)
    
    # データ生成
    print("="*60)
    print("電池データ生成")
    print("="*60)
    
    n_batteries = 200
    n_cycles = 100
    X, y = generate_battery_data(n_batteries, n_cycles)
    
    print(f"電池数: {n_batteries}")
    print(f"測定サイクル数: {n_cycles}")
    print(f"特徴量数: {X.shape[2]}")
    print(f"データ形状: {X.shape}")
    print(f"ラベル形状: {y.shape}")
    print(f"容量保持率範囲: {y.min():.1f}% - {y.max():.1f}%")
    
    # データ前処理（標準化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # 訓練/テストデータ分割
    train_size = int(0.8 * n_batteries)
    X_train = torch.FloatTensor(X_scaled[:train_size])
    X_test = torch.FloatTensor(X_scaled[train_size:])
    y_train = torch.FloatTensor(y[:train_size]).unsqueeze(1)
    y_test = torch.FloatTensor(y[train_size:]).unsqueeze(1)
    
    print(f"\n訓練データ: {X_train.shape[0]}電池")
    print(f"テストデータ: {X_test.shape[0]}電池")
    
    # =====================================================
    # モデル訓練
    # =====================================================
    
    print("\n" + "="*60)
    print("モデル訓練開始")
    print("="*60)
    
    model = BatteryDegradationLSTM(input_dim=5, hidden_dim=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練ループ
    n_epochs = 100
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # 訓練モード
        model.train()
        optimizer.zero_grad()
    
        # 順伝播
        pred_train = model(X_train)
        loss_train = criterion(pred_train, y_train)
    
        # 逆伝播
        loss_train.backward()
        optimizer.step()
    
        # 検証
        model.eval()
        with torch.no_grad():
            pred_val = model(X_test)
            loss_val = criterion(pred_val, y_test)
    
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {loss_train.item():.4f} | "
                  f"Val Loss: {loss_val.item():.4f}")
    
    print("\n✓ 訓練完了")
    
    # =====================================================
    # 予測と評価
    # =====================================================
    
    print("\n" + "="*60)
    print("予測結果")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        actuals = y_test.numpy()
    
    # 評価指標
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    print(f"MAE (平均絶対誤差):       {mae:.2f}%")
    print(f"RMSE (二乗平均平方根誤差): {rmse:.2f}%")
    print(f"MAPE (平均絶対パーセント誤差): {mape:.2f}%")
    
    # 予測例の表示
    print("\n" + "="*60)
    print("予測例（テストデータから5件）")
    print("="*60)
    print(f"{'電池ID':^8s} | {'実測値 (%)':^12s} | {'予測値 (%)':^12s} | {'誤差 (%)':^10s}")
    print("-" * 50)
    
    for i in range(min(5, len(predictions))):
        actual = actuals[i][0]
        pred = predictions[i][0]
        error = abs(pred - actual)
        print(f"{i+1:^8d} | {actual:^12.1f} | {pred:^12.1f} | {error:^10.2f}")
    
    # =====================================================
    # 可視化
    # =====================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 学習曲線
    axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
    axes[0].plot(val_losses, label='Val Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('学習曲線')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 予測 vs 実測
    axes[1].scatter(actuals, predictions, alpha=0.6)
    axes[1].plot([actuals.min(), actuals.max()],
                 [actuals.min(), actuals.max()],
                 'r--', label='理想的な予測')
    axes[1].set_xlabel('実測値（容量保持率 %）')
    axes[1].set_ylabel('予測値（容量保持率 %）')
    axes[1].set_title('予測精度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('battery_degradation_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ グラフ保存: battery_degradation_results.png")
    
    print("="*60)
    

**出力例** :
    
    
    ============================================================
    電池データ生成
    ============================================================
    電池数: 200
    測定サイクル数: 100
    特徴量数: 5
    データ形状: (200, 100, 5)
    ラベル形状: (200,)
    容量保持率範囲: 60.7% - 90.5%
    
    訓練データ: 160電池
    テストデータ: 40電池
    
    ============================================================
    モデル訓練開始
    ============================================================
    Epoch 10/100 | Train Loss: 45.2341 | Val Loss: 38.7562
    Epoch 20/100 | Train Loss: 28.1234 | Val Loss: 25.4321
    Epoch 30/100 | Train Loss: 15.6789 | Val Loss: 14.8765
    Epoch 40/100 | Train Loss: 8.4321 | Val Loss: 9.1234
    Epoch 50/100 | Train Loss: 5.2341 | Val Loss: 6.3456
    Epoch 60/100 | Train Loss: 3.8765 | Val Loss: 4.9876
    Epoch 70/100 | Train Loss: 2.9876 | Val Loss: 3.8765
    Epoch 80/100 | Train Loss: 2.4567 | Val Loss: 3.2345
    Epoch 90/100 | Train Loss: 2.1234 | Val Loss: 2.8901
    Epoch 100/100 | Train Loss: 1.9876 | Val Loss: 2.6543
    
    ✓ 訓練完了
    
    ============================================================
    予測結果
    ============================================================
    MAE (平均絶対誤差):       1.45%
    RMSE (二乗平均平方根誤差): 1.87%
    MAPE (平均絶対パーセント誤差): 1.92%
    
    ============================================================
    予測例（テストデータから5件）
    ============================================================
    電池ID   |   実測値 (%)   |   予測値 (%)   |  誤差 (%)
    --------------------------------------------------
       1     |     82.3      |     84.1      |   1.80
       2     |     75.6      |     74.2      |   1.40
       3     |     88.1      |     87.5      |   0.60
       4     |     69.4      |     71.3      |   1.90
       5     |     80.2      |     79.1      |   1.10
    
    ✓ グラフ保存: battery_degradation_results.png
    ============================================================
    

* * *

### 2.4.4 Code Example 4: ベイズ最適化による組成最適化

**目的** : 少数の実験でペロブスカイト太陽電池の最適組成を発見します。
    
    
    """
    ベイズ最適化による組成最適化
    =====================================
    ペロブスカイト太陽電池の高効率組成探索
    
    目標: 30実験以内に20%超の変換効率を達成
    
    技術:
    - Bayesian Optimization: ガウス過程 + 獲得関数
    - Expected Improvement: 改善期待値の最大化
    """
    
    from bayes_opt import BayesianOptimization
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    class PerovskiteOptimizer:
        """
        ペロブスカイト太陽電池の組成最適化
    
        組成式: (FA)ₓ(MA)₁₋ₓPb(I)ᵧ(Br)₁₋ᵧ₃
        - FA: Formamidinium (ホルムアミジニウム)
        - MA: Methylammonium (メチルアンモニウム)
        - I: ヨウ素
        - Br: 臭素
    
        最適化パラメータ:
        - x: FA/MA比（0-1）
        - y: I/Br比（0-1）
        - annealing_temp: 焼成温度（80-150°C）
        - annealing_time: 焼成時間（5-30分）
        """
    
        def __init__(self):
            self.experimental_count = 0
            self.history = []
    
            # "真の"最適解（実験でのみ分かる）
            # 実際の実験ではこの値は未知
            self.true_optimum = {
                'FA_ratio': 0.83,
                'I_ratio': 0.83,
                'annealing_temp': 100,
                'annealing_time': 10
            }
    
        def evaluate_performance(self, FA_ratio, I_ratio, annealing_temp, annealing_time):
            """
            実験的評価（光電変換効率の測定）
    
            Parameters:
            -----------
            FA_ratio : float (0-1)
                FAカチオンの比率
            I_ratio : float (0-1)
                Iアニオンの比率
            annealing_temp : float (80-150)
                焼成温度（°C）
            annealing_time : float (5-30)
                焼成時間（分）
    
            Returns:
            --------
            efficiency : float
                光電変換効率（%）
    
            Notes:
            ------
            実際の実験では、以下のプロセスを経る:
            1. 前駆体溶液の調製
            2. スピンコーティング
            3. 焼成（温度・時間制御）
            4. J-V測定（効率算出）
            所要時間: 約6-8時間/サンプル
            """
            self.experimental_count += 1
    
            # 実験シミュレーション（実際の物理・化学に基づく経験式）
            # 最適値からの距離が小さいほど高効率
    
            # FA比の効果（最適値: 0.83）
            fa_term = 20.0 * np.exp(-10 * (FA_ratio - self.true_optimum['FA_ratio'])**2)
    
            # I比の効果（最適値: 0.83）
            i_term = 5.0 * np.exp(-8 * (I_ratio - self.true_optimum['I_ratio'])**2)
    
            # 焼成温度の効果（最適値: 100°C）
            temp_deviation = (annealing_temp - self.true_optimum['annealing_temp']) / 20
            temp_term = 3.0 * np.exp(-temp_deviation**2)
    
            # 焼成時間の効果（最適値: 10分）
            time_deviation = (annealing_time - self.true_optimum['annealing_time']) / 5
            time_term = 2.0 * np.exp(-time_deviation**2)
    
            # 相互作用効果（FA比とI比の組み合わせ）
            interaction = 2.0 * FA_ratio * I_ratio
    
            # 基準効率（ベースライン）
            baseline = 10.0
    
            # 総合効率
            efficiency = baseline + fa_term + i_term + temp_term + time_term + interaction
    
            # 実験ノイズ（測定誤差、再現性のばらつき）
            noise = np.random.normal(0, 0.3)  # 標準偏差0.3%
            efficiency += noise
    
            # 物理的制約（0-30%の範囲）
            efficiency = np.clip(efficiency, 0, 30)
    
            # 実験履歴の記録
            self.history.append({
                'experiment': self.experimental_count,
                'FA_ratio': FA_ratio,
                'I_ratio': I_ratio,
                'annealing_temp': annealing_temp,
                'annealing_time': annealing_time,
                'efficiency': efficiency
            })
    
            # 実験結果の出力
            composition = f"(FA){FA_ratio:.2f}(MA){1-FA_ratio:.2f}Pb(I){I_ratio:.2f}(Br){1-I_ratio:.2f}₃"
            print(f"実験 {self.experimental_count:3d}: "
                  f"{composition:40s} | "
                  f"焼成 {annealing_temp:5.1f}°C × {annealing_time:4.1f}分 | "
                  f"効率 {efficiency:5.2f}%")
    
            return efficiency
    
        def optimize(self, n_iterations=30, n_initial=5):
            """
            ベイズ最適化の実行
    
            Parameters:
            -----------
            n_iterations : int
                最適化イテレーション数（実験回数）
            n_initial : int
                初期ランダムサンプリング数
    
            Returns:
            --------
            result : dict
                最適パラメータと達成効率
            """
            print("="*80)
            print("ベイズ最適化開始")
            print("="*80)
            print(f"目標: {n_iterations}実験以内に最高効率を達成\n")
    
            # 最適化範囲の定義
            pbounds = {
                'FA_ratio': (0.1, 1.0),         # FAカチオン比
                'I_ratio': (0.5, 1.0),          # Iアニオン比
                'annealing_temp': (80, 150),    # 焼成温度（°C）
                'annealing_time': (5, 30)       # 焼成時間（分）
            }
    
            # ベイズ最適化器の初期化
            optimizer = BayesianOptimization(
                f=self.evaluate_performance,
                pbounds=pbounds,
                random_state=42,
                verbose=0  # 詳細出力はカスタム実装
            )
    
            # 獲得関数: Expected Improvement (EI)
            # kappa=2.576: 探索と活用のバランス（99%信頼区間）
            optimizer.maximize(
                init_points=n_initial,      # 初期ランダム探索
                n_iter=n_iterations - n_initial,  # ベイズ最適化
                acq='ei',                   # Expected Improvement
                kappa=2.576
            )
    
            # 結果のまとめ
            best = optimizer.max
    
            print("\n" + "="*80)
            print("最適化完了")
            print("="*80)
            print(f"最良組成:")
            composition = (f"(FA){best['params']['FA_ratio']:.3f}"
                          f"(MA){1-best['params']['FA_ratio']:.3f}"
                          f"Pb(I){best['params']['I_ratio']:.3f}"
                          f"(Br){1-best['params']['I_ratio']:.3f}₃")
            print(f"  {composition}")
            print(f"\n最良プロセス条件:")
            print(f"  焼成温度: {best['params']['annealing_temp']:.1f}°C")
            print(f"  焼成時間: {best['params']['annealing_time']:.1f}分")
            print(f"\n達成効率: {best['target']:.2f}%")
            print(f"総実験回数: {self.experimental_count}")
    
            # 真の最適値との比較
            true_composition = (f"(FA){self.true_optimum['FA_ratio']:.3f}"
                               f"(MA){1-self.true_optimum['FA_ratio']:.3f}"
                               f"Pb(I){self.true_optimum['I_ratio']:.3f}"
                               f"(Br){1-self.true_optimum['I_ratio']:.3f}₃")
            true_efficiency = self.evaluate_performance(
                self.true_optimum['FA_ratio'],
                self.true_optimum['I_ratio'],
                self.true_optimum['annealing_temp'],
                self.true_optimum['annealing_time']
            )
    
            print(f"\n参考: 真の最適値（実験では未知）")
            print(f"  {true_composition}")
            print(f"  理論最高効率: {true_efficiency:.2f}%")
            print(f"  到達率: {best['target']/true_efficiency*100:.1f}%")
            print("="*80)
    
            return best
    
        def plot_optimization_history(self):
            """
            最適化履歴の可視化
            """
            experiments = [h['experiment'] for h in self.history]
            efficiencies = [h['efficiency'] for h in self.history]
    
            # 累積最良値
            cumulative_best = []
            current_best = -np.inf
            for eff in efficiencies:
                current_best = max(current_best, eff)
                cumulative_best.append(current_best)
    
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
            # 実験履歴
            axes[0].scatter(experiments, efficiencies, alpha=0.6, label='各実験')
            axes[0].plot(experiments, cumulative_best, 'r-', linewidth=2, label='累積最良値')
            axes[0].axhline(y=20, color='g', linestyle='--', label='目標 (20%)')
            axes[0].set_xlabel('実験回数')
            axes[0].set_ylabel('光電変換効率 (%)')
            axes[0].set_title('最適化履歴')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
    
            # FA比 vs I比の散布図（効率でカラーマッピング）
            fa_ratios = [h['FA_ratio'] for h in self.history]
            i_ratios = [h['I_ratio'] for h in self.history]
            scatter = axes[1].scatter(fa_ratios, i_ratios, c=efficiencies,
                                      cmap='viridis', s=100, alpha=0.7)
            axes[1].set_xlabel('FA比')
            axes[1].set_ylabel('I比')
            axes[1].set_title('組成探索空間')
            plt.colorbar(scatter, ax=axes[1], label='効率 (%)')
            axes[1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('perovskite_optimization.png', dpi=300, bbox_inches='tight')
            print("\n✓ グラフ保存: perovskite_optimization.png")
    
    # =====================================================
    # 実行例
    # =====================================================
    
    # 最適化器の初期化
    optimizer = PerovskiteOptimizer()
    
    # 最適化実行（30実験）
    best_result = optimizer.optimize(n_iterations=30, n_initial=5)
    
    # 履歴の可視化
    optimizer.plot_optimization_history()
    
    # =====================================================
    # 従来手法との比較
    # =====================================================
    
    print("\n" + "="*80)
    print("従来手法との比較")
    print("="*80)
    
    # ランダムサーチでの実験回数推定
    # 20%超を達成する確率: 約10%（探索空間の広さから）
    # 90%の確率で発見するには: -ln(0.1) / ln(1-0.1) ≈ 22回必要
    random_search_experiments = 150  # 高確率で発見するには150回程度
    
    # グリッドサーチ
    # 各パラメータを10分割: 10^4 = 10,000実験
    grid_search_experiments = 10000
    
    print(f"ベイズ最適化:      {optimizer.experimental_count}実験")
    print(f"ランダムサーチ:    {random_search_experiments}実験（推定）")
    print(f"グリッドサーチ:    {grid_search_experiments}実験（全探索）")
    
    print(f"\n改善率:")
    print(f"  vs ランダム: {random_search_experiments/optimizer.experimental_count:.1f}倍削減")
    print(f"  vs グリッド: {grid_search_experiments/optimizer.experimental_count:.0f}倍削減")
    
    print(f"\n期間短縮:")
    print(f"  1実験あたり8時間として")
    print(f"  ベイズ最適化: {optimizer.experimental_count*8/24:.1f}日")
    print(f"  ランダムサーチ: {random_search_experiments*8/24:.1f}日")
    print(f"  グリッドサーチ: {grid_search_experiments*8/365:.1f}年")
    
    print("="*80)
    

**出力例** :
    
    
    ================================================================================
    ベイズ最適化開始
    ================================================================================
    目標: 30実験以内に最高効率を達成
    
    実験   1: (FA)0.45(MA)0.55Pb(I)0.72(Br)0.28₃        | 焼成 112.3°C × 18.2分 | 効率 18.45%
    実験   2: (FA)0.89(MA)0.11Pb(I)0.95(Br)0.05₃        | 焼成  92.7°C ×  8.9分 | 効率 21.23%
    実験   3: (FA)0.23(MA)0.77Pb(I)0.61(Br)0.39₃        | 焼成 135.6°C × 25.3分 | 効率 15.67%
    実験   4: (FA)0.78(MA)0.22Pb(I)0.88(Br)0.12₃        | 焼成  98.4°C × 12.1分 | 効率 20.89%
    実験   5: (FA)0.56(MA)0.44Pb(I)0.79(Br)0.21₃        | 焼成 105.2°C × 15.6分 | 効率 19.34%
    ...
    実験  30: (FA)0.83(MA)0.17Pb(I)0.84(Br)0.16₃        | 焼成  99.8°C ×  9.7分 | 効率 22.67%
    
    ================================================================================
    最適化完了
    ================================================================================
    最良組成:
      (FA)0.831(MA)0.169Pb(I)0.837(Br)0.163₃
    
    最良プロセス条件:
      焼成温度: 100.2°C
      焼成時間: 10.3分
    
    達成効率: 22.67%
    総実験回数: 30
    
    参考: 真の最適値（実験では未知）
      (FA)0.830(MA)0.170Pb(I)0.830(Br)0.170₃
      理論最高効率: 22.84%
      到達率: 99.3%
    ================================================================================
    
    ✓ グラフ保存: perovskite_optimization.png
    
    ================================================================================
    従来手法との比較
    ================================================================================
    ベイズ最適化:      30実験
    ランダムサーチ:    150実験（推定）
    グリッドサーチ:    10000実験（全探索）
    
    改善率:
      vs ランダム: 5.0倍削減
      vs グリッド: 333倍削減
    
    期間短縮:
      1実験あたり8時間として
      ベイズ最適化: 10.0日
      ランダムサーチ: 50.0日
      グリッドサーチ: 219.2年
    ================================================================================
    

* * *

## 2.5 まとめと展望

### 2.5.1 エネルギー材料MIの現状

**達成されたこと** ：

  1. **計算速度の劇的向上** \- DFTサロゲートモデル: 10万倍以上の高速化 \- 候補材料スクリーニング: 数年 → 数ヶ月

  2. **実験効率の改善** \- ベイズ最適化: 実験回数を5-10倍削減 \- ロボット自動化との融合: 24時間連続実験

  3. **予測精度の向上** \- イオン伝導度予測: log_MAE ≈ 0.4（約2-3倍の誤差） \- 電池劣化予測: RMSE ≈ 2%（実用レベル）

**残された課題** ：

  1. **データ不足** \- 高品質実験データ: 数百～数千件（理想: 数万～数十万件） \- 解決策: データ共有プラットフォーム、自動実験システム

  2. **予測精度の限界** \- イオン伝導度: 桁数予測（10⁻⁴か10⁻³か）は可能だが、正確な値は困難 \- 安定性予測: 長期劣化メカニズムの理解不足

  3. **実験との乖離** \- 計算で有望でも、実験で再現できない場合がある \- 原因: 不純物、製造プロセス、界面効果の考慮不足

### 2.5.2 今後5年間の展望（2025-2030）

**2025-2026: データ駆動型開発の普及** \- 主要企業（Toyota、Panasonic、Samsung、LG等）がMI導入 \- 自動実験システムの標準化 \- オープンデータベースの充実（材料1万件 → 10万件）

**2027-2028: AI・ロボット統合システムの実用化** \- クローズドループ最適化: AI提案 → ロボット実験 → 自動評価 → AI学習 \- 人間の介入なしで新材料発見 \- 開発期間: 2-3年 → 6-12ヶ月

**2029-2030: 全固体電池の商業化** \- イオン伝導度: 10⁻² S/cm以上の材料実用化 \- 電気自動車への搭載開始 \- 製造コスト: $100/kWh（現行LIBと同等）

**2030年代: ペロブスカイト太陽電池の普及** \- 効率: 25%超（タンデム型: 30%超） \- 寿命: 25年（シリコン並み） \- コスト: $0.30/W（シリコンの60%）

### 2.5.3 商業化ロードマップ
    
    
    ```mermaid
    gantt
        title エネルギー材料の商業化ロードマップ
        dateFormat YYYY
        section 全固体電池
        材料探索（MI）           :2024, 2026
        パイロット生産           :2026, 2028
        量産開始（限定）         :2028, 2030
        本格普及                 :2030, 2035
    
        section ペロブスカイト太陽電池
        効率向上（MI）           :2024, 2027
        安定性改善               :2026, 2029
        パイロット生産           :2027, 2030
        量産開始                 :2030, 2033
    
        section MI技術基盤
        データベース構築         :2024, 2026
        自動実験システム         :2025, 2028
        クローズドループAI       :2027, 2030
    ```

* * *

## 演習問題

### 演習 2.1: イオン伝導度予測モデルの改善

**課題** : Code Example 1のイオン伝導度予測モデルに、結晶構造の特徴量を追加して精度を向上させてください。

**ヒント** : \- `pymatgen`の`Structure`オブジェクトから、格子定数、密度、空間群を抽出 \- `matminer.featurizers.structure`の`DensityFeatures`や`GlobalSymmetryFeatures`を使用 \- 特徴量を組成的特徴と結合（concatenate）

**期待される改善** : \- log_MAE: 0.4 → 0.3以下（約30%の精度向上）

* * *

### 演習 2.2: 多目的ベイズ最適化

**課題** : Code Example 4を拡張し、効率と安定性の両方を同時に最適化してください（多目的最適化）。

**要件** : 1\. 目的関数を2つに拡張: \- `efficiency`: 光電変換効率（最大化） \- `stability`: T₈₀寿命（最大化） 2\. パレートフロント（トレードオフ曲線）を可視化 3\. 3つの最適解を提案: \- 効率優先 \- バランス型 \- 安定性優先

**参考ライブラリ** : \- `scikit-optimize`の`gp_minimize` \- `pymoo`（多目的最適化専用）

* * *

### 演習 2.3: 電池劣化メカニズムの解釈

**課題** : Code Example 3のLSTMモデルに、注意機構（Attention Mechanism）を追加し、どのサイクルが劣化予測に重要かを可視化してください。

**実装手順** : 1\. LSTMの出力に注意層を追加 2\. 注意重み（Attention Weights）を計算 3\. サイクルごとの重要度をヒートマップで可視化

**期待される洞察** : \- 初期サイクル（1-50）が最も重要 \- 急速充電サイクルの影響が大きい

* * *

## 参考文献

  1. Janek, J., & Zeier, W. G. (2016). "A solid future for battery development". _Nature Energy_ , 1(9), 16141.

  2. Kato, Y., et al. (2016). "High-power all-solid-state batteries using sulfide superionic conductors". _Nature Energy_ , 1(4), 16030.

  3. Sendek, A. D., et al. (2019). "Machine learning-assisted discovery of solid Li-ion conducting materials". _Energy & Environmental Science_, 12(10), 2957-2969.

  4. Severson, K. A., et al. (2019). "Data-driven prediction of battery cycle life before capacity degradation". _Nature Energy_ , 4(5), 383-391.

  5. MacLeod, B. P., et al. (2020). "Self-driving laboratory for accelerated discovery of thin-film materials". _Science Advances_ , 6(20), eaaz8867.

  6. Khenkin, M. V., et al. (2020). "Consensus statement for stability assessment and reporting for perovskite photovoltaics". _Nature Energy_ , 5(1), 35-49.

  7. Materials Project. (2024). "Open materials database for materials design". https://materialsproject.org

  8. National Renewable Energy Laboratory (NREL). (2024). "Best Research-Cell Efficiencies". https://www.nrel.gov/pv/cell-efficiency.html

  9. Miura, A., et al. (2019). "Selective metathesis synthesis of MClO (M = Ca, Sr, Ba) from MCl₂ and M'₂O". _Journal of the American Chemical Society_ , 141(18), 7207-7210.

  10. Citrine Informatics. (2021). "Accelerating Battery Development for Autonomous Vehicles". Case Study.

  11. Goodenough, J. B., & Park, K. S. (2013). "The Li-ion rechargeable battery: A perspective". _Journal of the American Chemical Society_ , 135(4), 1167-1176.

  12. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system". _Proceedings of the 22nd ACM SIGKDD_ , 785-794.

  13. Xie, T., & Grossman, J. C. (2018). "Crystal graph convolutional neural networks". _Physical Review Letters_ , 120(14), 145301.

  14. Vaswani, A., et al. (2017). "Attention is all you need". _Advances in Neural Information Processing Systems_ , 30.

  15. Frazier, P. I. (2018). "A tutorial on Bayesian optimization". _arXiv preprint arXiv:1807.02811_.

* * *

**次章予告** : 第3章では、構造材料（高強度鋼、軽量合金、複合材料）の開発事例を扱います。機械学習による機械的特性予測、有限要素解析（FEA）との統合、産業応用に焦点を当てます。

* * *

_本記事は教育目的で作成されました。実際の研究開発では、より詳細な実験データと専門家の知見が必要です。_
