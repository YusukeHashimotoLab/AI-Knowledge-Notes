---
chapter_number: 2
chapter_title: "触媒設計に特化したMI手法"
subtitle: "記述子設計からベイズ最適化まで"
series: "触媒設計MI応用シリーズ"
difficulty: "中級"
reading_time: "30-35分"
code_examples: 2
exercises: 7
mermaid_diagrams: 1
prerequisites:
  - "第1章の内容理解"
  - "機械学習の基礎"
  - "Python（NumPy, scikit-learn）"
learning_objectives:
  basic:
    - "触媒記述子の4タイプを理解し使い分けられる"
    - "Sabatier原理と火山型プロットを説明できる"
  practical:
    - "活性・選択性予測モデルを構築できる"
    - "ベイズ最適化の原理と獲得関数を理解している"
  advanced:
    - "DFT計算とMLの統合手法を実装できる"
    - "Multi-fidelity最適化で計算コストを削減できる"
keywords:
  - "記述子設計"
  - "d-band理論"
  - "Sabatier原理"
  - "ベイズ最適化"
  - "Gaussian Process"
  - "DFT統合"
  - "Catalysis-Hub"
  - "Transfer Learning"
---
# 第2章：触媒設計に特化したMI手法

## 学習目標

1. **記述子設計**：触媒記述子の4タイプを理解し、使い分けられる
2. **予測モデル**：活性・選択性予測モデルの構築手順を説明できる
3. **ベイズ最適化**：原理と応用方法を理解している
4. **DFT統合**：第一原理計算とMLの統合手法を把握している
5. **データベース**：主要な触媒データベースの特徴と使い分けを理解している

---

## 2.1 触媒記述子（Descriptor）

### 2.1.1 記述子の役割

記述子とは、触媒の特性を数値で表現したものです。機械学習モデルの入力として使用されます。

**良い記述子の条件**：
- ✅ 物理的意味が明確
- ✅ 計算が容易
- ✅ 活性と相関がある
- ✅ 一般性がある（異なる反応系にも適用可能）

### 2.1.2 記述子の分類

**1. 電子的記述子（Electronic Descriptors）**

| 記述子 | 定義 | 触媒活性との関係 |
|--------|------|-----------------|
| **d軌道占有数** | 遷移金属のd軌道に存在する電子数 | 吸着エネルギーを決定 |
| **d-bandセンター** | d軌道のエネルギー準位の重心 | 高いほど吸着が強い |
| **仕事関数** | 表面から電子を取り出すのに必要なエネルギー | 電子移動反応に影響 |
| **Bader電荷** | 原子に局在した電荷 | 酸化還元活性に相関 |

**d-band理論（Nørskov）**：
```
吸着エネルギー ∝ d-bandセンター位置

d-bandがフェルミ準位に近い
  → 反結合軌道のoccupation増加
  → 吸着が強い
  → 活性が高い（ただし脱離が遅い）

最適なd-bandセンター：中間値（Sabatier原理）
```

**2. 幾何的記述子（Geometrical Descriptors）**

| 記述子 | 説明 | 例 |
|--------|------|-----|
| **配位数（CN）** | 原子の隣接原子数 | CNが低い（エッジ、コーナー）ほど活性 |
| **原子半径** | 金属原子のサイズ | 格子ひずみと相関 |
| **表面積（BET）** | 触媒の比表面積 | 大きいほど活性サイト増加 |
| **細孔径** | ゼオライトの孔径 | 形状選択性を決定 |
| **結晶面** | (111), (100), (110)など | 活性サイトの密度が異なる |

**3. 組成記述子（Compositional Descriptors）**

| 記述子 | 定義 | 活用例 |
|--------|------|--------|
| **元素組成** | 各元素のモル分率 | 組成最適化 |
| **電気陰性度** | 電子を引き寄せる強さ | 酸化還元活性 |
| **イオン半径** | イオン状態での原子サイズ | 担体との相互作用 |
| **融点** | 金属の融点 | 熱安定性の指標 |

**4. 反応記述子（Reaction Descriptors）**

| 記述子 | 定義 | 用途 |
|--------|------|------|
| **吸着エネルギー** | 分子が表面に吸着する際のエネルギー | 活性の直接指標 |
| **活性化エネルギー** | 反応障壁 | 反応速度の予測 |
| **遷移状態エネルギー** | 遷移状態の安定性 | 律速段階の特定 |

---

## 2.2 Sabatier原理とd-band理論

### 2.2.1 Sabatier原理

**定義**：
最適な触媒は、反応中間体と「ちょうど良い強さ」で相互作用する。

```
吸着が弱すぎる:
  → 反応物が表面に留まらない
  → 活性が低い

吸着が強すぎる:
  → 生成物が表面から脱離できない
  → 活性が低い

最適な吸着強度:
  → 火山型プロット（Volcano Plot）の頂点
```

**火山型プロット**：
```
活性（TOF）
    |
    |        *
    |      /   \
    |     /     \
    |    /       \
    |   /         \
    |  /           \
    |_________________ 吸着エネルギー
   弱い  最適  強い
```

### 2.2.2 スケーリング関係（Scaling Relations）

多くの吸着エネルギーには線形関係があります：

```
E(OH*) = 0.5 * E(O*) + 0.25 eV

E(CHO*) = E(CO*) + 0.8 eV

⇒ 1つの記述子（例：E(O*)）で複数の吸着種を予測可能
⇒ 記述子の次元削減
```

---

## 2.3 活性・選択性予測モデル

### 2.3.1 回帰モデル（活性予測）

**目的**：触媒の活性（TOF、転化率）を予測

**ワークフロー**：
```
1. データ収集
   - 実験データ：活性測定
   - DFTデータ：吸着エネルギー

2. 記述子計算
   - 電子的：d-band center
   - 幾何的：配位数
   - 組成的：元素組成

3. モデル訓練
   - Random Forest
   - Gradient Boosting
   - Neural Network

4. 性能評価
   - R²、RMSE、MAE
   - Cross-validation

5. 予測
   - 未知触媒の活性予測
```

**推奨モデル**：

| モデル | 利点 | 欠点 | 推奨データ数 |
|--------|------|------|-------------|
| **Random Forest** | 解釈可能、安定 | 外挿弱い | 100+ |
| **XGBoost** | 高精度、高速 | ハイパーパラメータ多 | 200+ |
| **Neural Network** | 複雑な関係学習 | 過学習しやすい | 500+ |
| **Gaussian Process** | 不確実性定量化 | スケールしない | <500 |

### 2.3.2 分類モデル（活性触媒のスクリーニング）

**目的**：活性触媒と不活性触媒を分類

**クラス定義**：
```
活性触媒：TOF > 閾値（例：1 s⁻¹）
不活性触媒：TOF ≤ 閾値
```

**評価指標**：
- **Precision**：予測した活性触媒のうち、実際に活性な割合
- **Recall**：実際の活性触媒のうち、正しく予測できた割合
- **F1 Score**：PrecisionとRecallの調和平均
- **ROC-AUC**：分類性能の総合評価

---

## 2.4 ベイズ最適化による触媒探索

### 2.4.1 ベイズ最適化の原理

**目的**：最小の実験回数で最適触媒を発見

**コンポーネント**：
1. **サロゲートモデル**（Surrogate Model）：Gaussian Process
2. **獲得関数**（Acquisition Function）：次に試す候補を選定

**アルゴリズム**：
```
1. 初期実験（10-20サンプル）
   → 組成と活性のデータ取得

2. Gaussian Processでサロゲートモデル訓練
   → 未知組成の活性を予測（平均 + 不確実性）

3. 獲得関数で次実験を選定
   - EI（Expected Improvement）
   - UCB（Upper Confidence Bound）
   - PI（Probability of Improvement）

4. 選定した組成で実験

5. データ更新してステップ2に戻る

6. 収束条件達成まで反復
```

### 2.4.2 獲得関数の比較

| 獲得関数 | 式 | 特徴 | 推奨シーン |
|---------|-----|------|-----------|
| **EI** | E[max(f(x) - f(x⁺), 0)] | バランス型 | 汎用的 |
| **UCB** | μ(x) + β·σ(x) | 探索重視 | 広範囲探索 |
| **PI** | P(f(x) > f(x⁺)) | 活用重視 | 局所最適化 |

**パラメータ調整**：
- β（UCBの探索度）：初期は大きく（3.0）、後期は小さく（1.0）
- ξ（EIのtrade-off）：通常0.01-0.1

### 2.4.3 多目的ベイズ最適化

**目的**：活性と選択性を同時最適化

**パレートフロント**：
```
選択性
    |
100%|      * (理想)
    |    *   *
    |  *       *
    | *         *
    |*___________*___ 活性（TOF）
   0%           高

パレートフロント：どちらかを改善すると、もう一方が悪化する境界
```

**手法**：
- ParEGO（Pareto Efficient Global Optimization）
- NSGA-II（Non-dominated Sorting Genetic Algorithm II）
- EHVI（Expected Hypervolume Improvement）

---

## 2.5 DFT計算との統合

### 2.5.1 DFT（Density Functional Theory）とは

**目的**：量子力学に基づき、原子レベルで電子状態を計算

**計算可能な物性**：
- 吸着エネルギー
- 活性化エネルギー（遷移状態）
- 電子密度分布
- バンド構造

**計算コスト**：
- 1構造：数時間〜数日（CPUコア数による）
- 遷移状態探索：数日〜数週間

### 2.5.2 Multi-Fidelity Optimization

**戦略**：安価な低精度計算と高精度計算を組み合わせ

```
Low-Fidelity:
- 経験的モデル（結合価力場）
- 小さなk-pointメッシュ
- 低カットオフエネルギー
- コスト：1分/構造

High-Fidelity:
- 収束したDFT計算
- 密なk-pointメッシュ
- 高カットオフエネルギー
- コスト：10時間/構造

Multi-Fidelity:
1. Low-Fidelityで1万構造スクリーニング（約7日）
2. 上位100構造をHigh-Fidelityで計算（約42日）
3. 両方のデータでML訓練
4. 予測精度：High-Fidelity単独と同等
5. 総コスト：約1/10
```

### 2.5.3 Transfer Learning

**アイデア**：既存の反応系の知識を新しい反応系に転移

**例**：
```
Source Task: CO酸化（大量データあり）
Target Task: NO還元（データ少ない）

手順:
1. Source TaskでDNN訓練
2. Target Taskで転移学習
   - 下層（汎用的特徴）：固定
   - 上層（タスク固有）：再訓練
3. 必要データ：1/5〜1/10
```

---

## 2.6 主要データベースとツール

### 2.6.1 触媒データベース

**1. Catalysis-Hub.org**
- **内容**：20,000以上の触媒反応エネルギー
- **データ**：DFT計算結果（吸着エネルギー、遷移状態）
- **形式**：JSON API、Python API
- **URL**：https://www.catalysis-hub.org/

**2. Materials Project**
- **内容**：140,000以上の無機材料
- **データ**：結晶構造、バンドギャップ、形成エネルギー
- **API**：Python（pymatgen）
- **URL**：https://materialsproject.org/

**3. NIST Kinetics Database**
- **内容**：化学反応速度定数
- **データ**：Arrhenius parameters（A, Ea）
- **形式**：Web検索
- **URL**：https://kinetics.nist.gov/

### 2.6.2 計算ツール

**1. ASE（Atomic Simulation Environment）**
- **言語**：Python
- **機能**：
  - 構造最適化
  - 振動解析
  - NEB（遷移状態探索）
  - 各種計算エンジンとの連携（VASP, Quantum ESPRESSO）
- **インストール**：`conda install -c conda-forge ase`

**2. Pymatgen**
- **機能**：
  - 結晶構造の読み書き
  - 対称性解析
  - 相図計算
- **Materials Projectとの連携**
- **インストール**：`pip install pymatgen`

**3. matminer**
- **機能**：
  - 記述子の自動計算（200種類以上）
  - データベースからのデータ取得
  - 特徴量エンジニアリング
- **インストール**：`pip install matminer`

---

## 2.7 触媒MIワークフロー

### 統合ワークフロー

<div class="mermaid">
graph TD
    A[ターゲット反応設定] --> B[初期データ収集]
    B --> C[記述子計算]
    C --> D[MLモデル訓練]
    D --> E[ベイズ最適化]
    E --> F[候補触媒選定]
    F --> G{DFT検証}
    G -->|低活性| E
    G -->|高活性| H[実験検証]
    H --> I{目標達成?}
    I -->|No| C
    I -->|Yes| J[最適触媒]
</div>

### 実装例（疑似コード）

```python
# ステップ1：データ収集
data = load_catalysis_hub_data(reaction='CO_oxidation')

# ステップ2：記述子計算
descriptors = calculate_descriptors(data['structures'])

# ステップ3：MLモデル訓練
X_train, X_test, y_train, y_test = train_test_split(descriptors, data['activity'])
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ステップ4：ベイズ最適化
optimizer = BayesianOptimization(model, acquisition='EI')
for i in range(50):
    next_candidate = optimizer.suggest()
    dft_energy = run_dft(next_candidate)  # DFT計算
    optimizer.update(next_candidate, dft_energy)

# ステップ5：最適触媒
best_catalyst = optimizer.get_best()
```

---

## まとめ

本章では、触媒設計に特化したMI手法を学びました：

### 学んだこと

1. **記述子**：電子的、幾何的、組成的、反応記述子の4タイプ
2. **Sabatier原理**：最適な吸着強度、火山型プロット
3. **予測モデル**：Random Forest、XGBoost、Neural Networkの使い分け
4. **ベイズ最適化**：効率的探索、獲得関数（EI, UCB, PI）
5. **DFT統合**：Multi-Fidelity、Transfer Learning
6. **データベース**：Catalysis-Hub、Materials Project、ASE

### 次のステップ

第3章では、Pythonで実際に触媒MIを実装します：
- ASEによる構造操作
- 活性予測モデルの構築
- ベイズ最適化による組成探索
- DFT計算との統合
- 30個の実行可能なコード例

**[第3章へ進む →](./chapter3-hands-on.html)**

---

## 演習問題

### 基礎レベル

**問題1**：d-band理論において、d-bandセンターがフェルミ準位に近いとき、吸着が強くなる理由を説明してください。

**問題2**：以下の記述子を、電子的・幾何的・組成的・反応記述子に分類してください：
- 配位数
- 吸着エネルギー
- 電気陰性度
- 仕事関数

**問題3**：Sabatier原理を「吸着が弱すぎる場合」「強すぎる場合」「最適な場合」の3つに分けて説明してください。

### 中級レベル

**問題4**：ベイズ最適化の3つの獲得関数（EI, UCB, PI）を比較し、それぞれどのような状況で使用すべきか説明してください。

**問題5**：Multi-Fidelity Optimizationが計算コストを削減できる理由を、Low-FidelityとHigh-Fidelityの特徴を含めて説明してください。

### 上級レベル

**問題6**：CO2還元触媒の設計において、活性と選択性を同時に最適化する必要があります。多目的ベイズ最適化を用いた探索戦略を提案してください。以下を含めること：
- 目的関数の定義
- パレートフロントの概念
- 具体的な獲得関数

**問題7**：Transfer Learningを用いて、データが少ない新規反応系（例：アンモニア分解）の触媒設計を行う場合の戦略を設計してください。Source Taskとして何を選ぶべきか、その理由も含めて説明してください。

---

## 参考文献

### 重要論文

1. **Nørskov, J. K., et al.** (2011). "Towards the computational design of solid catalysts." *Nature Chemistry*, 3, 273-278.

2. **Ulissi, Z. W., et al.** (2017). "To address surface reaction network complexity using scaling relations machine learning and DFT calculations." *Nature Communications*, 8, 14621.

3. **Wertheim, M. K., et al.** (2020). "Bayesian optimization for catalysis." *ACS Catalysis*, 10(20), 12186-12200.

### データベース・ツール

- **Catalysis-Hub**: https://www.catalysis-hub.org/
- **Materials Project**: https://materialsproject.org/
- **ASE Documentation**: https://wiki.fysik.dtu.dk/ase/
- **matminer**: https://hackingmaterials.lbl.gov/matminer/

---

**最終更新**: 2025年10月19日
**バージョン**: 1.0
