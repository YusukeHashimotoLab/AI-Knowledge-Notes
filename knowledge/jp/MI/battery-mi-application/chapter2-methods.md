---
chapter_number: 2
chapter_title: "電池材料設計に特化したMI手法"
subtitle: "記述子設計から予測モデル構築まで"
series: "電池材料MI応用シリーズ"
difficulty: "中級"
reading_time: "30-35分"
code_examples: 4
exercises: 5
mermaid_diagrams: 0
prerequisites:
  - "第1章の内容理解"
  - "機械学習の基礎"
  - "Python（NumPy, scikit-learn）"
learning_objectives:
  basic:
    - "電池材料記述子の種類と用途を理解する"
    - "容量・電圧予測モデルの原理を説明できる"
  practical:
    - "Graph Neural Networkの利点を活用できる"
    - "ベイズ最適化で材料探索を実行できる"
  advanced:
    - "Multi-fidelity最適化で計算コストを削減できる"
    - "サイクル劣化予測モデルを構築できる"
keywords:
  - "記述子設計"
  - "容量予測"
  - "Graph Neural Network"
  - "ベイズ最適化"
  - "サイクル劣化"
  - "LSTM"
  - "Materials Project"
  - "PyBaMM"
---
# 第2章：電池材料設計に特化したMI手法

**学習目標:**
- 電池材料記述子の種類と用途を理解する
- 容量・電圧予測モデルの構築方法を習得する
- サイクル劣化予測手法を学ぶ
- 高速材料スクリーニング戦略を把握する

**読了時間**: 30-35分

---

## 2.1 電池材料記述子（Descriptor）

記述子とは、材料の特性を数値化した特徴量です。適切な記述子選択が予測精度の鍵となります。

### 2.1.1 構造記述子（Structural Descriptors）

**結晶構造パラメータ:**
- **格子定数（Lattice Parameters）**: a, b, c, α, β, γ
  - 例: LiCoO₂（a = 2.82 Å, c = 14.05 Å）
  - 影響：Li⁺拡散経路、イオン伝導度
- **空間群（Space Group）**: 対称性の分類
  - 例: R-3m（層状構造）、Fd-3m（スピネル構造）
- **体積変化（Volume Change）**: 充放電時の膨張・収縮
  - 計算：`ΔV = (V_charged - V_discharged) / V_discharged × 100%`
  - 目標：< 5%（構造安定性）

**配位環境:**
- **配位数（Coordination Number）**: 遷移金属イオンの配位数
  - 例: Coが酸素6配位（八面体）
- **結合距離（Bond Length）**: M-O距離
  - 例: Co-O = 1.92 Å（LiCoO₂）
- **ポリへドロン歪み**: 八面体の歪み度

### 2.1.2 電子記述子（Electronic Descriptors）

**バンド構造:**
- **バンドギャップ（Band Gap）**: 絶縁性/導電性の指標
  - 正極材料：< 3 eV（電子伝導性確保）
  - 固体電解質：> 5 eV（絶縁性確保）
- **状態密度（DOS: Density of States）**: エネルギー準位分布
  - フェルミ準位付近のDOSが導電性に影響
- **d-band中心（d-band Center）**: 遷移金属のd軌道エネルギー
  - Ni, Co, Mnの酸化還元特性に関連

**電荷:**
- **Bader電荷解析**: 原子の有効電荷
  - 例: Li⁺（+0.85）、Co³⁺（+1.5）
- **酸化状態（Oxidation State）**: 充放電時の変化
  - 例: Co³⁺ ⇌ Co⁴⁺（LiCoO₂の充放電）

**仕事関数（Work Function）:**
- 定義：真空準位とフェルミ準位の差
- 影響：電極-電解質界面の電子移動

### 2.1.3 化学記述子（Chemical Descriptors）

**元素特性:**
- **イオン半径（Ionic Radius）**: Li⁺（0.76 Å）、Na⁺（1.02 Å）
  - 影響：イオン拡散速度、構造安定性
- **電気陰性度（Electronegativity）**: Paulingスケール
  - 影響：共有結合性、酸化還元電位
- **原子質量（Atomic Mass）**: エネルギー密度に影響

**組成:**
- **Li/M比**: Li₁₊ₓCoO₂のx（過剰Li量）
- **遷移金属比**: NCM622（Ni:Co:Mn = 6:2:2）
- **ドーパント**: Al, Mg, Ti添加

### 2.1.4 電気化学記述子（Electrochemical Descriptors）

**熱力学特性:**
- **電位（Voltage）**: vs. Li/Li⁺
  - 計算（DFT）: `V = -ΔG / (nF)`
  - n: 電子数、F: ファラデー定数（96,485 C/mol）
- **容量（Capacity）**: mAh/g
  - 理論容量: `C = nF / (3.6 × M)`
  - M: 分子量（g/mol）
- **形成エネルギー（Formation Energy）**: 安定性指標
  - `E_f = E_compound - Σ E_elements`

**動力学特性:**
- **イオン伝導度（Ionic Conductivity）**: S/cm
  - 液体電解質：10⁻² S/cm
  - 固体電解質目標：> 10⁻³ S/cm
- **拡散係数（Diffusion Coefficient）**: cm²/s
  - Li⁺拡散：10⁻⁸～10⁻¹² cm²/s
- **活性化エネルギー（Activation Energy）**: eV
  - イオン拡散障壁、反応障壁

---

## 2.2 容量・電圧予測モデル

### 2.2.1 回帰モデル

**Random Forest:**
- **利点**: 非線形関係を捉える、特徴量重要度が得られる
- **欠点**: 外挿性能が低い
- **適用例**: 正極材料の容量予測（R² > 0.90）

**XGBoost (Extreme Gradient Boosting):**
- **利点**: 高精度、過学習制御（正則化）
- **欠点**: ハイパーパラメータ調整が必要
- **適用例**: 電圧プロファイル予測（MAE < 0.1 V）

**Neural Network:**
- **利点**: 表現力が高い、大規模データで高精度
- **欠点**: データ量が必要、解釈性が低い
- **適用例**: 多変数同時予測（容量 + 電圧 + サイクル寿命）

### 2.2.2 Graph Neural Network（GNN）

**概要:**
- 結晶構造を直接入力（原子 = ノード、結合 = エッジ）
- 畳み込み演算で局所構造を学習

**アーキテクチャ:**
```
結晶構造 → Graph Embedding → Convolution Layers → Readout → 予測値
```

**利点:**
- 記述子設計不要（End-to-End学習）
- 対称性、周期性を自動学習
- 新規構造への汎化性能が高い

**代表的手法:**
- **CGCNN** (Crystal Graph Convolutional Neural Network)
- **MEGNet** (MatErials Graph Network)
- **SchNet**: 連続フィルター畳み込み

**適用例:**
- Materials Project 69,000材料で学習
- 容量予測：MAE = 8.5 mAh/g
- 電圧予測：MAE = 0.09 V

### 2.2.3 Transfer Learning（転移学習）

**原理:**
- ソースタスク（大規模データ）で事前学習
- ターゲットタスク（少数データ）でファインチューニング

**電池への応用:**
- ソース：LIB正極材料（10,000サンプル）
- ターゲット：全固体電池正極（100サンプル）
- 効果：予測精度20-30%向上

**実装:**
```python
# 事前学習済みモデル
pretrained_model = load_model('lib_cathode_model.h5')

# 最終層を置き換え
model = Sequential([
    pretrained_model.layers[:-1],  # 特徴抽出部
    Dense(64, activation='relu'),
    Dense(1)  # 新しいタスク用
])

# ファインチューニング
model.compile(optimizer=Adam(lr=1e-4), loss='mse')
model.fit(X_target, y_target, epochs=50)
```

### 2.2.4 物理ベースモデルとMLの統合

**Multi-fidelity Optimization:**
- 低精度・高速：経験的モデル、ML予測
- 高精度・低速：DFT計算
- 統合：Gaussian Processで両者を融合

**Bayesian Model Averaging:**
- 複数モデル（ML、DFT、実験）の予測を統合
- 不確実性を定量化

---

## 2.3 サイクル劣化予測

### 2.3.1 劣化メカニズム

**SEI（Solid Electrolyte Interphase）成長:**
- 負極表面での電解液分解
- 容量損失：Li⁺の不可逆消費
- 抵抗増加：イオン伝導阻害

**リチウム析出（Li Plating）:**
- 急速充電時に発生
- リスク：内部短絡、熱暴走
- 検出：充電曲線の異常（電圧平坦部）

**構造崩壊:**
- 正極材料の相転移、亀裂発生
- 原因：充放電時の体積変化
- 指標：XRD, TEMでの構造変化

**電解液分解:**
- 高温、高電圧での分解
- ガス発生：CO₂, CO, C₂H₄
- 対策：添加剤、難燃性電解液

### 2.3.2 時系列モデル（LSTM/GRU）

**LSTM (Long Short-Term Memory):**
- **構造**: 入力ゲート、忘却ゲート、出力ゲート
- **利点**: 長期依存関係を学習
- **適用**: 充放電曲線→容量予測

**アーキテクチャ:**
```
入力: [V(t), I(t), T(t)]  # 電圧、電流、温度
  ↓
LSTM Layer (64 units)
  ↓
LSTM Layer (32 units)
  ↓
Dense Layer (16 units)
  ↓
出力: SOH(t+k)  # k サイクル後のSOH
```

**GRU (Gated Recurrent Unit):**
- LSTM簡略版（ゲート数削減）
- 計算コスト低、精度はLSTMと同等

### 2.3.3 寿命予測（RUL: Remaining Useful Life）

**定義:**
- 現在から容量80%到達までのサイクル数

**手法:**
- **Early Prediction**: 初期100サイクルから予測
- **特徴量**: 容量減衰率、電圧曲線の形状変化、内部抵抗
- **モデル**: LSTM、XGBoost、Gaussian Process

**成果例:**
- 初期100サイクルからRUL予測
- 予測誤差：< 10%（MIT, 2019）
- 早期スクリーニング：不良品を200サイクル以内に検出

### 2.3.4 異常検知（Anomaly Detection）

**手法:**
- **Isolation Forest**: 外れ値検出
- **Autoencoder**: 正常データで学習、異常を再構成誤差で検出
- **One-Class SVM**: 正常データの境界を学習

**適用:**
- 劣化加速の早期検出
- 内部短絡の予兆検出
- 製造不良の判別

---

## 2.4 高速材料スクリーニング

### 2.4.1 ベイズ最適化

**原理:**
- Gaussian Processでサロゲートモデル構築
- 獲得関数で次実験を選択
- 実験→更新→次実験のループ

**獲得関数:**
```
EI (Expected Improvement):
  EI(x) = E[max(0, f(x) - f_best)]

UCB (Upper Confidence Bound):
  UCB(x) = μ(x) + κσ(x)
  κ: 探索vs活用のバランス

PI (Probability of Improvement):
  PI(x) = P(f(x) > f_best)
```

**電池材料への応用:**
- 組成最適化：NCMのNi:Co:Mn比
- 電解液組成：溶媒比率、塩濃度
- 合成条件：温度、時間、雰囲気

**成果:**
- 実験数70%削減
- 最適組成発見までの期間：1年 → 3ヶ月

### 2.4.2 Active Learning

**サイクル:**
```
1. 初期データで予測モデル訓練
2. 不確実性が高いサンプルを選択
3. 実験（or DFT計算）で測定
4. データ追加してモデル更新
5. ステップ2に戻る
```

**選択基準:**
- **Uncertainty Sampling**: 予測の不確実性が高い
- **Query-by-Committee**: 複数モデルの予測が異なる
- **Expected Model Change**: モデルへの影響が大きい

**適用例:**
- 固体電解質探索：10,000候補から実験50件で最適材料発見
- イオン伝導度予測：R² = 0.85 → 0.95（Active Learning後）

### 2.4.3 Multi-fidelity Optimization

**概要:**
- 低fidelity（低コスト・低精度）：経験的計算、MLモデル
- 高fidelity（高コスト・高精度）：DFT計算、実験
- 両者を統合して効率的探索

**手法:**
- **Co-Kriging**: 複数fidelityのデータを同時に扱う
- **Multi-task Learning**: 異なるfidelityを別タスクとして学習

**電池への応用:**
- 低fidelity：GNN予測（秒単位）
- 中fidelity：DFT（時間単位）
- 高fidelity：実験（週単位）
- 統合効果：総コスト50%削減

---

## 2.5 主要データベースとツール

### 2.5.1 Materials Project

**URL**: https://materialsproject.org/

**データ:**
- 材料数：140,000+
- 電池関連：電圧、容量、相安定性、イオン伝導度
- DFT計算：構造最適化、電子構造

**API:**
```python
from pymatgen.ext.matproj import MPRester

with MPRester("YOUR_API_KEY") as mpr:
    # LiCoO2の検索
    data = mpr.query(
        criteria={"formula": "LiCoO2"},
        properties=["material_id", "energy", "band_gap"]
    )
```

**活用例:**
- 正極材料スクリーニング
- 電圧予測モデルの訓練データ
- 構造記述子の自動計算

### 2.5.2 Battery Data Genome

**URL**: https://data.matr.io/

**データ:**
- 充放電曲線：20,000+セル
- サイクル試験データ：多様な条件
- 実験条件：温度、C-rate、電圧範囲

**特徴:**
- 生データ公開（前処理不要）
- 複数研究機関のデータ統合
- 機械学習ベンチマーク提供

**活用例:**
- サイクル劣化予測モデル訓練
- 異常検知アルゴリズム開発
- 充電プロトコル最適化

### 2.5.3 NIST Battery Database

**URL**: https://www.nist.gov/

**データ:**
- 標準データセット
- 測定プロトコル
- 品質管理データ

**適用:**
- モデル検証用標準データ
- 測定手法の標準化

### 2.5.4 PyBaMM (Python Battery Mathematical Modeling)

**URL**: https://pybamm.org/

**機能:**
- 電池モデリング：DFN, SPM, SPMe
- 物理パラメータライブラリ
- カスタムモデル構築

**主要モデル:**
- **DFN** (Doyle-Fuller-Newman): 詳細電気化学モデル
- **SPM** (Single Particle Model): 簡易モデル
- **SPMe** (SPM with Electrolyte): SPMの拡張

**使用例:**
```python
import pybamm

# DFNモデルの構築
model = pybamm.lithium_ion.DFN()

# パラメータ設定（Graphite || LCO）
parameter_values = pybamm.ParameterValues("Chen2020")

# 充放電シミュレーション
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sim.solve([0, 3600])  # 1時間シミュレーション

# 結果可視化
sim.plot()
```

**活用:**
- 充放電曲線予測
- パラメータフィッティング
- 新材料の性能シミュレーション

### 2.5.5 その他ツール

**matminer:**
- 材料記述子の自動計算
- 特徴量エンジニアリング

**PyTorch Geometric:**
- Graph Neural Networkライブラリ
- 結晶構造からの予測

**scikit-optimize:**
- ベイズ最適化ライブラリ
- 組成最適化

---

## 2.6 まとめ

### 本章で学んだこと

1. **電池材料記述子:**
   - 構造記述子（格子定数、配位環境）
   - 電子記述子（バンドギャップ、d-band中心）
   - 化学記述子（イオン半径、電気陰性度）
   - 電気化学記述子（電位、イオン伝導度）

2. **予測モデル:**
   - 回帰モデル（Random Forest, XGBoost, Neural Network）
   - Graph Neural Network（CGCNN, MEGNet）
   - Transfer Learning（少数データへの適用）
   - 物理ベースモデルとの統合

3. **サイクル劣化予測:**
   - 劣化メカニズム（SEI、Li析出、構造崩壊）
   - 時系列モデル（LSTM, GRU）
   - 寿命予測（RUL）
   - 異常検知（Isolation Forest, Autoencoder）

4. **高速スクリーニング:**
   - ベイズ最適化（獲得関数、サロゲートモデル）
   - Active Learning（効率的データ収集）
   - Multi-fidelity Optimization（計算コスト削減）

5. **データベース・ツール:**
   - Materials Project（140,000+材料）
   - Battery Data Genome（充放電曲線）
   - PyBaMM（電池シミュレーション）

### 次のステップ

第3章では、これらの手法をPythonで実装します：
- PyBaMMでの電池シミュレーション
- XGBoostによる容量予測
- LSTMによるサイクル劣化予測
- ベイズ最適化による材料探索
- 30個の実行可能なコード例

---

## 演習問題

**問1:** 正極材料LiNi₀.₈Co₀.₁Mn₀.₁O₂の理論容量を計算せよ（分子量: 96.5 g/mol、電子数: 1）。

**問2:** Graph Neural Networkが従来の記述子ベース手法より優れている点を3つ挙げよ。

**問3:** LSTMによるサイクル劣化予測で、初期100サイクルのデータから2,000サイクル後のSOHを予測する際の入力と出力を定義せよ。

**問4:** ベイズ最適化で正極材料のNi:Co:Mn比を最適化する際、獲得関数としてEIとUCBのどちらが適切か理由とともに説明せよ。

**問5:** Multi-fidelity Optimizationで、DFT計算と実験の2つのfidelityを統合する利点を、コストと精度の観点から論じよ（400字以内）。

---

## 参考文献

1. Sendek, A. D. et al. "Machine Learning-Assisted Discovery of Solid Li-Ion Conducting Materials." *Chem. Mater.* (2019).
2. Chen, C. et al. "A Critical Review of Machine Learning of Energy Materials." *Adv. Energy Mater.* (2020).
3. Attia, P. M. et al. "Closed-loop optimization of fast-charging protocols." *Nature* (2020).
4. Xie, T. & Grossman, J. C. "Crystal Graph Convolutional Neural Networks." *Phys. Rev. Lett.* (2018).
5. Severson, K. A. et al. "Data-driven prediction of battery cycle life." *Nat. Energy* (2019).

---

**次章**: [第3章：Pythonで実装する電池MI](chapter3-hands-on.md)

**ライセンス**: このコンテンツはCC BY 4.0ライセンスの下で提供されています。
