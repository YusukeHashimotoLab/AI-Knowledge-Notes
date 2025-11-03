---
title: "第3章：Pythonで体験するMLP - SchNetPackハンズオン"
subtitle: "環境構築から訓練、MLP-MDまで"
level: "beginner-intermediate"
difficulty: "初級〜中級"
target_audience: "undergraduate, graduate"
estimated_time: "30-35分"
learning_objectives:
  - Python環境でMLPツール（SchNetPack）をセットアップできる
  - 小規模データセット（MD17）でSchNetモデルを訓練できる
  - 訓練したMLPの精度を評価し、問題を診断できる
  - MLP-MDシミュレーションを実行し、結果を解析できる
topics: ["schnetpack", "python", "hands-on", "training", "md-simulation"]
prerequisites: ["第1章", "第2章", "Python基礎", "Jupyter Notebook"]
series: "MLP入門シリーズ v1.0"
series_order: 3
version: "1.0"
created_at: "2025-10-17"
template_version: "1.0"
---

# 第3章：Pythonで体験するMLP - SchNetPackハンズオン

## 学習目標

この章を読むことで、以下を習得できます：
- Python環境でSchNetPackをインストールし、環境をセットアップできる
- 小規模データセット（MD17のアスピリン分子）を用いてMLPモデルを訓練できる
- 訓練済みモデルの精度を評価し、エネルギー・力の予測誤差を確認できる
- MLP-MDシミュレーションを実行し、トラジェクトリを解析できる
- よくあるエラーと対処法を理解する

---

## 3.1 環境構築：必要なツールのインストール

MLPを実践するには、Python環境とSchNetPackのセットアップが必要です。

### 必要なソフトウェア

| ツール | バージョン | 用途 |
|--------|----------|------|
| **Python** | 3.9-3.11 | 基盤言語 |
| **PyTorch** | 2.0+ | ディープラーニングフレームワーク |
| **SchNetPack** | 2.0+ | MLP訓練・推論 |
| **ASE** | 3.22+ | 原子構造操作、MD実行 |
| **NumPy/Matplotlib** | 最新版 | データ解析・可視化 |

### インストール手順

**ステップ1: Conda環境の作成**

```bash
# 新しいConda環境を作成（Python 3.10）
conda create -n mlp-tutorial python=3.10 -y
conda activate mlp-tutorial
```

**ステップ2: PyTorchのインストール**

```bash
# CPU版（ローカルマシン、軽量）
conda install pytorch cpuonly -c pytorch

# GPU版（CUDAが利用可能な場合）
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

**ステップ3: SchNetPackとASEのインストール**

```bash
# SchNetPack（pip推奨）
pip install schnetpack

# ASE（原子シミュレーション環境）
pip install ase

# 可視化ツール
pip install matplotlib seaborn
```

**ステップ4: 動作確認**

```python
# Example 1: 環境確認スクリプト（5行）
import torch
import schnetpack as spk
print(f"PyTorch: {torch.__version__}")
print(f"SchNetPack: {spk.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
```

**期待される出力**:
```
PyTorch: 2.1.0
SchNetPack: 2.0.3
GPU available: False  # CPUの場合
```

---

## 3.2 データ準備：MD17データセットの取得

SchNetPackは、小規模分子のベンチマークデータセット**MD17**を内蔵しています。

### MD17データセットとは

- **内容**: DFT計算による分子動力学トラジェクトリ
- **対象分子**: アスピリン、ベンゼン、エタノールなど10種類
- **データ数**: 各分子約10万配置
- **精度**: PBE/def2-SVP レベル（DFT）
- **用途**: MLP手法のベンチマーク

### データのダウンロードと読み込み

**Example 2: MD17データセットのロード（10行）**

```python
from schnetpack.datasets import MD17
from schnetpack.data import AtomsDataModule

# アスピリン分子のデータセット（約10万配置）をダウンロード
dataset = MD17(
    datapath='./data',
    molecule='aspirin',
    download=True
)

print(f"Total samples: {len(dataset)}")
print(f"Properties: {dataset.available_properties}")
print(f"First sample: {dataset[0]}")
```

**出力**:
```
Total samples: 211762
Properties: ['energy', 'forces']
First sample: {'_atomic_numbers': tensor([...]), 'energy': tensor(-1234.5), 'forces': tensor([...])}
```

### データの分割

**Example 3: 訓練/検証/テストセットの分割（10行）**

```python
# データを訓練:検証:テスト = 70%:15%:15%に分割
data_module = AtomsDataModule(
    datapath='./data',
    dataset=dataset,
    batch_size=32,
    num_train=100000,      # 訓練データ数
    num_val=10000,          # 検証データ数
    num_test=10000,         # テストデータ数
    split_file='split.npz', # 分割情報を保存
)
data_module.prepare_data()
data_module.setup()
```

**説明**:
- `batch_size=32`: 32配置ずつまとめて処理（メモリ効率）
- `num_train=100000`: 大量データで汎化性能向上
- `split_file`: 分割をファイルに保存（再現性確保）

---

## 3.3 SchNetPackでのモデル訓練

SchNetモデルを訓練し、エネルギーと力を学習します。

### SchNetアーキテクチャの設定

**Example 4: SchNetモデルの定義（15行）**

```python
import schnetpack.transform as trn
from schnetpack.representation import SchNet
from schnetpack.model import AtomisticModel
from schnetpack.task import ModelOutput

# 1. SchNet表現層（原子配置→特徴ベクトル）
representation = SchNet(
    n_atom_basis=128,      # 原子特徴ベクトルの次元
    n_interactions=6,      # メッセージパッシング層の数
    cutoff=5.0,            # カットオフ半径（Å）
    n_filters=128          # フィルタ数
)

# 2. 出力層（エネルギー予測）
output = ModelOutput(
    name='energy',
    loss_fn=torch.nn.MSELoss(),
    metrics={'MAE': spk.metrics.MeanAbsoluteError()}
)
```

**パラメータ解説**:
- `n_atom_basis=128`: 各原子の特徴ベクトルが128次元（典型的な値）
- `n_interactions=6`: 6層のメッセージパッシング（深いほど長距離相互作用を捉える）
- `cutoff=5.0Å`: この距離以上の原子間相互作用を無視（計算効率）

### 訓練の実行

**Example 5: 訓練ループの設定（15行）**

```python
import pytorch_lightning as pl
from schnetpack.task import AtomisticTask

# 訓練タスクの定義
task = AtomisticTask(
    model=AtomisticModel(representation, [output]),
    outputs=[output],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={'lr': 1e-4}  # 学習率
)

# Trainerの設定
trainer = pl.Trainer(
    max_epochs=50,               # 最大50エポック
    accelerator='cpu',           # CPU使用（GPU: 'gpu'）
    devices=1,
    default_root_dir='./training'
)

# 訓練開始
trainer.fit(task, datamodule=data_module)
```

**訓練時間の目安**:
- CPU（4コア）: 約2-3時間（10万配置）
- GPU（RTX 3090）: 約15-20分

### 訓練の進捗確認

**Example 6: TensorBoardでの可視化（10行）**

```python
# TensorBoardの起動（別ターミナル）
# tensorboard --logdir=./training/lightning_logs

# Pythonからのログ確認
import pandas as pd

metrics = pd.read_csv('./training/lightning_logs/version_0/metrics.csv')
print(metrics[['epoch', 'train_loss', 'val_loss']].tail(10))
```

**期待される出力**:
```
   epoch  train_loss  val_loss
40    40      0.0023    0.0031
41    41      0.0022    0.0030
42    42      0.0021    0.0029
...
```

**観察ポイント**:
- `train_loss`と`val_loss`がともに減少 → 正常に学習中
- `val_loss`が増加し始めたら **過学習**の兆候 → Early Stoppingを検討

---

## 3.4 精度検証：エネルギーと力の予測精度

訓練したモデルがDFT精度を達成しているか評価します。

### テストセットでの評価

**Example 7: テストセット評価（12行）**

```python
# テストセットで評価
test_results = trainer.test(task, datamodule=data_module)

# 結果の表示
print(f"Energy MAE: {test_results[0]['test_energy_MAE']:.4f} eV")
print(f"Energy RMSE: {test_results[0]['test_energy_RMSE']:.4f} eV")

# 力の評価（別途計算が必要）
from schnetpack.metrics import MeanAbsoluteError
force_mae = MeanAbsoluteError(target='forces')
# ... 力の評価コード
```

**良好な精度の目安**（アスピリン分子、21原子）:
- **エネルギーMAE**: < 1 kcal/mol（< 0.043 eV）
- **力のMAE**: < 1 kcal/mol/Å（< 0.043 eV/Å）

### 予測値と真値の相関プロット

**Example 8: 予測精度の可視化（15行）**

```python
import matplotlib.pyplot as plt
import numpy as np

# テストデータで予測
model = task.model
predictions, targets = [], []

for batch in data_module.test_dataloader():
    pred = model(batch)['energy'].detach().numpy()
    true = batch['energy'].numpy()
    predictions.extend(pred)
    targets.extend(true)

# 散布図プロット
plt.scatter(targets, predictions, alpha=0.5, s=1)
plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
plt.xlabel('DFT Energy (eV)')
plt.ylabel('MLP Predicted Energy (eV)')
plt.title('Energy Prediction Accuracy')
plt.show()
```

**理想的な結果**:
- 点が赤い対角線（y=x）上に密集
- R² > 0.99（決定係数）

---

## 3.5 MLP-MDシミュレーション：分子動力学の実行

訓練したMLPを使って、DFTより10⁴倍高速なMDシミュレーションを実行します。

### ASEでのMLP-MD設定

**Example 9: MLP-MD計算の準備（10行）**

```python
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
import schnetpack.interfaces.ase_interface as spk_ase

# MLPをASE Calculatorとしてラップ
calculator = spk_ase.SpkCalculator(
    model_file='./training/best_model.ckpt',
    device='cpu'
)

# 初期構造の準備（MD17の最初の配置）
atoms = dataset.get_atoms(0)
atoms.calc = calculator
```

### 初期速度の設定と平衡化

**Example 10: 温度初期化（10行）**

```python
# 300Kでの速度分布を設定
temperature = 300  # K
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

# 運動量をゼロに（系全体の並進を除去）
from ase.md.velocitydistribution import Stationary
Stationary(atoms)

print(f"Initial kinetic energy: {atoms.get_kinetic_energy():.3f} eV")
print(f"Initial potential energy: {atoms.get_potential_energy():.3f} eV")
```

### MDシミュレーションの実行

**Example 11: MD実行とトラジェクトリ保存（12行）**

```python
from ase.io.trajectory import Trajectory

# MDシミュレータの設定
timestep = 0.5 * units.fs  # 0.5フェムト秒
dyn = VelocityVerlet(atoms, timestep=timestep)

# トラジェクトリファイル出力
traj = Trajectory('aspirin_md.traj', 'w', atoms)
dyn.attach(traj.write, interval=10)  # 10ステップごとに保存

# 10,000ステップ（5ピコ秒）のMD実行
dyn.run(10000)
print("MD simulation completed!")
```

**計算時間の目安**:
- CPU（4コア）: 約5分（10,000ステップ）
- DFTなら: 約1週間（10,000ステップ）
- **10⁴倍の高速化達成！**

### トラジェクトリの解析

**Example 12: エネルギー保存とRDF計算（15行）**

```python
from ase.io import read
import numpy as np

# トラジェクトリの読み込み
traj_data = read('aspirin_md.traj', index=':')

# エネルギー保存の確認
energies = [a.get_total_energy() for a in traj_data]
plt.plot(energies)
plt.xlabel('Time step')
plt.ylabel('Total Energy (eV)')
plt.title('Energy Conservation Check')
plt.show()

# エネルギードリフト（単調増加/減少）の計算
drift = (energies[-1] - energies[0]) / len(energies)
print(f"Energy drift: {drift:.6f} eV/step")
```

**良好なシミュレーションの指標**:
- エネルギードリフト: < 0.001 eV/step
- 全エネルギーが時間とともに振動（保存則）

---

## 3.6 物性計算：振動スペクトルと拡散係数

MLP-MDから物理的な物性値を計算します。

### 振動スペクトル（パワースペクトル）

**Example 13: 振動スペクトル計算（15行）**

```python
from scipy.fft import fft, fftfreq

# 1つの原子の速度時系列を抽出
atom_idx = 0  # 最初の原子
velocities = np.array([a.get_velocities()[atom_idx] for a in traj_data])

# x方向速度のフーリエ変換
vx = velocities[:, 0]
freq = fftfreq(len(vx), d=timestep)
spectrum = np.abs(fft(vx))**2

# 正の周波数のみプロット
mask = freq > 0
plt.plot(freq[mask] * 1e15 / (2 * np.pi), spectrum[mask])  # Hz → THz変換
plt.xlabel('Frequency (THz)')
plt.ylabel('Power Spectrum')
plt.title('Vibrational Spectrum')
plt.xlim(0, 100)
plt.show()
```

**解釈**:
- ピークが分子の振動モードに対応
- DFTで計算した振動スペクトルと比較することで精度検証

### 平均二乗変位（MSD）と拡散係数

**Example 14: MSD計算（15行）**

```python
def calculate_msd(traj, atom_idx=0):
    """平均二乗変位を計算"""
    positions = np.array([a.positions[atom_idx] for a in traj])
    msd = np.zeros(len(positions))

    for t in range(len(positions)):
        displacement = positions[t:] - positions[:-t or None]
        msd[t] = np.mean(np.sum(displacement**2, axis=1))

    return msd

# MSD計算とプロット
msd = calculate_msd(traj_data)
time_ps = np.arange(len(msd)) * timestep / units.fs * 1e-3  # ピコ秒

plt.plot(time_ps, msd)
plt.xlabel('Time (ps)')
plt.ylabel('MSD (Ų)')
plt.title('Mean Square Displacement')
plt.show()
```

**拡散係数の計算**:
```python
# MSDの線形領域から拡散係数を計算（Einstein関係式）
# D = lim_{t→∞} MSD(t) / (6t)
linear_region = slice(100, 500)
fit = np.polyfit(time_ps[linear_region], msd[linear_region], deg=1)
D = fit[0] / 6  # Ų/ps → cm²/s変換が必要
print(f"Diffusion coefficient: {D:.6f} Ų/ps")
```

---

## 3.7 Active Learning：効率的なデータ追加

モデルが不確実な配置を自動検出し、DFT計算を追加します。

### アンサンブル不確実性の評価

**Example 15: 予測の不確実性（15行）**

```python
# 複数の独立したモデルを訓練（省略：Example 5を3回実行）
models = [model1, model2, model3]  # 3つの独立モデル

def predict_with_uncertainty(atoms, models):
    """アンサンブル予測と不確実性"""
    predictions = []
    for model in models:
        atoms.calc = spk_ase.SpkCalculator(model_file=model, device='cpu')
        predictions.append(atoms.get_potential_energy())

    mean = np.mean(predictions)
    std = np.std(predictions)
    return mean, std

# MDトラジェクトリの各配置で不確実性評価
uncertainties = []
for atoms in traj_data[::100]:  # 100フレームごと
    _, std = predict_with_uncertainty(atoms, models)
    uncertainties.append(std)

# 不確実性が高い配置を特定
threshold = np.percentile(uncertainties, 95)
high_uncertainty_frames = np.where(np.array(uncertainties) > threshold)[0]
print(f"High uncertainty frames: {high_uncertainty_frames}")
```

**次のステップ**:
- 不確実性の高い配置をDFT計算に追加
- データセットを更新してモデル再訓練
- 精度向上を確認

---

## 3.8 トラブルシューティング：よくあるエラーと対処法

実践でよく遭遇する問題と解決策を紹介します。

| エラー | 原因 | 対処法 |
|--------|------|--------|
| **Out of Memory (OOM)** | バッチサイズが大きすぎる | `batch_size`を32→16→8と減らす |
| **Loss becomes NaN** | 学習率が高すぎる | `lr=1e-4`→`1e-5`に下げる |
| **Energy drift in MD** | タイムステップが大きすぎる | `timestep=0.5fs`→`0.25fs`に減らす |
| **Poor generalization** | 訓練データが偏っている | Active Learningでデータ多様化 |
| **CUDA error** | GPU互換性の問題 | PyTorchとCUDAバージョン確認 |

### デバッグのベストプラクティス

```python
# 1. 小規模データでテスト
data_module.num_train = 1000  # 1,000配置でクイックテスト

# 2. 1バッチでのオーバーフィッティング確認
trainer = pl.Trainer(max_epochs=100, overfit_batches=1)
# 訓練誤差が0に近づけば、モデルに学習能力あり

# 3. グラディエントのクリッピング
task = AtomisticTask(..., gradient_clip_val=1.0)  # 勾配爆発防止
```

---

## 3.9 本章のまとめ

### 学んだこと

1. **環境構築**
   - Conda環境、PyTorch、SchNetPackのインストール
   - GPU/CPU環境の選択

2. **データ準備**
   - MD17データセットのダウンロードと読み込み
   - 訓練/検証/テストセットへの分割

3. **モデル訓練**
   - SchNetアーキテクチャの設定（6層、128次元）
   - 50エポックの訓練（CPU: 2-3時間）
   - TensorBoardでの進捗確認

4. **精度検証**
   - エネルギーMAE < 1 kcal/mol達成を確認
   - 予測値vs真値の相関プロット
   - R² > 0.99の高精度

5. **MLP-MD実行**
   - ASE Calculatorとしての統合
   - 10,000ステップ（5ピコ秒）のMD実行
   - DFTより10⁴倍高速化を体験

6. **物性計算**
   - 振動スペクトル（フーリエ変換）
   - 拡散係数（平均二乗変位から計算）

7. **Active Learning**
   - アンサンブル不確実性による配置選択
   - データ追加の自動化戦略

### 重要なポイント

- **SchNetPackは実装が容易**: 数十行のコードでMLP訓練が可能
- **小規模データ（10万配置）で実用精度達成**: MD17は優れたベンチマーク
- **MLP-MDは実用的**: DFTの10⁴倍高速、個人のPCで実行可能
- **Active Learningで効率化**: 重要な配置を自動発見、データ収集コスト削減

### 次の章へ

第4章では、最新のMLP手法（NequIP、MACE）と実際の研究応用例を学びます：
- E(3)等変グラフニューラルネットワークの理論
- データ効率の劇的向上（10万→3,000配置）
- 触媒反応、バッテリー材料への応用事例
- 大規模シミュレーション（100万原子）の実現

---

## 演習問題

### 問題1（難易度：easy）

Example 4のSchNet設定で、`n_interactions`（メッセージパッシング層の数）を3, 6, 9に変えて訓練し、テストMAEがどのように変化するか予測してください。

<details>
<summary>ヒント</summary>

層が深いほど、長距離の原子間相互作用を捉えられます。しかし、深すぎると過学習のリスクも。

</details>

<details>
<summary>解答例</summary>

**予測される結果**:

| `n_interactions` | テストMAE予測 | 訓練時間 | 特徴 |
|-----------------|-------------|---------|------|
| **3** | 0.8-1.2 kcal/mol | 1時間 | 浅いため長距離相互作用を捉えきれない |
| **6** | 0.5-0.8 kcal/mol | 2-3時間 | バランスが良い（推奨） |
| **9** | 0.6-1.0 kcal/mol | 4-5時間 | 過学習リスク、訓練データ不足なら精度低下 |

**実験方法**:
```python
for n in [3, 6, 9]:
    representation = SchNet(n_interactions=n, ...)
    task = AtomisticTask(...)
    trainer.fit(task, datamodule=data_module)
    results = trainer.test(task, datamodule=data_module)
    print(f"n={n}: MAE={results[0]['test_energy_MAE']:.4f} eV")
```

**結論**: 小分子（アスピリン21原子）では`n_interactions=6`が最適。大規模系（100原子以上）では9-12層が有効な場合もある。

</details>

### 問題2（難易度：medium）

Example 11のMLP-MDで、エネルギードリフトが許容範囲を超えた場合（例: 0.01 eV/step）、どのような対処法が考えられますか？3つ挙げてください。

<details>
<summary>ヒント</summary>

タイムステップ、訓練精度、MDアルゴリズムの3つの観点から考えましょう。

</details>

<details>
<summary>解答例</summary>

**対処法1: タイムステップを小さくする**
```python
timestep = 0.25 * units.fs  # 0.5fs → 0.25fsに半減
dyn = VelocityVerlet(atoms, timestep=timestep)
```
- **理由**: 小さいタイムステップは数値積分の誤差を減らす
- **デメリット**: 2倍の計算時間

**対処法2: モデル訓練精度を向上**
```python
# より多くのデータで訓練
data_module.num_train = 200000  # 10万→20万配置に増加

# または力の損失関数の重みを増やす
task = AtomisticTask(..., loss_weights={'energy': 1.0, 'forces': 1000})
```
- **理由**: 力の予測精度が低いとMDが不安定
- **目標**: 力のMAE < 0.05 eV/Å

**対処法3: Langevin動力学に変更（熱浴結合）**
```python
from ase.md.langevin import Langevin
dyn = Langevin(atoms, timestep=0.5*units.fs,
               temperature_K=300, friction=0.01)
```
- **理由**: 熱浴がエネルギードリフトを吸収
- **注意**: 厳密な微小正準アンサンブル（NVE）ではなくなる

**優先順位**: 対処法2（精度向上）→ 対処法1（タイムステップ）→ 対処法3（Langevin）

</details>

---

## 参考文献

1. Schütt, K. T., et al. (2019). "SchNetPack: A Deep Learning Toolbox For Atomistic Systems." *Journal of Chemical Theory and Computation*, 15(1), 448-455.
   DOI: [10.1021/acs.jctc.8b00908](https://doi.org/10.1021/acs.jctc.8b00908)

2. Chmiela, S., et al. (2017). "Machine learning of accurate energy-conserving molecular force fields." *Science Advances*, 3(5), e1603015.
   DOI: [10.1126/sciadv.1603015](https://doi.org/10.1126/sciadv.1603015)

3. Larsen, A. H., et al. (2017). "The atomic simulation environment—a Python library for working with atoms." *Journal of Physics: Condensed Matter*, 29(27), 273002.
   DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)

4. Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance deep learning library." *Advances in Neural Information Processing Systems*, 32.
   arXiv: [1912.01703](https://arxiv.org/abs/1912.01703)

5. Zhang, L., et al. (2020). "Active learning of uniformly accurate interatomic potentials for materials simulation." *Physical Review Materials*, 3(2), 023804.
   DOI: [10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1103/PhysRevMaterials.3.023804)

6. Schütt, K. T., et al. (2017). "Quantum-chemical insights from deep tensor neural networks." *Nature Communications*, 8(1), 13890.
   DOI: [10.1038/ncomms13890](https://doi.org/10.1038/ncomms13890)

---

## 著者情報

**作成者**: MI Knowledge Hub Content Team
**監修**: Dr. Yusuke Hashimoto（東北大学）
**作成日**: 2025-10-17
**バージョン**: 1.0（Chapter 3 initial version）
**シリーズ**: MLP入門シリーズ

**更新履歴**:
- 2025-10-17: v1.0 第3章初版作成
  - Python環境構築（Conda, PyTorch, SchNetPack）
  - MD17データセット準備と分割
  - SchNetモデル訓練（15コード例）
  - MLP-MD実行と解析（トラジェクトリ、振動スペクトル、MSD）
  - Active Learning不確実性評価
  - トラブルシューティング表（5項目）
  - 演習問題2問（easy, medium）
  - 参考文献6件

**ライセンス**: Creative Commons BY-NC-SA 4.0
