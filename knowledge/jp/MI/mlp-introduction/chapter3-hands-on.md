---
title: 第3章：Pythonで体験するMLP - SchNetPackハンズオン
chapter_title: 第3章：Pythonで体験するMLP - SchNetPackハンズオン
---

# 第3章：Pythonで体験するMLP - SchNetPackハンズオン

小さなデータセットでの訓練・評価ワークフローを一通り回します。再現性確保と過学習対策のチェックポイントも明確にします。

**💡 補足:** シード固定、データ分割、学習曲線の記録が基本三点セット。早期終了と重み減衰で安定化します。

## 学習目標

この章を読むことで、以下を習得できます：  
\- Python環境でSchNetPackをインストールし、環境をセットアップできる  
\- 小規模データセット（MD17のアスピリン分子）を用いてMLPモデルを訓練できる  
\- 訓練済みモデルの精度を評価し、エネルギー・力の予測誤差を確認できる  
\- MLP-MDシミュレーションを実行し、トラジェクトリを解析できる  
\- よくあるエラーと対処法を理解する

* * *

## 3.1 環境構築：必要なツールのインストール

MLPを実践するには、Python環境とSchNetPackのセットアップが必要です。

### 必要なソフトウェア

ツール | バージョン | 用途  
---|---|---  
**Python** | 3.9-3.11 | 基盤言語  
**PyTorch** | 2.0+ | ディープラーニングフレームワーク  
**SchNetPack** | 2.0+ | MLP訓練・推論  
**ASE** | 3.22+ | 原子構造操作、MD実行  
**NumPy/Matplotlib** | 最新版 | データ解析・可視化  
  
### インストール手順

**ステップ1: Conda環境の作成**
    
    
    # 新しいConda環境を作成（Python 3.10）
    conda create -n mlp-tutorial python=3.10 -y
    conda activate mlp-tutorial
    

**ステップ2: PyTorchのインストール**
    
    
    # CPU版（ローカルマシン、軽量）
    conda install pytorch cpuonly -c pytorch
    
    # GPU版（CUDAが利用可能な場合）
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    

**ステップ3: SchNetPackとASEのインストール**
    
    
    # SchNetPack（pip推奨）
    pip install schnetpack
    
    # ASE（原子シミュレーション環境）
    pip install ase
    
    # 可視化ツール
    pip install matplotlib seaborn
    

**ステップ4: 動作確認**
    
    
    # Example 1: 環境確認スクリプト（5行）
    import torch
    import schnetpack as spk
    print(f"PyTorch: {torch.__version__}")
    print(f"SchNetPack: {spk.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    

**期待される出力** :
    
    
    PyTorch: 2.1.0
    SchNetPack: 2.0.3
    GPU available: False  # CPUの場合
    

* * *

## 3.2 データ準備：MD17データセットの取得

SchNetPackは、小規模分子のベンチマークデータセット**MD17** を内蔵しています。

### MD17データセットとは

  * **内容** : DFT計算による分子動力学トラジェクトリ
  * **対象分子** : アスピリン、ベンゼン、エタノールなど10種類
  * **データ数** : 各分子約10万配置
  * **精度** : PBE/def2-SVP レベル（DFT）
  * **用途** : MLP手法のベンチマーク

### データのダウンロードと読み込み

**Example 2: MD17データセットのロード（10行）**
    
    
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
    

**出力** :
    
    
    Total samples: 211762
    Properties: ['energy', 'forces']
    First sample: {'_atomic_numbers': tensor([...]), 'energy': tensor(-1234.5), 'forces': tensor([...])}
    

### データの分割

**Example 3: 訓練/検証/テストセットの分割（10行）**
    
    
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
    

**説明** :  
\- `batch_size=32`: 32配置ずつまとめて処理（メモリ効率）  
\- `num_train=100000`: 大量データで汎化性能向上  
\- `split_file`: 分割をファイルに保存（再現性確保）

* * *

## 3.3 SchNetPackでのモデル訓練

SchNetモデルを訓練し、エネルギーと力を学習します。

### SchNetアーキテクチャの設定

**Example 4: SchNetモデルの定義（15行）**
    
    
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
    

**パラメータ解説** :  
\- `n_atom_basis=128`: 各原子の特徴ベクトルが128次元（典型的な値）  
\- `n_interactions=6`: 6層のメッセージパッシング（深いほど長距離相互作用を捉える）  
\- `cutoff=5.0Å`: この距離以上の原子間相互作用を無視（計算効率）

### 訓練の実行

**Example 5: 訓練ループの設定（15行）**
    
    
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
    

**訓練時間の目安** :  
\- CPU（4コア）: 約2-3時間（10万配置）  
\- GPU（RTX 3090）: 約15-20分

### 訓練の進捗確認

**Example 6: TensorBoardでの可視化（10行）**
    
    
    # TensorBoardの起動（別ターミナル）
    # tensorboard --logdir=./training/lightning_logs
    
    # Pythonからのログ確認
    import pandas as pd
    
    metrics = pd.read_csv('./training/lightning_logs/version_0/metrics.csv')
    print(metrics[['epoch', 'train_loss', 'val_loss']].tail(10))
    

**期待される出力** :
    
    
       epoch  train_loss  val_loss
    40    40      0.0023    0.0031
    41    41      0.0022    0.0030
    42    42      0.0021    0.0029
    ...
    

**観察ポイント** :  
\- `train_loss`と`val_loss`がともに減少 → 正常に学習中  
\- `val_loss`が増加し始めたら **過学習** の兆候 → Early Stoppingを検討

* * *

## 3.4 精度検証：エネルギーと力の予測精度

訓練したモデルがDFT精度を達成しているか評価します。

### テストセットでの評価

**Example 7: テストセット評価（12行）**
    
    
    # テストセットで評価
    test_results = trainer.test(task, datamodule=data_module)
    
    # 結果の表示
    print(f"Energy MAE: {test_results[0]['test_energy_MAE']:.4f} eV")
    print(f"Energy RMSE: {test_results[0]['test_energy_RMSE']:.4f} eV")
    
    # 力の評価（別途計算が必要）
    from schnetpack.metrics import MeanAbsoluteError
    force_mae = MeanAbsoluteError(target='forces')
    # ... 力の評価コード
    

**良好な精度の目安** （アスピリン分子、21原子）:  
\- **エネルギーMAE** : < 1 kcal/mol（< 0.043 eV）  
\- **力のMAE** : < 1 kcal/mol/Å（< 0.043 eV/Å）

### 予測値と真値の相関プロット

**Example 8: 予測精度の可視化（15行）**
    
    
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
    

**理想的な結果** :  
\- 点が赤い対角線（y=x）上に密集  
\- R² > 0.99（決定係数）

* * *

## 3.5 MLP-MDシミュレーション：分子動力学の実行

訓練したMLPを使って、DFTより10⁴倍高速なMDシミュレーションを実行します。

### ASEでのMLP-MD設定

**Example 9: MLP-MD計算の準備（10行）**
    
    
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
    

### 初期速度の設定と平衡化

**Example 10: 温度初期化（10行）**
    
    
    # 300Kでの速度分布を設定
    temperature = 300  # K
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # 運動量をゼロに（系全体の並進を除去）
    from ase.md.velocitydistribution import Stationary
    Stationary(atoms)
    
    print(f"Initial kinetic energy: {atoms.get_kinetic_energy():.3f} eV")
    print(f"Initial potential energy: {atoms.get_potential_energy():.3f} eV")
    

### MDシミュレーションの実行

**Example 11: MD実行とトラジェクトリ保存（12行）**
    
    
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
    

**計算時間の目安** :  
\- CPU（4コア）: 約5分（10,000ステップ）  
\- DFTなら: 約1週間（10,000ステップ）  
\- **10⁴倍の高速化達成！**

### トラジェクトリの解析

**Example 12: エネルギー保存とRDF計算（15行）**
    
    
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
    

**良好なシミュレーションの指標** :  
\- エネルギードリフト: < 0.001 eV/step  
\- 全エネルギーが時間とともに振動（保存則）

* * *

## 3.6 物性計算：振動スペクトルと拡散係数

MLP-MDから物理的な物性値を計算します。

### 振動スペクトル（パワースペクトル）

**Example 13: 振動スペクトル計算（15行）**
    
    
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
    

**解釈** :  
\- ピークが分子の振動モードに対応  
\- DFTで計算した振動スペクトルと比較することで精度検証

### 平均二乗変位（MSD）と拡散係数

**Example 14: MSD計算（15行）**
    
    
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
    

**拡散係数の計算** :
    
    
    # MSDの線形領域から拡散係数を計算（Einstein関係式）
    # D = lim_{t→∞} MSD(t) / (6t)
    linear_region = slice(100, 500)
    fit = np.polyfit(time_ps[linear_region], msd[linear_region], deg=1)
    D = fit[0] / 6  # Ų/ps → cm²/s変換が必要
    print(f"Diffusion coefficient: {D:.6f} Ų/ps")
    

* * *

## 3.7 Active Learning：効率的なデータ追加

モデルが不確実な配置を自動検出し、DFT計算を追加します。

### アンサンブル不確実性の評価

**Example 15: 予測の不確実性（15行）**
    
    
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
    

**次のステップ** :  
\- 不確実性の高い配置をDFT計算に追加  
\- データセットを更新してモデル再訓練  
\- 精度向上を確認

* * *

## 3.8 トラブルシューティング：よくあるエラーと対処法

実践でよく遭遇する問題と解決策を紹介します。

エラー | 原因 | 対処法  
---|---|---  
**Out of Memory (OOM)** | バッチサイズが大きすぎる | `batch_size`を32→16→8と減らす  
**Loss becomes NaN** | 学習率が高すぎる | `lr=1e-4`→`1e-5`に下げる  
**Energy drift in MD** | タイムステップが大きすぎる | `timestep=0.5fs`→`0.25fs`に減らす  
**Poor generalization** | 訓練データが偏っている | Active Learningでデータ多様化  
**CUDA error** | GPU互換性の問題 | PyTorchとCUDAバージョン確認  
  
### デバッグのベストプラクティス
    
    
    # 1. 小規模データでテスト
    data_module.num_train = 1000  # 1,000配置でクイックテスト
    
    # 2. 1バッチでのオーバーフィッティング確認
    trainer = pl.Trainer(max_epochs=100, overfit_batches=1)
    # 訓練誤差が0に近づけば、モデルに学習能力あり
    
    # 3. グラディエントのクリッピング
    task = AtomisticTask(..., gradient_clip_val=1.0)  # 勾配爆発防止
    

* * *

## 3.9 本章のまとめ

### 学んだこと

  1. **環境構築**  
\- Conda環境、PyTorch、SchNetPackのインストール  
\- GPU/CPU環境の選択

  2. **データ準備**  
\- MD17データセットのダウンロードと読み込み  
\- 訓練/検証/テストセットへの分割

  3. **モデル訓練**  
\- SchNetアーキテクチャの設定（6層、128次元）  
\- 50エポックの訓練（CPU: 2-3時間）  
\- TensorBoardでの進捗確認

  4. **精度検証**  
\- エネルギーMAE < 1 kcal/mol達成を確認  
\- 予測値vs真値の相関プロット  
\- R² > 0.99の高精度

  5. **MLP-MD実行**  
\- ASE Calculatorとしての統合  
\- 10,000ステップ（5ピコ秒）のMD実行  
\- DFTより10⁴倍高速化を体験

  6. **物性計算**  
\- 振動スペクトル（フーリエ変換）  
\- 拡散係数（平均二乗変位から計算）

  7. **Active Learning**  
\- アンサンブル不確実性による配置選択  
\- データ追加の自動化戦略

### 重要なポイント

  * **SchNetPackは実装が容易** : 数十行のコードでMLP訓練が可能
  * **小規模データ（10万配置）で実用精度達成** : MD17は優れたベンチマーク
  * **MLP-MDは実用的** : DFTの10⁴倍高速、個人のPCで実行可能
  * **Active Learningで効率化** : 重要な配置を自動発見、データ収集コスト削減

### 次の章へ

第4章では、最新のMLP手法（NequIP、MACE）と実際の研究応用例を学びます：  
\- E(3)等変グラフニューラルネットワークの理論  
\- データ効率の劇的向上（10万→3,000配置）  
\- 触媒反応、バッテリー材料への応用事例  
\- 大規模シミュレーション（100万原子）の実現

* * *

## 演習問題

### 問題1（難易度：easy）

Example 4のSchNet設定で、`n_interactions`（メッセージパッシング層の数）を3, 6, 9に変えて訓練し、テストMAEがどのように変化するか予測してください。

ヒント 層が深いほど、長距離の原子間相互作用を捉えられます。しかし、深すぎると過学習のリスクも。  解答例 **予測される結果**: | `n_interactions` | テストMAE予測 | 訓練時間 | 特徴 | |-----------------|-------------|---------|------| | **3** | 0.8-1.2 kcal/mol | 1時間 | 浅いため長距離相互作用を捉えきれない | | **6** | 0.5-0.8 kcal/mol | 2-3時間 | バランスが良い（推奨） | | **9** | 0.6-1.0 kcal/mol | 4-5時間 | 過学習リスク、訓練データ不足なら精度低下 | **実験方法**: 
    
    
    for n in [3, 6, 9]:
        representation = SchNet(n_interactions=n, ...)
        task = AtomisticTask(...)
        trainer.fit(task, datamodule=data_module)
        results = trainer.test(task, datamodule=data_module)
        print(f"n={n}: MAE={results[0]['test_energy_MAE']:.4f} eV")
    

**結論**: 小分子（アスピリン21原子）では`n_interactions=6`が最適。大規模系（100原子以上）では9-12層が有効な場合もある。 

### 問題2（難易度：medium）

Example 11のMLP-MDで、エネルギードリフトが許容範囲を超えた場合（例: 0.01 eV/step）、どのような対処法が考えられますか？3つ挙げてください。

ヒント タイムステップ、訓練精度、MDアルゴリズムの3つの観点から考えましょう。  解答例 **対処法1: タイムステップを小さくする** 
    
    
    timestep = 0.25 * units.fs  # 0.5fs → 0.25fsに半減
    dyn = VelocityVerlet(atoms, timestep=timestep)
    

\- **理由**: 小さいタイムステップは数値積分の誤差を減らす \- **デメリット**: 2倍の計算時間 **対処法2: モデル訓練精度を向上** 
    
    
    # より多くのデータで訓練
    data_module.num_train = 200000  # 10万→20万配置に増加
    
    # または力の損失関数の重みを増やす
    task = AtomisticTask(..., loss_weights={'energy': 1.0, 'forces': 1000})
    

\- **理由**: 力の予測精度が低いとMDが不安定 \- **目標**: 力のMAE < 0.05 eV/Å **対処法3: Langevin動力学に変更（熱浴結合）** 
    
    
    from ase.md.langevin import Langevin
    dyn = Langevin(atoms, timestep=0.5*units.fs,
                   temperature_K=300, friction=0.01)
    

\- **理由**: 熱浴がエネルギードリフトを吸収 \- **注意**: 厳密な微小正準アンサンブル（NVE）ではなくなる **優先順位**: 対処法2（精度向上）→ 対処法1（タイムステップ）→ 対処法3（Langevin） 

* * *

## 3.10 データライセンスと再現性

本章のハンズオンコードを再現するために必要なデータセットとツールのバージョン情報を記載します。

### 3.10.1 使用データセット

データセット | 説明 | ライセンス | アクセス方法  
---|---|---|---  
**MD17** | 小分子MD軌道（アスピリン、ベンゼンなど10種類） | CC0 1.0 (Public Domain) | SchNetPack内蔵 (`MD17(molecule='aspirin')`)  
**アスピリン分子** | 211,762配置、DFT（PBE/def2-SVP） | CC0 1.0 | [sgdml.org](<http://sgdml.org/#datasets>)  
  
**注意事項** :  
\- **商用利用** : CC0ライセンスにより、商用利用・改変・再配布すべて自由  
\- **論文引用** : MD17使用時は以下を引用  
Chmiela, S., et al. (2017). "Machine learning of accurate energy-conserving molecular force fields." _Science Advances_ , 3(5), e1603015.  
\- **データ完全性** : SchNetPackのダウンロード機能はSHA256チェックサムで検証

### 3.10.2 コード再現性のための環境情報

本章のコード例を正確に再現するために、以下のバージョンを使用してください。

ツール | 推奨バージョン | インストールコマンド | 互換性  
---|---|---|---  
**Python** | 3.10.x | `conda create -n mlp python=3.10` | 3.9-3.11で動作確認済み  
**PyTorch** | 2.1.0 | `conda install pytorch=2.1.0` | 2.0以上必須  
**SchNetPack** | 2.0.3 | `pip install schnetpack==2.0.3` | 2.0系と1.x系でAPIが異なる  
**ASE** | 3.22.1 | `pip install ase==3.22.1` | 3.20以上推奨  
**PyTorch Lightning** | 2.1.0 | `pip install pytorch-lightning==2.1.0` | SchNetPack 2.0.3と互換  
**NumPy** | 1.24.3 | `pip install numpy==1.24.3` | 1.20以上  
**Matplotlib** | 3.7.1 | `pip install matplotlib==3.7.1` | 3.5以上  
  
**環境ファイルの保存** :
    
    
    # 現在の環境を再現可能な形で保存
    conda env export > environment.yml
    
    # 他の環境で再現
    conda env create -f environment.yml
    

**Dockerによる再現性確保** （推奨）:
    
    
    # Dockerfile例
    FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    RUN pip install schnetpack==2.0.3 ase==3.22.1 pytorch-lightning==2.1.0
    

### 3.10.3 訓練ハイパーパラメータの完全記録

Example 4, 5で使用したハイパーパラメータの全記録（論文再現用）:

パラメータ | 値 | 説明  
---|---|---  
`n_atom_basis` | 128 | 原子特徴ベクトルの次元  
`n_interactions` | 6 | メッセージパッシング層の数  
`cutoff` | 5.0 Å | 原子間相互作用のカットオフ半径  
`n_filters` | 128 | 畳み込みフィルタの数  
`batch_size` | 32 | ミニバッチサイズ  
`learning_rate` | 1e-4 | 初期学習率（AdamW）  
`max_epochs` | 50 | 最大訓練エポック数  
`num_train` | 100,000 | 訓練データ数  
`num_val` | 10,000 | 検証データ数  
`num_test` | 10,000 | テストデータ数  
`random_seed` | 42 | 乱数シード（データ分割の再現性）  
  
**完全な再現コード** :
    
    
    import torch
    torch.manual_seed(42)  # 再現性確保
    
    representation = SchNet(
        n_atom_basis=128, n_interactions=6, cutoff=5.0, n_filters=128
    )
    task = AtomisticTask(
        model=AtomisticModel(representation, [output]),
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={'lr': 1e-4, 'weight_decay': 0.01}
    )
    trainer = pl.Trainer(max_epochs=50, deterministic=True)
    

### 3.10.4 エネルギー・力の単位換算表

本章で使用する物理量の単位換算（SchNetPackとASEの標準単位）:

物理量 | SchNetPack/ASE | eV | kcal/mol | Hartree  
---|---|---|---|---  
**エネルギー** | eV | 1.0 | 23.06 | 0.03674  
**力** | eV/Å | 1.0 | 23.06 | 0.01945  
**距離** | Å | - | - | 1.889726 Bohr  
**時間** | fs (フェムト秒) | - | - | 0.02419 a.u.  
  
**単位変換例** :
    
    
    from ase import units
    
    # エネルギー換算
    energy_ev = 1.0  # eV
    energy_kcal = energy_ev * 23.06052  # kcal/mol
    energy_hartree = energy_ev * 0.036749  # Hartree
    
    # ASEの単位定数を使用（推奨）
    print(f"{energy_ev} eV = {energy_ev * units.eV / units.kcal * units.mol} kcal/mol")
    

* * *

## 3.11 実践上の注意点：ハンズオンでの失敗パターン

### 3.11.1 環境構築とインストールの落とし穴

**失敗1: PyTorchとCUDAバージョンの不一致**

**問題** :
    
    
    RuntimeError: CUDA error: no kernel image is available for execution on the device
    

**原因** :  
PyTorch 2.1.0はCUDA 11.8または12.1でコンパイルされているが、システムのCUDAが10.2など古いバージョン

**診断コード** :
    
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    
    # システムCUDAバージョン確認（ターミナル）
    # nvcc --version
    

**対処法** :
    
    
    # システムCUDA 11.8の場合
    conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    
    # CUDA利用不可の場合はCPU版に切り替え
    conda install pytorch==2.1.0 cpuonly -c pytorch
    

**予防策** :  
環境構築前に `nvidia-smi` でGPUドライバとCUDAバージョンを確認する

**失敗2: SchNetPack 1.x と 2.x のAPI混同**

**問題** :
    
    
    AttributeError: module 'schnetpack' has no attribute 'AtomsData'
    

**原因** :  
SchNetPack 1.x系の古いチュートリアルコードを2.x系で実行

**バージョン確認** :
    
    
    import schnetpack as spk
    print(spk.__version__)  # 2.0.3なら本章のコードが動作
    

**主なAPI変更** :

SchNetPack 1.x | SchNetPack 2.x  
---|---  
`spk.AtomsData` | `spk.data.AtomsDataModule`  
`spk.atomistic.Atomwise` | `spk.task.ModelOutput`  
`spk.train.Trainer` | `pytorch_lightning.Trainer`  
  
**対処法** :  
本章のコード例（2.x系）を使用するか、SchNetPack公式ドキュメント（[schnetpack.readthedocs.io](<https://schnetpack.readthedocs.io>)）の2.x系チュートリアルを参照

**失敗3: メモリ不足（OOM）の誤診断**

**問題** :
    
    
    RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
    

**よくある誤解** :  
「GPUメモリ不足なのでGPUを買い替える必要がある」→ **間違い**

**診断手順** :
    
    
    # 1. バッチサイズを確認
    print(f"Current batch size: {data_module.batch_size}")
    
    # 2. GPUメモリ使用量を確認
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    

**対処法（優先順）** :

  1. **バッチサイズを減らす** : `batch_size=32` → `16` → `8` → `4`
  2. **勾配累積** : 小バッチを複数回累積して疑似的に大バッチ

    
    
    trainer = pl.Trainer(accumulate_grad_batches=4)  # 4バッチごとに更新
    

  3. **Mixed Precision訓練** : メモリ使用量を半減

    
    
    trainer = pl.Trainer(precision=16)  # float16使用
    

**目安** :  
\- GPU 4GB: batch_size=4-8  
\- GPU 8GB: batch_size=16-32  
\- GPU 24GB: batch_size=64-128

### 3.11.2 訓練とデバッグの落とし穴

**失敗4: 訓練誤差が減少しない（NaN損失）**

**問題** :
    
    
    Epoch 5: train_loss=nan, val_loss=nan
    

**原因トップ3** :

  1. **学習率が高すぎる** : 勾配爆発 → パラメータがNaNに
  2. **データ正規化の欠如** : エネルギーの絶対値が大きすぎる（例: -1000 eV）
  3. **力の損失係数が不適切** : 力の損失が支配的すぎる

**診断コード** :
    
    
    # 訓練開始直後にモデルの出力を確認
    for batch in data_module.train_dataloader():
        output = task.model(batch)
        print(f"Energy prediction: {output['energy'][:5]}")  # 最初の5サンプル
        print(f"Energy target: {batch['energy'][:5]}")
        break
    
    # NaNチェック
    print(f"Has NaN in prediction: {torch.isnan(output['energy']).any()}")
    

**対処法** :

  1. **学習率を下げる** :

    
    
    optimizer_args={'lr': 1e-5}  # 1e-4 → 1e-5に減少
    

  2. **勾配クリッピング** :

    
    
    trainer = pl.Trainer(gradient_clip_val=1.0)  # 勾配ノルムを1.0以下に
    

  3. **データ正規化** （SchNetPack 2.xは自動だが、手動確認）:

    
    
    import schnetpack.transform as trn
    data_module.train_transforms = [
        trn.SubtractCenterOfMass(),
        trn.RemoveOffsets('energy', remove_mean=True)  # エネルギーオフセット除去
    ]
    

**失敗5: 過学習の見逃し**

**問題** :  
訓練誤差は減少するが、検証誤差が停滞または増加
    
    
    Epoch 30: train_loss=0.001, val_loss=0.050  # val_lossが悪化
    

**原因** :  
モデルが訓練データを記憶し、未知データへの汎化性能が低下

**診断グラフ** :
    
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    metrics = pd.read_csv('./training/lightning_logs/version_0/metrics.csv')
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

**対処法** :

  1. **Early Stopping** （自動停止）:

    
    
    from pytorch_lightning.callbacks import EarlyStopping
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    trainer = pl.Trainer(callbacks=[early_stop])
    

  2. **データ増強** （訓練データを増やす）:

    
    
    data_module.num_train = 200000  # 10万 → 20万に増加
    

  3. **Dropout（非推奨、MLPでは効果薄い）** : 代わりにモデルのパラメータ数を減らす

    
    
    representation = SchNet(n_atom_basis=64, n_interactions=4)  # 128→64に削減
    

### 3.11.3 MLP-MDシミュレーションの落とし穴

**失敗6: エネルギードリフトの過小評価**

**問題** :  
MDシミュレーション中にエネルギーが単調増加/減少（保存則の破れ）

**許容範囲の誤解** :

  * 「少しくらいのドリフトは仕方ない」→ **危険**
  * 0.01 eV/stepのドリフトでも、10,000ステップで100 eVのエネルギー変化（非現実的）

**定量診断** :
    
    
    from ase.io import read
    
    traj = read('aspirin_md.traj', index=':')
    energies = [a.get_total_energy() for a in traj]
    
    # ドリフトの計算（線形フィット）
    import numpy as np
    time_steps = np.arange(len(energies))
    drift_rate, offset = np.polyfit(time_steps, energies, deg=1)
    print(f"Energy drift: {drift_rate:.6f} eV/step")
    
    # 許容範囲チェック
    if abs(drift_rate) > 0.001:
        print("⚠️ WARNING: Excessive energy drift detected!")
    

**対処法（問題2の詳細版）** :

  1. **力の訓練精度を向上** （最重要）:

    
    
    # 力の損失関数の重みを大幅に増加
    task = AtomisticTask(
        loss_weights={'energy': 0.01, 'forces': 0.99}  # 力に99%の重み
    )
    

  2. **タイムステップの最適化** :

    
    
    # 安定性テスト
    for dt in [0.1, 0.25, 0.5, 1.0]:  # fs
        # dt=0.5で安定、dt=1.0で発散なら、dt=0.5を採用
    

  3. **MLP精度の限界を認識** :  
力のMAE > 0.1 eV/Åの場合、長時間MD（>10 ps）は信頼性低下  
→ Active Learningで訓練データ追加

**失敗7: MD結果の物理的妥当性の未検証**

**問題** :  
「MDが完走したから成功」と誤解 → 実は非物理的な構造変化

**検証すべき項目** :

**1\. 温度制御の確認** :
    
    
    temperatures = [a.get_temperature() for a in traj]
    print(f"Average T: {np.mean(temperatures):.1f} K (target: 300 K)")
    print(f"Std T: {np.std(temperatures):.1f} K")
    # 標準偏差が30K以上なら異常
    

**2\. 構造の破壊チェック** :
    
    
    from ase.geometry.analysis import Analysis
    
    # 初期構造と最終構造の比較
    ana_init = Analysis(traj[0])
    ana_final = Analysis(traj[-1])
    
    # 結合が切れていないか確認
    bonds_init = ana_init.all_bonds[0]
    bonds_final = ana_final.all_bonds[0]
    print(f"Initial bonds: {len(bonds_init)}, Final bonds: {len(bonds_final)}")
    
    # 結合数が変化 → 構造破壊の可能性
    

**3\. 動径分布関数（RDF）の妥当性** :
    
    
    # 第一ピーク位置がDFT計算やX線回折データと一致するか確認
    # （実装は高度なため、省略）
    

**対処法** :  
物理的に妥当な結果が得られない場合、訓練データの範囲外（外挿）の可能性  
→ Active Learningで該当する配置を訓練データに追加

* * *

## 3.12 章末チェックリスト：ハンズオンの品質保証

この章を完了したら、以下の項目を確認してください。全てチェックできれば、実際の研究プロジェクトでMLPを活用する準備が整っています。

### 3.12.1 概念理解（Understanding）

**環境とツールの理解** :

  * □ SchNetPackの役割（MLP訓練ライブラリ）を説明できる
  * □ PyTorchとSchNetPackの関係（PyTorchベースのMLP実装）を理解している
  * □ ASEの役割（原子構造操作、MD実行）を説明できる
  * □ MD17データセットの特徴（10種類の小分子、DFT精度）を理解している

**モデル訓練の理解** :

  * □ SchNetのハイパーパラメータ（`n_atom_basis`、`n_interactions`、`cutoff`）の意味を説明できる
  * □ 訓練/検証/テストセットの役割を理解している
  * □ 過学習の兆候（検証誤差の増加）を識別できる
  * □ エネルギーMAE < 1 kcal/molが高精度の目安であることを理解している

**MLP-MDの理解** :

  * □ MLPがASE Calculatorとして統合される仕組みを理解している
  * □ エネルギー保存則とエネルギードリフトの違いを説明できる
  * □ タイムステップ（0.5 fs）が安定性に影響する理由を理解している
  * □ MLP-MDがDFTより10⁴倍高速な理由を説明できる

### 3.12.2 実践スキル（Doing）

**環境構築** :

  * □ Conda環境を作成し、Python 3.10をインストールできる
  * □ PyTorch（CPU版/GPU版）を正しくインストールできる
  * □ SchNetPack 2.0.3とASE 3.22.1をインストールできる
  * □ 環境確認スクリプトを実行し、バージョンを確認できる

**データ準備と訓練** :

  * □ MD17データセットをダウンロードし、10万配置に分割できる
  * □ SchNetモデルを定義し、ハイパーパラメータを設定できる
  * □ 50エポックの訓練を実行し、TensorBoardで進捗を確認できる
  * □ テストセットでMAEを評価し、目標精度（< 1 kcal/mol）を達成できる

**MLP-MDシミュレーション** :

  * □ 訓練済みモデルをASE Calculatorとしてラップできる
  * □ Maxwell-Boltzmann分布で初期速度を設定できる
  * □ 10,000ステップ（5ピコ秒）のMDを実行できる
  * □ トラジェクトリを保存し、エネルギー保存を確認できる

**解析とトラブルシューティング** :

  * □ 振動スペクトル（パワースペクトル）を計算できる
  * □ 平均二乗変位（MSD）から拡散係数を計算できる
  * □ Out of Memory（OOM）エラーに対処できる（バッチサイズ削減）
  * □ NaN損失の原因を診断し、学習率を調整できる

### 3.12.3 応用力（Applying）

**自分の研究への適用計画** :

  * □ 自分の研究対象（分子、材料）でMD17相当のデータセットを設計できる
  * □ 必要なDFT計算数（目標精度と系のサイズから）を見積もれる
  * □ SchNetのハイパーパラメータを自分の系に最適化する戦略を立てられる
  * □ MLP-MDで得たい物性（拡散係数、振動スペクトル、反応経路）を明確にできる

**問題解決とデバッグ** :

  * □ 訓練が収束しない場合の診断手順（学習率、データ正規化、勾配クリッピング）を実行できる
  * □ エネルギードリフトの原因を特定し、対処法を選択できる
  * □ 過学習を検出し、Early StoppingやData Augmentationを適用できる
  * □ GPU/CPUリソースに応じてバッチサイズと訓練時間を最適化できる

**Advanced技術への準備** :

  * □ Active Learning（Example 15）の概念を理解し、実装の流れを説明できる
  * □ アンサンブル不確実性による配置選択の重要性を理解している
  * □ 次章（NequIP、MACE）でデータ効率がどう改善されるか期待を持っている
  * □ SchNetPackのドキュメント（[schnetpack.readthedocs.io](<https://schnetpack.readthedocs.io>)）を活用して、独学で学習を続けられる

**次章へのブリッジ** :

  * □ SchNetの限界（データ効率、回転等変性）を認識している
  * □ E(3)等変性アーキテクチャ（NequIP、MACE）がどう改善するか興味を持っている
  * □ 実際の研究応用例（触媒、バッテリー、創薬）を第4章で学ぶ準備ができている

* * *

## 参考文献

  1. Schütt, K. T., et al. (2019). "SchNetPack: A Deep Learning Toolbox For Atomistic Systems." _Journal of Chemical Theory and Computation_ , 15(1), 448-455.  
DOI: [10.1021/acs.jctc.8b00908](<https://doi.org/10.1021/acs.jctc.8b00908>)

  2. Chmiela, S., et al. (2017). "Machine learning of accurate energy-conserving molecular force fields." _Science Advances_ , 3(5), e1603015.  
DOI: [10.1126/sciadv.1603015](<https://doi.org/10.1126/sciadv.1603015>)

  3. Larsen, A. H., et al. (2017). "The atomic simulation environment—a Python library for working with atoms." _Journal of Physics: Condensed Matter_ , 29(27), 273002.  
DOI: [10.1088/1361-648X/aa680e](<https://doi.org/10.1088/1361-648X/aa680e>)

  4. Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance deep learning library." _Advances in Neural Information Processing Systems_ , 32.  
arXiv: [1912.01703](<https://arxiv.org/abs/1912.01703>)

  5. Zhang, L., et al. (2020). "Active learning of uniformly accurate interatomic potentials for materials simulation." _Physical Review Materials_ , 3(2), 023804.  
DOI: [10.1103/PhysRevMaterials.3.023804](<https://doi.org/10.1103/PhysRevMaterials.3.023804>)

  6. Schütt, K. T., et al. (2017). "Quantum-chemical insights from deep tensor neural networks." _Nature Communications_ , 8(1), 13890.  
DOI: [10.1038/ncomms13890](<https://doi.org/10.1038/ncomms13890>)

* * *

## 著者情報

**作成者** : MI Knowledge Hub Content Team  
**作成日** : 2025-10-17  
**バージョン** : 1.1（Chapter 3 quality improvement）  
**シリーズ** : MLP入門シリーズ

**更新履歴** :  
\- 2025-10-19: v1.1 品質向上改訂  
\- データライセンスと再現性セクション追加（MD17データセット、アスピリン分子情報）  
\- コード再現性情報（Python 3.10.x, PyTorch 2.1.0, SchNetPack 2.0.3, ASE 3.22.1）  
\- 訓練ハイパーパラメータの完全記録（11項目、論文再現用）  
\- エネルギー・力の単位換算表（eV, kcal/mol, Hartree相互変換）  
\- 実践上の注意点セクション追加（7つの失敗パターン: CUDA不一致、API混同、OOM、NaN損失、過学習、エネルギードリフト、物理妥当性）  
\- 章末チェックリスト追加（概念理解12項目、実践スキル16項目、応用力16項目）  
\- 2025-10-17: v1.0 第3章初版作成  
\- Python環境構築（Conda, PyTorch, SchNetPack）  
\- MD17データセット準備と分割  
\- SchNetモデル訓練（15コード例）  
\- MLP-MD実行と解析（トラジェクトリ、振動スペクトル、MSD）  
\- Active Learning不確実性評価  
\- トラブルシューティング表（5項目）  
\- 演習問題2問（easy, medium）  
\- 参考文献6件

**ライセンス** : Creative Commons BY-NC-SA 4.0

[← シリーズ目次に戻る](<index.html>)
