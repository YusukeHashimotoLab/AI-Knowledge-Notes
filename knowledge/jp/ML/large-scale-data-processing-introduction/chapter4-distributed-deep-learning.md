---
title: 分散深層学習の基礎と実践
chapter_title: 分散深層学習の基礎と実践
subtitle: PyTorch DDPとHorovodによる大規模モデルの訓練
reading_time: 45-50分
difficulty: 中級～上級
code_examples: 8
exercises: 5
---

# 第4章：分散深層学習の基礎と実践

## 学習目標

  * 分散学習の主要戦略（Data/Model/Pipeline Parallelism）を理解する
  * PyTorch DDPによるマルチGPU訓練を実装できる
  * HorovodのAllReduceアーキテクチャを理解し、実装できる
  * 大規模モデル訓練のテクニック（AMP、Gradient Accumulation）を習得する
  * 分散学習のベストプラクティスとデバッグ手法を学ぶ

**読了時間** : 45-50分

\--- 

## 4.1 分散学習の戦略

### 4.1.1 なぜ分散学習が必要か

**現代の深層学習の課題:**

  * **モデルサイズの増大** : GPT-3（175B parameters）、BERT-Large（340M parameters）
  * **データセットの巨大化** : ImageNet-21K（14M画像）、Common Crawl（数百TB）
  * **訓練時間の問題** : 単一GPU訓練では数週間～数ヶ月

**分散学習による解決:**

  * **訓練時間の短縮** : 8 GPU並列で理想的には8倍高速化
  * **大規模モデルの実現** : メモリを複数GPU/ノードに分散
  * **コスト効率** : クラウド環境での効率的なリソース利用

### 4.1.2 Data Parallelism（データ並列）

**基本原理:**

  * モデルの完全なコピーを各GPUに配置
  * データバッチを分割して各GPUに配分
  * 各GPUで独立に順伝播・逆伝播
  * 勾配を全GPUで集約（AllReduce）
  * 統合された勾配でモデルを更新

**メリット:**

  * 実装が比較的簡単
  * モデルがGPUメモリに収まる場合に有効
  * 高いスケーラビリティ（数百GPUまで）

**デメリット:**

  * 各GPUにモデル全体が必要（メモリ制約）
  * 勾配同期の通信オーバーヘッド

### 4.1.3 Model Parallelism（モデル並列）

**基本原理:**

  * モデルを複数のGPUに分割
  * 各GPUが異なるレイヤー/パラメータを担当
  * データは全GPUで共有

**分割方法:**

  * **層単位分割** : レイヤー1-5をGPU0、6-10をGPU1
  * **テンソル分割** : 各レイヤーの重み行列を分割（Megatron-LM）

**メリット:**

  * GPUメモリを超える巨大モデルに対応
  * 勾配同期不要（層間通信のみ）

**デメリット:**

  * GPU間の依存関係で並列度が低下
  * 実装が複雑
  * 通信ボトルネック

### 4.1.4 Pipeline Parallelism（パイプライン並列）

**基本原理:**

  * モデルを複数ステージに分割（各GPUが担当）
  * データをマイクロバッチに分割
  * パイプライン的に順次処理
  * GPUのアイドル時間を削減

**GPipe手法:**

  * マイクロバッチ分割でパイプライン効率を向上
  * 勾配累積（Gradient Accumulation）と組み合わせ
  * 再計算（Recomputation）でメモリ削減

**メリット:**

  * モデル並列より高い並列度
  * GPU利用率の向上

**デメリット:**

  * パイプラインバブル（アイドル時間）
  * 実装の複雑さ

### 4.1.5 Hybrid Approaches（ハイブリッドアプローチ）

**3D Parallelism（Megatron-LM）:**

  * **Data Parallelism** : ノード間
  * **Model Parallelism** : ノード内GPU間（テンソル分割）
  * **Pipeline Parallelism** : レイヤー分割

**ZeRO（DeepSpeed）:**

  * オプティマイザ状態の分割（ZeRO-1）
  * 勾配の分割（ZeRO-2）
  * パラメータの分割（ZeRO-3）
  * Data Parallelismの効率を最大化

### 4.1.6 戦略の比較図
    
    
    ```mermaid
    graph TB
        subgraph "Data Parallelism"
            D1[GPU 0Model CopyData Batch 1]
            D2[GPU 1Model CopyData Batch 2]
            D3[GPU 2Model CopyData Batch 3]
            D1 -.AllReduce.-> D2
            D2 -.AllReduce.-> D3
        end
    
        subgraph "Model Parallelism"
            M1[GPU 0Layer 1-3]
            M2[GPU 1Layer 4-6]
            M3[GPU 2Layer 7-9]
            M1 -->|Forward| M2
            M2 -->|Forward| M3
            M3 -.Backward.-> M2
            M2 -.Backward.-> M1
        end
    
        subgraph "Pipeline Parallelism"
            P1[GPU 0Stage 1Micro-batch 1,2,3]
            P2[GPU 1Stage 2Micro-batch 1,2,3]
            P3[GPU 2Stage 3Micro-batch 1,2,3]
            P1 ==>|Pipeline| P2
            P2 ==>|Pipeline| P3
        end
    ```

\--- 

## 4.2 PyTorch Distributed Data Parallel (DDP)

### 4.2.1 torch.distributed の基礎

**主要概念:**

  * **Process Group** : 並列プロセスの集合
  * **Rank** : プロセスの一意なID（0, 1, 2, ...）
  * **World Size** : 総プロセス数
  * **Backend** : 通信ライブラリ（NCCL, Gloo, MPI）

**バックエンドの選択:**

  * **NCCL** : GPU間通信に最適（推奨）
  * **Gloo** : CPUとGPUの両方に対応
  * **MPI** : HPCクラスタで使用

#### コード例1: 基本的なDistributed初期化

distributed_init.py - 分散環境の初期化
    
    
    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    def setup(rank, world_size):
        """
        分散環境のセットアップ
    
        Args:
            rank: プロセスのランク（0からworld_size-1）
            world_size: 総プロセス数
        """
        # 環境変数の設定
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
        # プロセスグループの初期化
        dist.init_process_group(
            backend='nccl',  # GPU間通信にNCCL使用
            rank=rank,
            world_size=world_size
        )
    
        # 各プロセスを対応するGPUに割り当て
        torch.cuda.set_device(rank)
    
        print(f"Process {rank}/{world_size} initialized on GPU {rank}")
    
    def cleanup():
        """分散環境のクリーンアップ"""
        dist.destroy_process_group()
    
    def demo_basic_operations(rank, world_size):
        """
        基本的な分散操作のデモ
        """
        setup(rank, world_size)
    
        # 各プロセスでテンソルを作成
        tensor = torch.ones(2, 2).cuda(rank) * (rank + 1)
        print(f"Rank {rank} - Original tensor:\n{tensor}")
    
        # AllReduce: 全プロセスのテンソルを合計
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank} - After AllReduce:\n{tensor}")
    
        # Broadcast: Rank 0のテンソルを全プロセスに配布
        if rank == 0:
            broadcast_tensor = torch.tensor([100.0, 200.0]).cuda(rank)
        else:
            broadcast_tensor = torch.zeros(2).cuda(rank)
    
        dist.broadcast(broadcast_tensor, src=0)
        print(f"Rank {rank} - After Broadcast: {broadcast_tensor}")
    
        cleanup()
    
    if __name__ == "__main__":
        world_size = 4  # 4つのGPUを使用
        mp.spawn(
            demo_basic_operations,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    

**実行方法:**
    
    
    # 単一ノード、4 GPU
    python distributed_init.py
    
    # 複数ノード（ノードあたり4 GPU、2ノード）
    # Node 0:
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr="192.168.1.1" \
        --master_port=12355 \
        distributed_init.py
    
    # Node 1:
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr="192.168.1.1" \
        --master_port=12355 \
        distributed_init.py
    

### 4.2.2 DDP実装

#### コード例2: PyTorch DDPによる画像分類訓練

ddp_training.py - ResNet18のDDP訓練
    
    
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
    import torchvision
    import torchvision.transforms as transforms
    
    def setup(rank, world_size):
        """分散環境のセットアップ"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    def cleanup():
        """分散環境のクリーンアップ"""
        dist.destroy_process_group()
    
    def prepare_dataloader(rank, world_size, batch_size=32):
        """
        分散データローダーの準備
    
        DistributedSamplerを使用して各プロセスに異なるデータを割り当て
        """
        # データの前処理
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        # CIFAR-10データセット
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    
        # DistributedSampler: データをworld_size個のチャンクに分割
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
    
        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
        return dataloader, sampler
    
    def train_epoch(model, dataloader, optimizer, criterion, rank, epoch):
        """
        1エポックの訓練
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
    
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
    
            # 順伝播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
    
            # 逆伝播（DDPが自動的に勾配を同期）
            loss.backward()
            optimizer.step()
    
            # 統計
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.*correct/total:.2f}%")
    
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
    
        return avg_loss, accuracy
    
    def main(rank, world_size):
        """
        メイン訓練ループ
        """
        print(f"Running DDP on rank {rank}.")
        setup(rank, world_size)
    
        # モデルの作成
        model = torchvision.models.resnet18(num_classes=10).cuda(rank)
    
        # DDPラッパーでモデルをラップ
        model = DDP(model, device_ids=[rank])
    
        # 損失関数とオプティマイザ
        criterion = nn.CrossEntropyLoss().cuda(rank)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
    
        # 学習率スケジューラ
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200
        )
    
        # データローダーの準備
        dataloader, sampler = prepare_dataloader(rank, world_size, batch_size=128)
    
        # 訓練ループ
        num_epochs = 100
        for epoch in range(num_epochs):
            # エポック開始時にsamplerのシードを設定（シャッフルの再現性）
            sampler.set_epoch(epoch)
    
            # 訓練
            avg_loss, accuracy = train_epoch(
                model, dataloader, optimizer, criterion, rank, epoch
            )
    
            # 学習率更新
            scheduler.step()
    
            # Rank 0のみがログを出力
            if rank == 0:
                print(f"Epoch {epoch}/{num_epochs} - "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
                # モデル保存（Rank 0のみ）
                if (epoch + 1) % 10 == 0:
                    torch.save(
                        model.module.state_dict(),  # model.moduleで元のモデルにアクセス
                        f'checkpoint_epoch_{epoch+1}.pth'
                    )
    
        cleanup()
    
    if __name__ == "__main__":
        import torch.multiprocessing as mp
    
        world_size = torch.cuda.device_count()  # 利用可能なGPU数
        print(f"Training with {world_size} GPUs")
    
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    

**DDPの重要ポイント:**

  * **DistributedSampler** : 各プロセスに異なるデータを割り当て
  * **sampler.set_epoch()** : 各エポックでシャッフルを変える
  * **model.module** : DDPラッパーの元のモデルにアクセス
  * **Rank 0のみ保存** : モデル保存は1つのプロセスのみで実行

### 4.2.3 マルチノードGPU訓練

#### コード例3: SlurmによるマルチノードDDP

slurm_ddp.sh - Slurmスクリプト
    
    
    #!/bin/bash
    #SBATCH --job-name=ddp_training
    #SBATCH --nodes=4                    # 4ノード
    #SBATCH --ntasks-per-node=4          # ノードあたり4プロセス（4 GPU）
    #SBATCH --cpus-per-task=8            # プロセスあたり8 CPU
    #SBATCH --gres=gpu:4                 # ノードあたり4 GPU
    #SBATCH --time=24:00:00
    #SBATCH --output=logs/ddp_%j.out
    #SBATCH --error=logs/ddp_%j.err
    
    # モジュールのロード
    module load cuda/11.8
    module load anaconda3
    
    # 環境変数の設定
    export MASTER_PORT=12340
    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
    export WORLD_SIZE=$SLURM_NTASKS
    export NCCL_DEBUG=INFO
    
    # 各ノードで訓練を実行
    srun python -u ddp_training_multi_node.py \
        --epochs 100 \
        --batch-size 128 \
        --lr 0.1
    

ddp_training_multi_node.py - マルチノード対応版
    
    
    import os
    import argparse
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    def setup():
        """
        Slurm環境変数から分散設定を読み込み
        """
        # Slurmが設定する環境変数
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    
        # マスターアドレスとポート
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
    
        # 環境変数設定
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
    
        # 初期化
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
        # ローカルGPU設定
        torch.cuda.set_device(local_rank)
    
        return rank, world_size, local_rank
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.1)
        args = parser.parse_args()
    
        # 分散環境のセットアップ
        rank, world_size, local_rank = setup()
    
        if rank == 0:
            print(f"Training with {world_size} processes across "
                  f"{world_size // torch.cuda.device_count()} nodes")
    
        # モデル、データローダー、訓練ループは前述と同様
        # ...
    
        dist.destroy_process_group()
    
    if __name__ == "__main__":
        main()
    

\--- 

## 4.3 Horovod

### 4.3.1 AllReduceアーキテクチャ

**Horovodとは:**

  * Uber開発のオープンソース分散訓練フレームワーク
  * TensorFlow、PyTorch、Keras、MXNetに対応
  * MPIベースの効率的なAllReduce通信

**AllReduceの仕組み:**

  * **Ring-AllReduce** : データをリング状に通信
  * **通信量** : O(N)（Nは勾配サイズ）、プロセス数に依存しない
  * **帯域幅効率** : 全帯域を活用

#### Ring-AllReduceの動作
    
    
    ```mermaid
    sequenceDiagram
        participant GPU0
        participant GPU1
        participant GPU2
        participant GPU3
    
        Note over GPU0,GPU3: Step 1: Scatter-Reduce
        GPU0->>GPU1: Send chunk A
        GPU1->>GPU2: Send chunk B
        GPU2->>GPU3: Send chunk C
        GPU3->>GPU0: Send chunk D
    
        Note over GPU0,GPU3: Step 2: AllGather
        GPU0->>GPU1: Send reduced A
        GPU1->>GPU2: Send reduced B
        GPU2->>GPU3: Send reduced C
        GPU3->>GPU0: Send reduced D
    
        Note over GPU0,GPU3: All GPUs have complete reduced gradients
    ```

### 4.3.2 Horovod API

#### コード例4: HorovodによるPyTorch訓練

horovod_training.py - ResNet18のHorovod訓練
    
    
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import horovod.torch as hvd
    
    def train_horovod():
        """
        Horovodを使った分散訓練
        """
        # Horovodの初期化
        hvd.init()
    
        # 各プロセスを対応するGPUに割り当て
        torch.cuda.set_device(hvd.local_rank())
        device = torch.device('cuda')
    
        # データローダーの準備
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    
        # Horovod用サンプラー
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=hvd.size(),
            rank=hvd.rank()
        )
    
        train_loader = DataLoader(
            dataset,
            batch_size=128,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
    
        # モデルの作成
        model = torchvision.models.resnet18(num_classes=10).to(device)
    
        # オプティマイザ
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1 * hvd.size(),  # 学習率をワーカー数でスケール
            momentum=0.9,
            weight_decay=5e-4
        )
    
        # Horovodでオプティマイザをラップ
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=hvd.Compression.fp16,  # FP16圧縮で通信量削減
            op=hvd.Average  # 勾配の平均を取る
        )
    
        # 初期パラメータをブロードキャスト（全ワーカーで同じ初期値）
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
        # 損失関数
        criterion = nn.CrossEntropyLoss()
    
        # 訓練ループ
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            train_sampler.set_epoch(epoch)
    
            epoch_loss = 0.0
            correct = 0
            total = 0
    
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
    
                # Horovodが自動的に勾配をAllReduce
                optimizer.step()
    
                # 統計
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    
            # 全ワーカーで統計を集約
            epoch_loss = metric_average(epoch_loss, 'avg_loss')
            accuracy = metric_average(correct / total, 'avg_accuracy')
    
            # Rank 0のみログ出力
            if hvd.rank() == 0:
                print(f"Epoch {epoch}/{num_epochs} - "
                      f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    
                # モデル保存
                if (epoch + 1) % 10 == 0:
                    torch.save(model.state_dict(),
                              f'horovod_checkpoint_epoch_{epoch+1}.pth')
    
    def metric_average(val, name):
        """
        全ワーカーでメトリクスを平均
        """
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()
    
    if __name__ == "__main__":
        train_horovod()
    

**実行方法:**
    
    
    # 単一ノード、4 GPU
    horovodrun -np 4 python horovod_training.py
    
    # 複数ノード（各ノード4 GPU、2ノード）
    horovodrun -np 8 -H node1:4,node2:4 python horovod_training.py
    
    # Slurmクラスタ
    srun --ntasks=8 --gres=gpu:4 python horovod_training.py
    

### 4.3.3 TensorFlow/PyTorch統合

#### コード例5: HorovodによるTensorFlow訓練

horovod_tensorflow.py - TensorFlowでのHorovod使用
    
    
    import tensorflow as tf
    import horovod.tensorflow as hvd
    
    def train_tensorflow_horovod():
        """
        Horovod + TensorFlowでの分散訓練
        """
        # Horovod初期化
        hvd.init()
    
        # GPUメモリ成長を有効化
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()], 'GPU'
            )
    
        # データセット
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
        # データセットのシャーディング（各ワーカーに異なるデータ）
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
        dataset = dataset.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    
        # モデル
        model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            classes=10,
            input_shape=(32, 32, 3)
        )
    
        # オプティマイザ
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.1 * hvd.size(),
            momentum=0.9
        )
    
        # Horovodでオプティマイザをラップ
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            compression=hvd.Compression.fp16
        )
    
        # 損失関数とメトリクス
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
        @tf.function
        def training_step(images, labels, first_batch):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_fn(labels, predictions)
    
            # Horovodが勾配をAllReduce
            tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
            # 初回バッチでパラメータをブロードキャスト
            if first_batch:
                hvd.broadcast_variables(model.variables, root_rank=0)
                hvd.broadcast_variables(optimizer.variables(), root_rank=0)
    
            return loss
    
        # 訓練ループ
        for epoch in range(100):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(dataset):
                loss = training_step(images, labels, batch_idx == 0 and epoch == 0)
                epoch_loss += loss.numpy()
    
            # 平均損失を計算
            epoch_loss = hvd.allreduce(
                tf.constant(epoch_loss / len(dataset)),
                average=True
            ).numpy()
    
            if hvd.rank() == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
                # モデル保存
                if (epoch + 1) % 10 == 0:
                    model.save(f'tf_horovod_model_epoch_{epoch+1}.h5')
    
    if __name__ == "__main__":
        train_tensorflow_horovod()
    

### 4.3.4 性能比較: PyTorch DDP vs Horovod

項目 | PyTorch DDP | Horovod  
---|---|---  
**通信バックエンド** | NCCL, Gloo, MPI | MPI, NCCL  
**フレームワーク対応** | PyTorch専用 | TensorFlow, PyTorch, Keras, MXNet  
**実装の複雑さ** | 中程度 | シンプル  
**通信効率** | 高い（NCCL最適化） | 高い（Ring-AllReduce）  
**スケーラビリティ** | 数百GPU | 数千GPU（MPIベース）  
**勾配圧縮** | 手動実装 | 標準サポート（FP16）  
**動的グラフ対応** | 完全対応 | 完全対応  
**エコシステム** | PyTorch公式 | 独立プロジェクト  
  
**ベンチマーク結果（ResNet-50、ImageNet、8 GPU）:**

  * **PyTorch DDP** : 2,400 images/sec（スケーリング効率 92%）
  * **Horovod** : 2,350 images/sec（スケーリング効率 90%）

**推奨事項:**

  * **PyTorchのみ使用** → PyTorch DDP
  * **複数フレームワーク** → Horovod
  * **大規模クラスタ（100+ GPU）** → Horovod（MPIの安定性）

\--- 

## 4.4 大規模モデルの訓練テクニック

### 4.4.1 Gradient Accumulation（勾配累積）

**目的:**

  * GPUメモリ制約下で大きなバッチサイズを実現
  * 小バッチを複数回実行し、勾配を累積

**数式:**

$$ \nabla_\theta L_{\text{effective}} = \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta L(\text{mini-batch}_k) $$ 

$K$: 累積ステップ数、実効バッチサイズ = $K \times$ mini-batch size

#### コード例6: Gradient Accumulationの実装

gradient_accumulation.py - 勾配累積
    
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    
    def train_with_gradient_accumulation(
        model, dataloader, optimizer, criterion,
        accumulation_steps=4, device='cuda'
    ):
        """
        勾配累積を使った訓練
    
        Args:
            accumulation_steps: 勾配を累積するステップ数
        """
        model.train()
        optimizer.zero_grad()
    
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
    
            # 順伝播
            output = model(data)
            loss = criterion(output, target)
    
            # 損失を累積ステップ数で割る
            loss = loss / accumulation_steps
    
            # 逆伝播（勾配を累積）
            loss.backward()
    
            # accumulation_stepsごとにパラメータ更新
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    
                print(f"Batch {batch_idx+1}, Updated parameters")
    
        # 最後の余りのバッチを処理
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # 使用例
    model = torchvision.models.resnet50().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 小バッチサイズ（16）× 累積ステップ（4）= 実効バッチサイズ（64）
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    train_with_gradient_accumulation(
        model, train_loader, optimizer, criterion,
        accumulation_steps=4
    )
    

**メリット:**

  * GPUメモリ制約を回避
  * 大バッチサイズの訓練を可能に

**デメリット:**

  * 訓練速度の低下（通信オーバーヘッド削減の効果は小）
  * Batch Normalizationの挙動に注意（小バッチでの統計）

### 4.4.2 Mixed Precision Training (AMP)

**概要:**

  * FP16（半精度浮動小数点）とFP32を混在させて訓練
  * 計算高速化とメモリ削減
  * 損失スケーリングで数値安定性を確保

**効果:**

  * **高速化** : 1.5～2倍（Tensor Core活用）
  * **メモリ削減** : 約50%

#### コード例7: PyTorch AMPの使用

amp_training.py - Automatic Mixed Precision
    
    
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    import torchvision
    
    def train_with_amp(model, dataloader, optimizer, criterion, device='cuda'):
        """
        Automatic Mixed Precision (AMP) を使った訓練
        """
        model.train()
    
        # GradScaler: FP16の勾配を適切にスケール
        scaler = GradScaler()
    
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
    
            optimizer.zero_grad()
    
            # autocast: 自動的に最適な精度を選択
            with autocast():
                output = model(data)
                loss = criterion(output, target)
    
            # スケールした損失で逆伝播
            scaler.scale(loss).backward()
    
            # 勾配をアンスケールしてパラメータ更新
            scaler.step(optimizer)
            scaler.update()
    
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # 使用例
    model = torchvision.models.resnet50().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
    
    train_with_amp(model, train_loader, optimizer, criterion)
    

**AMP + Gradient Accumulation + DDP** の組み合わせ:

amp_grad_accum_ddp.py - 完全な最適化
    
    
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    def train_optimized(
        rank, world_size, model, dataset,
        batch_size=32, accumulation_steps=4, epochs=100
    ):
        """
        AMP + Gradient Accumulation + DDP の完全な実装
        """
        # 分散環境セットアップ
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
        # モデルをDDPでラップ
        model = model.cuda(rank)
        model = DDP(model, device_ids=[rank])
    
        # データローダー
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=4
        )
    
        # オプティマイザと損失関数
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
    
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad()
    
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.cuda(rank), target.cuda(rank)
    
                # AMPによる順伝播
                with autocast():
                    output = model(data)
                    loss = criterion(output, target) / accumulation_steps
    
                # 勾配累積
                scaler.scale(loss).backward()
    
                # 累積ステップごとにパラメータ更新
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
    
            # 最後の余りを処理
            if (batch_idx + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
            if rank == 0:
                print(f"Epoch {epoch} completed")
    
        dist.destroy_process_group()
    
    # 実行
    if __name__ == "__main__":
        import torch.multiprocessing as mp
    
        world_size = torch.cuda.device_count()
        model = torchvision.models.resnet50()
    
        mp.spawn(
            train_optimized,
            args=(world_size, model, dataset),
            nprocs=world_size,
            join=True
        )
    

### 4.4.3 Gradient Checkpointing（勾配チェックポイント）

**概要:**

  * 順伝播時に中間活性化を保存せず、逆伝播時に再計算
  * メモリ使用量を大幅削減（計算時間とのトレードオフ）

**効果:**

  * **メモリ削減** : O(n) → O(√n)（nはレイヤー数）
  * **計算増加** : 約20-30%

    
    
    from torch.utils.checkpoint import checkpoint
    
    class CheckpointedResNet(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.layer1 = original_model.layer1
            self.layer2 = original_model.layer2
            self.layer3 = original_model.layer3
            self.layer4 = original_model.layer4
            self.fc = original_model.fc
    
        def forward(self, x):
            # 各レイヤーでcheckpoint使用
            x = checkpoint(self.layer1, x)
            x = checkpoint(self.layer2, x)
            x = checkpoint(self.layer3, x)
            x = checkpoint(self.layer4, x)
            x = self.fc(x)
            return x
    

### 4.4.4 DeepSpeed ZeRO

**ZeRO（Zero Redundancy Optimizer）:**

  * Microsoft開発の超大規模モデル訓練フレームワーク
  * オプティマイザ状態、勾配、パラメータを分散

**ZeROの3段階:**

  * **ZeRO-1** : オプティマイザ状態の分割（4倍メモリ削減）
  * **ZeRO-2** : + 勾配の分割（8倍メモリ削減）
  * **ZeRO-3** : + パラメータの分割（NワーカーでN倍削減）

#### コード例8: DeepSpeed ZeROの使用

deepspeed_zero.py - DeepSpeed訓練
    
    
    import torch
    import torch.nn as nn
    import deepspeed
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    def train_with_deepspeed():
        """
        DeepSpeed ZeROを使った大規模モデル訓練
        """
        # DeepSpeed設定ファイル
        ds_config = {
            "train_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 3,  # ZeRO-3: 最大メモリ削減
                "offload_optimizer": {
                    "device": "cpu",  # オプティマイザ状態をCPUにオフロード
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",  # パラメータをCPUにオフロード
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_fp16_weights_on_model_save": True
            },
            "gradient_clipping": 1.0,
            "wall_clock_breakdown": False
        }
    
        # 大規模モデル（GPT-2 Large: 774M parameters）
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    
        # DeepSpeedエンジンの初期化
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
    
        # データローダー
        train_loader = ...  # データローダー準備
    
        # 訓練ループ
        for epoch in range(10):
            for batch in train_loader:
                inputs = batch['input_ids'].to(model_engine.local_rank)
                labels = batch['labels'].to(model_engine.local_rank)
    
                # 順伝播
                outputs = model_engine(inputs, labels=labels)
                loss = outputs.loss
    
                # DeepSpeedが自動で逆伝播とパラメータ更新
                model_engine.backward(loss)
                model_engine.step()
    
        # モデル保存（全パラメータを集約）
        model_engine.save_checkpoint('./checkpoints')
    
    if __name__ == "__main__":
        # DeepSpeed起動
        # deepspeed --num_gpus=8 deepspeed_zero.py
        train_with_deepspeed()
    

**実行コマンド:**
    
    
    # 単一ノード、8 GPU
    deepspeed --num_gpus=8 deepspeed_zero.py
    
    # 複数ノード（2ノード、各8 GPU）
    deepspeed --num_nodes=2 --num_gpus=8 --hostfile=hostfile deepspeed_zero.py
    

**ZeRO-3の効果（GPT-3 175B parameters）:**

  * **従来のDDP** : 訓練不可（GPUメモリ不足）
  * **ZeRO-3** : 128 GPU（各40GB）で訓練可能
  * **メモリ削減** : 64倍（128ワーカー）

\--- 

## 4.5 分散学習のベストプラクティス

### 4.5.1 通信最適化

**勾配バケッティング（Gradient Bucketing）:**

  * 小さな勾配をまとめて通信（レイテンシ削減）
  * PyTorch DDPはデフォルトで有効

    
    
    # DDPのバケットサイズ設定
    model = DDP(
        model,
        device_ids=[rank],
        bucket_cap_mb=25,  # バケットサイズ（MB）
        find_unused_parameters=False  # 使用しないパラメータ検出を無効化
    )
    

**勾配圧縮:**

  * FP16圧縮で通信量50%削減
  * スパース化、量子化

    
    
    # Horovodのfp16圧縮
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        compression=hvd.Compression.fp16
    )
    

**階層的AllReduce:**

  * ノード内NCCL → ノード間MPI
  * Horovod Hierarchical AllReduce

### 4.5.2 バッチサイズと学習率のスケーリング

**Linear Scaling Rule（線形スケーリング則）:**

$$ \text{LR}_{\text{distributed}} = \text{LR}_{\text{base}} \times \frac{\text{Batch}_{\text{distributed}}}{\text{Batch}_{\text{base}}} $$ 

**例:**

  * ベース: LR=0.1, Batch=256（単一GPU）
  * 8 GPU: LR=0.8, Batch=2,048（256×8）

**Warmup（ウォームアップ）:**

  * 最初のエポックで学習率を徐々に増加
  * 大バッチサイズでの訓練安定化

    
    
    def linear_warmup_cosine_decay(step, warmup_steps, total_steps, base_lr, max_lr):
        """
        Warmup + Cosine Decay 学習率スケジューラ
        """
        if step < warmup_steps:
            # Warmup: 線形増加
            lr = base_lr + (max_lr - base_lr) * step / warmup_steps
        else:
            # Cosine Decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = base_lr + 0.5 * (max_lr - base_lr) * (1 + math.cos(math.pi * progress))
        return lr
    

### 4.5.3 学習率調整のベストプラクティス

**LARS（Layer-wise Adaptive Rate Scaling）:**

  * レイヤーごとに学習率を適応調整
  * 超大バッチサイズ（32K～64K）で有効

**LAMB（Layer-wise Adaptive Moments optimizer for Batch training）:**

  * BERT訓練でバッチサイズ65,536を実現
  * AdamベースのLARS拡張

### 4.5.4 デバッグ分散コード

**よくあるエラー:**

**1\. デッドロック:**

  * 全プロセスで同じ集団通信を実行
  * 一部のプロセスのみが通信するとハング

    
    
    # 悪い例: Rank 0のみがallreduce
    if rank == 0:
        dist.all_reduce(tensor)  # 他のプロセスが待機 → デッドロック
    
    # 正しい例: 全プロセスがallreduce
    dist.all_reduce(tensor)
    if rank == 0:
        print(tensor)
    

**2\. メモリリーク:**

  * 勾配累積時のdetach忘れ
  * 計算グラフが保持され続ける

    
    
    # 勾配累積時の正しい実装
    loss = loss / accumulation_steps
    loss.backward()
    
    # メトリクス計算時はdetach
    total_loss += loss.detach().item()
    

**3\. 再現性の欠如:**

  * シード設定が不十分
  * 各プロセスで異なる初期化

    
    
    def set_seed(seed, rank):
        """
        再現性のためのシード設定
        """
        torch.manual_seed(seed + rank)  # ランクごとに異なるシード
        torch.cuda.manual_seed_all(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
    
        # CuDNNの決定的動作（速度低下あり）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

**デバッグツール:**

**NCCL環境変数:**
    
    
    export NCCL_DEBUG=INFO           # NCCLのデバッグ情報
    export NCCL_DEBUG_SUBSYS=ALL     # 全サブシステムをログ
    export NCCL_IB_DISABLE=1         # InfiniBandを無効化（デバッグ時）
    export NCCL_P2P_DISABLE=1        # P2P通信を無効化
    

**PyTorchプロファイラ:**
    
    
    from torch.profiler import profile, ProfilerActivity
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    

\--- 

## 4.6 まとめ

### 学んだこと

  1. **分散学習の戦略:**

     * Data Parallelism: モデルコピー、データ分割、勾配集約
     * Model Parallelism: モデル分割、メモリ制約の解決
     * Pipeline Parallelism: ステージ分割、GPU利用率向上
     * Hybrid Approaches: 3D Parallelism、ZeRO
  2. **PyTorch DDP:**

     * torch.distributedの基礎（Rank、World Size、Backend）
     * DistributedSamplerでデータ分割
     * DDPラッパーで自動勾配同期
     * マルチノード訓練（Slurm、SSH）
  3. **Horovod:**

     * Ring-AllReduceアーキテクチャ
     * MPI/NCCLベースの効率的通信
     * TensorFlow/PyTorch/Keras対応
     * FP16圧縮でさらなる高速化
  4. **大規模モデル訓練:**

     * Gradient Accumulation: メモリ制約の回避
     * Mixed Precision (AMP): 1.5～2倍高速化、50%メモリ削減
     * Gradient Checkpointing: O(n) → O(√n)メモリ削減
     * DeepSpeed ZeRO: 超大規模モデル（175B parameters）訓練
  5. **ベストプラクティス:**

     * 通信最適化: バケッティング、圧縮、階層的AllReduce
     * バッチサイズスケーリング: Linear Scaling Rule、Warmup
     * 学習率調整: LARS、LAMB
     * デバッグ: デッドロック回避、メモリリーク対策、再現性

### 次のステップ

第5章では、実世界の大規模データ処理アプリケーションを学びます:

  * 推薦システムの分散訓練（Netflix、Amazon規模）
  * 大規模言語モデルの事前学習（BERT、GPT）
  * リアルタイムストリーム処理（Apache Kafka + Spark Streaming）
  * マテリアルズ・インフォマティクスでの大規模スクリーニング

\--- 

## 演習問題

**問1:** Data ParallelismとModel Parallelismの違いを、メモリ使用量と通信パターンの観点から説明せよ。

**問2:** 8 GPUで訓練する場合、単一GPUでバッチサイズ32、学習率0.1だった場合、分散訓練での適切なバッチサイズと学習率を計算せよ。

**問3:** Gradient Accumulationを使って、実効バッチサイズ256を実現したい。GPUメモリ制約でバッチサイズ32しか扱えない場合、accumulation_stepsをいくつに設定すべきか。

**問4:** Mixed Precision Training (AMP) が数値安定性を保つために使用する「損失スケーリング」の仕組みを説明せよ。

**問5:** DeepSpeed ZeRO-3が、従来のData Parallelismと比べてメモリ効率が高い理由を、オプティマイザ状態、勾配、パラメータの分散という観点から論じよ（500字以内）。

\--- 

## 参考文献

  1. Goyal, P. et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." _arXiv:1706.02677_ (2017).
  2. Sergeev, A. & Del Balso, M. "Horovod: fast and easy distributed deep learning in TensorFlow." _arXiv:1802.05799_ (2018).
  3. Li, S. et al. "PyTorch Distributed: Experiences on Accelerating Data Parallel Training." _VLDB_ (2020).
  4. Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." _SC'20_ (2020).
  5. Huang, Y. et al. "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." _NeurIPS_ (2019).
  6. You, Y. et al. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." _ICLR_ (2020).
  7. Micikevicius, P. et al. "Mixed Precision Training." _ICLR_ (2018).
  8. Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." _arXiv:1909.08053_ (2019).

\--- 

**次章** : [第5章：実世界の大規模データ処理アプリケーション](<chapter5-large-scale-ml-pipeline.html>)

**ライセンス** : このコンテンツはCC BY 4.0ライセンスの下で提供されています。
