---
title: Fundamentals and Practice of Distributed Deep Learning
chapter_title: Fundamentals and Practice of Distributed Deep Learning
subtitle: Training Large-Scale Models with PyTorch DDP and Horovod
reading_time: 45-50 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 5
version: 1.0
---

# Chapter 4: Fundamentals and Practice of Distributed Deep Learning

This chapter covers the fundamentals of Fundamentals and Practice of Distributed Deep Learning, which distributed learning strategies. You will learn multi-GPU training with PyTorch DDP, and implement Horovod's AllReduce architecture, and large-scale model training techniques (AMP.

## Learning Objectives

  * Understand the main distributed learning strategies (Data/Model/Pipeline Parallelism)
  * Implement multi-GPU training with PyTorch DDP
  * Understand and implement Horovod's AllReduce architecture
  * Master large-scale model training techniques (AMP, Gradient Accumulation)
  * Learn best practices and debugging methods for distributed learning

**Reading time** : 45-50 minutes

\--- 

## 4.1 Distributed Learning Strategies

### 4.1.1 Why Distributed Learning is Necessary

**Challenges in modern deep learning:**

  * **Increasing model sizes** : GPT-3 (175B parameters), BERT-Large (340M parameters)
  * **Massive datasets** : ImageNet-21K (14M images), Common Crawl (hundreds of TB)
  * **Training time issues** : Single GPU training can take weeks to months

**Solutions through distributed learning:**

  * **Reduced training time** : Ideally 8x speedup with 8 GPU parallelization
  * **Enabling large-scale models** : Distributing memory across multiple GPUs/nodes
  * **Cost efficiency** : Efficient resource utilization in cloud environments

### 4.1.2 Data Parallelism

**Basic principle:**

  * Place a complete copy of the model on each GPU
  * Split data batches and distribute to each GPU
  * Independent forward and backward propagation on each GPU
  * Aggregate gradients across all GPUs (AllReduce)
  * Update model with unified gradients

**Advantages:**

  * Relatively simple implementation
  * Effective when the model fits in GPU memory
  * High scalability (up to hundreds of GPUs)

**Disadvantages:**

  * Entire model required on each GPU (memory constraint)
  * Communication overhead for gradient synchronization

### 4.1.3 Model Parallelism

**Basic principle:**

  * Split the model across multiple GPUs
  * Each GPU handles different layers/parameters
  * Data is shared across all GPUs

**Splitting methods:**

  * **Layer-wise splitting** : Layers 1-5 on GPU0, 6-10 on GPU1
  * **Tensor splitting** : Split weight matrices of each layer (Megatron-LM)

**Advantages:**

  * Support for huge models exceeding GPU memory
  * No gradient synchronization needed (only inter-layer communication)

**Disadvantages:**

  * Reduced parallelism due to inter-GPU dependencies
  * Complex implementation
  * Communication bottleneck

### 4.1.4 Pipeline Parallelism

**Basic principle:**

  * Split model into multiple stages (each handled by a GPU)
  * Divide data into micro-batches
  * Process sequentially in a pipeline fashion
  * Reduce GPU idle time

**GPipe method:**

  * Improve pipeline efficiency with micro-batch splitting
  * Combine with gradient accumulation
  * Memory reduction through recomputation

**Advantages:**

  * Higher parallelism than model parallelism
  * Improved GPU utilization

**Disadvantages:**

  * Pipeline bubble (idle time)
  * Implementation complexity

### 4.1.5 Hybrid Approaches

**3D Parallelism (Megatron-LM):**

  * **Data Parallelism** : Between nodes
  * **Model Parallelism** : Between GPUs within a node (tensor splitting)
  * **Pipeline Parallelism** : Layer splitting

**ZeRO (DeepSpeed):**

  * Optimizer state partitioning (ZeRO-1)
  * Gradient partitioning (ZeRO-2)
  * Parameter partitioning (ZeRO-3)
  * Maximize Data Parallelism efficiency

### 4.1.6 Strategy Comparison Diagram
    
    
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

### 4.2.1 torch.distributed Basics

**Key concepts:**

  * **Process Group** : Collection of parallel processes
  * **Rank** : Unique ID of a process (0, 1, 2, ...)
  * **World Size** : Total number of processes
  * **Backend** : Communication library (NCCL, Gloo, MPI)

**Backend selection:**

  * **NCCL** : Optimal for inter-GPU communication (recommended)
  * **Gloo** : Supports both CPU and GPU
  * **MPI** : Used in HPC clusters

#### Code Example 1: Basic Distributed Initialization

distributed_init.py - Distributed environment initialization
    
    
    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    def setup(rank, world_size):
        """
        Setup distributed environment
    
        Args:
            rank: Process rank (0 to world_size-1)
            world_size: Total number of processes
        """
        # Set environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL for inter-GPU communication
            rank=rank,
            world_size=world_size
        )
    
        # Assign each process to corresponding GPU
        torch.cuda.set_device(rank)
    
        print(f"Process {rank}/{world_size} initialized on GPU {rank}")
    
    def cleanup():
        """Cleanup distributed environment"""
        dist.destroy_process_group()
    
    def demo_basic_operations(rank, world_size):
        """
        Demo of basic distributed operations
        """
        setup(rank, world_size)
    
        # Create tensor on each process
        tensor = torch.ones(2, 2).cuda(rank) * (rank + 1)
        print(f"Rank {rank} - Original tensor:\n{tensor}")
    
        # AllReduce: Sum tensors from all processes
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank} - After AllReduce:\n{tensor}")
    
        # Broadcast: Distribute Rank 0's tensor to all processes
        if rank == 0:
            broadcast_tensor = torch.tensor([100.0, 200.0]).cuda(rank)
        else:
            broadcast_tensor = torch.zeros(2).cuda(rank)
    
        dist.broadcast(broadcast_tensor, src=0)
        print(f"Rank {rank} - After Broadcast: {broadcast_tensor}")
    
        cleanup()
    
    if __name__ == "__main__":
        world_size = 4  # Use 4 GPUs
        mp.spawn(
            demo_basic_operations,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    

**Execution method:**
    
    
    # Single node, 4 GPUs
    python distributed_init.py
    
    # Multiple nodes (4 GPUs per node, 2 nodes)
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
    

### 4.2.2 DDP Implementation

#### Code Example 2: Image Classification Training with PyTorch DDP

ddp_training.py - ResNet18 DDP training
    
    
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
        """Setup distributed environment"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    def cleanup():
        """Cleanup distributed environment"""
        dist.destroy_process_group()
    
    def prepare_dataloader(rank, world_size, batch_size=32):
        """
        Prepare distributed dataloader
    
        Use DistributedSampler to assign different data to each process
        """
        # Data preprocessing
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        # CIFAR-10 dataset
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    
        # DistributedSampler: Split data into world_size chunks
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
        Train for one epoch
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
    
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
    
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
    
            # Backward pass (DDP automatically synchronizes gradients)
            loss.backward()
            optimizer.step()
    
            # Statistics
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
        Main training loop
        """
        print(f"Running DDP on rank {rank}.")
        setup(rank, world_size)
    
        # Create model
        model = torchvision.models.resnet18(num_classes=10).cuda(rank)
    
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank])
    
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss().cuda(rank)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
    
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200
        )
    
        # Prepare dataloader
        dataloader, sampler = prepare_dataloader(rank, world_size, batch_size=128)
    
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            # Set sampler seed at epoch start (for shuffle reproducibility)
            sampler.set_epoch(epoch)
    
            # Train
            avg_loss, accuracy = train_epoch(
                model, dataloader, optimizer, criterion, rank, epoch
            )
    
            # Update learning rate
            scheduler.step()
    
            # Only Rank 0 outputs logs
            if rank == 0:
                print(f"Epoch {epoch}/{num_epochs} - "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
                # Save model (Rank 0 only)
                if (epoch + 1) % 10 == 0:
                    torch.save(
                        model.module.state_dict(),  # Access original model via model.module
                        f'checkpoint_epoch_{epoch+1}.pth'
                    )
    
        cleanup()
    
    if __name__ == "__main__":
        import torch.multiprocessing as mp
    
        world_size = torch.cuda.device_count()  # Number of available GPUs
        print(f"Training with {world_size} GPUs")
    
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    

**Important DDP points:**

  * **DistributedSampler** : Assigns different data to each process
  * **sampler.set_epoch()** : Changes shuffle for each epoch
  * **model.module** : Accesses original model within DDP wrapper
  * **Save only on Rank 0** : Model saving should be executed by only one process

### 4.2.3 Multi-Node GPU Training

#### Code Example 3: Multi-Node DDP with Slurm

slurm_ddp.sh - Slurm script
    
    
    #!/bin/bash
    #SBATCH --job-name=ddp_training
    #SBATCH --nodes=4                    # 4 nodes
    #SBATCH --ntasks-per-node=4          # 4 processes per node (4 GPUs)
    #SBATCH --cpus-per-task=8            # 8 CPUs per process
    #SBATCH --gres=gpu:4                 # 4 GPUs per node
    #SBATCH --time=24:00:00
    #SBATCH --output=logs/ddp_%j.out
    #SBATCH --error=logs/ddp_%j.err
    
    # Load modules
    module load cuda/11.8
    module load anaconda3
    
    # Set environment variables
    export MASTER_PORT=12340
    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
    export WORLD_SIZE=$SLURM_NTASKS
    export NCCL_DEBUG=INFO
    
    # Execute training on each node
    srun python -u ddp_training_multi_node.py \
        --epochs 100 \
        --batch-size 128 \
        --lr 0.1
    

ddp_training_multi_node.py - Multi-node compatible version
    
    
    import os
    import argparse
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    def setup():
        """
        Read distributed settings from Slurm environment variables
        """
        # Environment variables set by Slurm
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    
        # Master address and port
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
    
        # Set environment variables
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
    
        # Initialize
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
        # Local GPU setting
        torch.cuda.set_device(local_rank)
    
        return rank, world_size, local_rank
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.1)
        args = parser.parse_args()
    
        # Setup distributed environment
        rank, world_size, local_rank = setup()
    
        if rank == 0:
            print(f"Training with {world_size} processes across "
                  f"{world_size // torch.cuda.device_count()} nodes")
    
        # Model, dataloader, and training loop same as before
        # ...
    
        dist.destroy_process_group()
    
    if __name__ == "__main__":
        main()
    

\--- 

## 4.3 Horovod

### 4.3.1 AllReduce Architecture

**What is Horovod:**

  * Open-source distributed training framework developed by Uber
  * Supports TensorFlow, PyTorch, Keras, and MXNet
  * Efficient AllReduce communication based on MPI

**AllReduce mechanism:**

  * **Ring-AllReduce** : Communicate data in a ring topology
  * **Communication volume** : O(N) (N is gradient size), independent of number of processes
  * **Bandwidth efficiency** : Utilizes full bandwidth

#### Ring-AllReduce Operation
    
    
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

#### Code Example 4: PyTorch Training with Horovod

horovod_training.py - ResNet18 Horovod training
    
    
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import horovod.torch as hvd
    
    def train_horovod():
        """
        Distributed training using Horovod
        """
        # Initialize Horovod
        hvd.init()
    
        # Assign each process to corresponding GPU
        torch.cuda.set_device(hvd.local_rank())
        device = torch.device('cuda')
    
        # Prepare dataloader
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
    
        # Horovod sampler
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
    
        # Create model
        model = torchvision.models.resnet18(num_classes=10).to(device)
    
        # Optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1 * hvd.size(),  # Scale learning rate by number of workers
            momentum=0.9,
            weight_decay=5e-4
        )
    
        # Wrap optimizer with Horovod
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=hvd.Compression.fp16,  # Reduce communication with FP16 compression
            op=hvd.Average  # Average gradients
        )
    
        # Broadcast initial parameters (same initial values across all workers)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
        # Loss function
        criterion = nn.CrossEntropyLoss()
    
        # Training loop
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
    
                # Horovod automatically performs AllReduce on gradients
                optimizer.step()
    
                # Statistics
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    
            # Aggregate statistics across all workers
            epoch_loss = metric_average(epoch_loss, 'avg_loss')
            accuracy = metric_average(correct / total, 'avg_accuracy')
    
            # Only Rank 0 outputs logs
            if hvd.rank() == 0:
                print(f"Epoch {epoch}/{num_epochs} - "
                      f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    
                # Save model
                if (epoch + 1) % 10 == 0:
                    torch.save(model.state_dict(),
                              f'horovod_checkpoint_epoch_{epoch+1}.pth')
    
    def metric_average(val, name):
        """
        Average metrics across all workers
        """
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()
    
    if __name__ == "__main__":
        train_horovod()
    

**Execution method:**
    
    
    # Single node, 4 GPUs
    horovodrun -np 4 python horovod_training.py
    
    # Multiple nodes (4 GPUs per node, 2 nodes)
    horovodrun -np 8 -H node1:4,node2:4 python horovod_training.py
    
    # Slurm cluster
    srun --ntasks=8 --gres=gpu:4 python horovod_training.py
    

### 4.3.3 TensorFlow/PyTorch Integration

#### Code Example 5: TensorFlow Training with Horovod

horovod_tensorflow.py - Using Horovod with TensorFlow
    
    
    import tensorflow as tf
    import horovod.tensorflow as hvd
    
    def train_tensorflow_horovod():
        """
        Distributed training with Horovod + TensorFlow
        """
        # Initialize Horovod
        hvd.init()
    
        # Enable GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()], 'GPU'
            )
    
        # Dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
        # Dataset sharding (different data for each worker)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
        dataset = dataset.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    
        # Model
        model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            classes=10,
            input_shape=(32, 32, 3)
        )
    
        # Optimizer
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.1 * hvd.size(),
            momentum=0.9
        )
    
        # Wrap optimizer with Horovod
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            compression=hvd.Compression.fp16
        )
    
        # Loss function and metrics
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
        @tf.function
        def training_step(images, labels, first_batch):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_fn(labels, predictions)
    
            # Horovod performs AllReduce on gradients
            tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
            # Broadcast parameters on first batch
            if first_batch:
                hvd.broadcast_variables(model.variables, root_rank=0)
                hvd.broadcast_variables(optimizer.variables(), root_rank=0)
    
            return loss
    
        # Training loop
        for epoch in range(100):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(dataset):
                loss = training_step(images, labels, batch_idx == 0 and epoch == 0)
                epoch_loss += loss.numpy()
    
            # Calculate average loss
            epoch_loss = hvd.allreduce(
                tf.constant(epoch_loss / len(dataset)),
                average=True
            ).numpy()
    
            if hvd.rank() == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
                # Save model
                if (epoch + 1) % 10 == 0:
                    model.save(f'tf_horovod_model_epoch_{epoch+1}.h5')
    
    if __name__ == "__main__":
        train_tensorflow_horovod()
    

### 4.3.4 Performance Comparison: PyTorch DDP vs Horovod

Item | PyTorch DDP | Horovod  
---|---|---  
**Communication Backend** | NCCL, Gloo, MPI | MPI, NCCL  
**Framework Support** | PyTorch only | TensorFlow, PyTorch, Keras, MXNet  
**Implementation Complexity** | Moderate | Simple  
**Communication Efficiency** | High (NCCL optimized) | High (Ring-AllReduce)  
**Scalability** | Hundreds of GPUs | Thousands of GPUs (MPI-based)  
**Gradient Compression** | Manual implementation | Standard support (FP16)  
**Dynamic Graph Support** | Full support | Full support  
**Ecosystem** | PyTorch official | Independent project  
  
**Benchmark results (ResNet-50, ImageNet, 8 GPUs):**

  * **PyTorch DDP** : 2,400 images/sec (92% scaling efficiency)
  * **Horovod** : 2,350 images/sec (90% scaling efficiency)

**Recommendations:**

  * **PyTorch only** → PyTorch DDP
  * **Multiple frameworks** → Horovod
  * **Large clusters (100+ GPUs)** → Horovod (MPI stability)

\--- 

## 4.4 Large-Scale Model Training Techniques

### 4.4.1 Gradient Accumulation

**Purpose:**

  * Achieve large batch sizes under GPU memory constraints
  * Execute small batches multiple times and accumulate gradients

**Mathematical formula:**

$$ \nabla_\theta L_{\text{effective}} = \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta L(\text{mini-batch}_k) $$ 

$K$: Number of accumulation steps, effective batch size = $K \times$ mini-batch size

#### Code Example 6: Gradient Accumulation Implementation

gradient_accumulation.py - Gradient accumulation
    
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    
    def train_with_gradient_accumulation(
        model, dataloader, optimizer, criterion,
        accumulation_steps=4, device='cuda'
    ):
        """
        Training with gradient accumulation
    
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        model.train()
        optimizer.zero_grad()
    
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
    
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
    
            # Divide loss by accumulation steps
            loss = loss / accumulation_steps
    
            # Backward pass (accumulate gradients)
            loss.backward()
    
            # Update parameters every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    
                print(f"Batch {batch_idx+1}, Updated parameters")
    
        # Process remaining batches
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Usage example
    model = torchvision.models.resnet50().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Small batch size (16) × accumulation steps (4) = effective batch size (64)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    train_with_gradient_accumulation(
        model, train_loader, optimizer, criterion,
        accumulation_steps=4
    )
    

**Advantages:**

  * Avoid GPU memory constraints
  * Enable large batch size training

**Disadvantages:**

  * Reduced training speed (small effect on communication overhead reduction)
  * Be careful with Batch Normalization behavior (statistics on small batches)

### 4.4.2 Mixed Precision Training (AMP)

**Overview:**

  * Mix FP16 (half-precision floating point) and FP32 for training
  * Accelerate computation and reduce memory
  * Ensure numerical stability with loss scaling

**Effects:**

  * **Speed up** : 1.5~2x (utilizing Tensor Cores)
  * **Memory reduction** : About 50%

#### Code Example 7: Using PyTorch AMP

amp_training.py - Automatic Mixed Precision
    
    
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    import torchvision
    
    def train_with_amp(model, dataloader, optimizer, criterion, device='cuda'):
        """
        Training with Automatic Mixed Precision (AMP)
        """
        model.train()
    
        # GradScaler: Properly scale FP16 gradients
        scaler = GradScaler()
    
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
    
            optimizer.zero_grad()
    
            # autocast: Automatically select optimal precision
            with autocast():
                output = model(data)
                loss = criterion(output, target)
    
            # Backward pass with scaled loss
            scaler.scale(loss).backward()
    
            # Unscale gradients and update parameters
            scaler.step(optimizer)
            scaler.update()
    
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Usage example
    model = torchvision.models.resnet50().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
    
    train_with_amp(model, train_loader, optimizer, criterion)
    

**Combining AMP + Gradient Accumulation + DDP** :

amp_grad_accum_ddp.py - Complete optimization
    
    
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
        Complete implementation of AMP + Gradient Accumulation + DDP
        """
        # Setup distributed environment
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
        # Wrap model with DDP
        model = model.cuda(rank)
        model = DDP(model, device_ids=[rank])
    
        # Dataloader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=4
        )
    
        # Optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
    
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad()
    
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.cuda(rank), target.cuda(rank)
    
                # Forward pass with AMP
                with autocast():
                    output = model(data)
                    loss = criterion(output, target) / accumulation_steps
    
                # Gradient accumulation
                scaler.scale(loss).backward()
    
                # Update parameters every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
    
            # Process remaining batches
            if (batch_idx + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
            if rank == 0:
                print(f"Epoch {epoch} completed")
    
        dist.destroy_process_group()
    
    # Execute
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
    

### 4.4.3 Gradient Checkpointing

**Overview:**

  * Don't save intermediate activations during forward pass, recompute during backward pass
  * Significantly reduce memory usage (trade-off with computation time)

**Effects:**

  * **Memory reduction** : O(n) → O(√n) (n is number of layers)
  * **Computation increase** : About 20-30%

    
    
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
            # Use checkpoint for each layer
            x = checkpoint(self.layer1, x)
            x = checkpoint(self.layer2, x)
            x = checkpoint(self.layer3, x)
            x = checkpoint(self.layer4, x)
            x = self.fc(x)
            return x
    

### 4.4.4 DeepSpeed ZeRO

**ZeRO (Zero Redundancy Optimizer):**

  * Microsoft's ultra-large-scale model training framework
  * Distribute optimizer states, gradients, and parameters

**Three stages of ZeRO:**

  * **ZeRO-1** : Optimizer state partitioning (4x memory reduction)
  * **ZeRO-2** : + Gradient partitioning (8x memory reduction)
  * **ZeRO-3** : + Parameter partitioning (Nx reduction with N workers)

#### Code Example 8: Using DeepSpeed ZeRO

deepspeed_zero.py - DeepSpeed training
    
    
    import torch
    import torch.nn as nn
    import deepspeed
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    def train_with_deepspeed():
        """
        Large-scale model training using DeepSpeed ZeRO
        """
        # DeepSpeed configuration file
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
                "stage": 3,  # ZeRO-3: Maximum memory reduction
                "offload_optimizer": {
                    "device": "cpu",  # Offload optimizer states to CPU
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",  # Offload parameters to CPU
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
    
        # Large-scale model (GPT-2 Large: 774M parameters)
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    
        # Initialize DeepSpeed engine
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
    
        # Dataloader
        train_loader = ...  # Prepare dataloader
    
        # Training loop
        for epoch in range(10):
            for batch in train_loader:
                inputs = batch['input_ids'].to(model_engine.local_rank)
                labels = batch['labels'].to(model_engine.local_rank)
    
                # Forward pass
                outputs = model_engine(inputs, labels=labels)
                loss = outputs.loss
    
                # DeepSpeed automatically handles backward pass and parameter updates
                model_engine.backward(loss)
                model_engine.step()
    
        # Save model (aggregate all parameters)
        model_engine.save_checkpoint('./checkpoints')
    
    if __name__ == "__main__":
        # Launch DeepSpeed
        # deepspeed --num_gpus=8 deepspeed_zero.py
        train_with_deepspeed()
    

**Execution command:**
    
    
    # Single node, 8 GPUs
    deepspeed --num_gpus=8 deepspeed_zero.py
    
    # Multiple nodes (2 nodes, 8 GPUs each)
    deepspeed --num_nodes=2 --num_gpus=8 --hostfile=hostfile deepspeed_zero.py
    

**ZeRO-3 effect (GPT-3 175B parameters):**

  * **Conventional DDP** : Training impossible (GPU memory insufficient)
  * **ZeRO-3** : Trainable with 128 GPUs (40GB each)
  * **Memory reduction** : 64x (128 workers)

\--- 

## 4.5 Distributed Learning Best Practices

### 4.5.1 Communication Optimization

**Gradient Bucketing:**

  * Communicate small gradients together (reduce latency)
  * PyTorch DDP enabled by default

    
    
    # DDP bucket size setting
    model = DDP(
        model,
        device_ids=[rank],
        bucket_cap_mb=25,  # Bucket size (MB)
        find_unused_parameters=False  # Disable unused parameter detection
    )
    

**Gradient compression:**

  * FP16 compression reduces communication volume by 50%
  * Sparsification, quantization

    
    
    # Horovod fp16 compression
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        compression=hvd.Compression.fp16
    )
    

**Hierarchical AllReduce:**

  * Intra-node NCCL → Inter-node MPI
  * Horovod Hierarchical AllReduce

### 4.5.2 Batch Size and Learning Rate Scaling

**Linear Scaling Rule:**

$$ \text{LR}_{\text{distributed}} = \text{LR}_{\text{base}} \times \frac{\text{Batch}_{\text{distributed}}}{\text{Batch}_{\text{base}}} $$ 

**Example:**

  * Base: LR=0.1, Batch=256 (single GPU)
  * 8 GPUs: LR=0.8, Batch=2,048 (256×8)

**Warmup:**

  * Gradually increase learning rate in first epochs
  * Stabilize training with large batch sizes

    
    
    def linear_warmup_cosine_decay(step, warmup_steps, total_steps, base_lr, max_lr):
        """
        Warmup + Cosine Decay learning rate scheduler
        """
        if step < warmup_steps:
            # Warmup: Linear increase
            lr = base_lr + (max_lr - base_lr) * step / warmup_steps
        else:
            # Cosine Decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = base_lr + 0.5 * (max_lr - base_lr) * (1 + math.cos(math.pi * progress))
        return lr
    

### 4.5.3 Learning Rate Adjustment Best Practices

**LARS (Layer-wise Adaptive Rate Scaling):**

  * Adaptively adjust learning rate per layer
  * Effective for ultra-large batch sizes (32K~64K)

**LAMB (Layer-wise Adaptive Moments optimizer for Batch training):**

  * Achieved batch size 65,536 for BERT training
  * Adam-based LARS extension

### 4.5.4 Debugging Distributed Code

**Common errors:**

**1\. Deadlock:**

  * All processes must execute the same collective communication
  * Hangs if only some processes communicate

    
    
    # Bad example: Only Rank 0 performs allreduce
    if rank == 0:
        dist.all_reduce(tensor)  # Other processes wait → Deadlock
    
    # Correct example: All processes perform allreduce
    dist.all_reduce(tensor)
    if rank == 0:
        print(tensor)
    

**2\. Memory leak:**

  * Forgetting to detach during gradient accumulation
  * Computation graph continues to be retained

    
    
    # Correct implementation for gradient accumulation
    loss = loss / accumulation_steps
    loss.backward()
    
    # Detach when computing metrics
    total_loss += loss.detach().item()
    

**3\. Lack of reproducibility:**

  * Insufficient seed setting
  * Different initialization in each process

    
    
    def set_seed(seed, rank):
        """
        Seed setting for reproducibility
        """
        torch.manual_seed(seed + rank)  # Different seed per rank
        torch.cuda.manual_seed_all(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
    
        # CuDNN deterministic behavior (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

**Debugging tools:**

**NCCL environment variables:**
    
    
    export NCCL_DEBUG=INFO           # NCCL debug information
    export NCCL_DEBUG_SUBSYS=ALL     # Log all subsystems
    export NCCL_IB_DISABLE=1         # Disable InfiniBand (for debugging)
    export NCCL_P2P_DISABLE=1        # Disable P2P communication
    

**PyTorch profiler:**
    
    
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

## 4.6 Summary

### What We Learned

  1. **Distributed learning strategies:**

     * Data Parallelism: Model copying, data splitting, gradient aggregation
     * Model Parallelism: Model splitting, resolving memory constraints
     * Pipeline Parallelism: Stage splitting, improving GPU utilization
     * Hybrid Approaches: 3D Parallelism, ZeRO
  2. **PyTorch DDP:**

     * torch.distributed basics (Rank, World Size, Backend)
     * Data splitting with DistributedSampler
     * Automatic gradient synchronization with DDP wrapper
     * Multi-node training (Slurm, SSH)
  3. **Horovod:**

     * Ring-AllReduce architecture
     * Efficient communication based on MPI/NCCL
     * TensorFlow/PyTorch/Keras support
     * Further speedup with FP16 compression
  4. **Large-scale model training:**

     * Gradient Accumulation: Avoiding memory constraints
     * Mixed Precision (AMP): 1.5~2x speedup, 50% memory reduction
     * Gradient Checkpointing: O(n) → O(√n) memory reduction
     * DeepSpeed ZeRO: Ultra-large-scale model (175B parameters) training
  5. **Best practices:**

     * Communication optimization: Bucketing, compression, hierarchical AllReduce
     * Batch size scaling: Linear Scaling Rule, Warmup
     * Learning rate adjustment: LARS, LAMB
     * Debugging: Avoiding deadlocks, memory leak countermeasures, reproducibility

### Next Steps

In Chapter 5, we will learn about real-world large-scale data processing applications:

  * Distributed training of recommendation systems (Netflix, Amazon scale)
  * Pre-training of large language models (BERT, GPT)
  * Real-time stream processing (Apache Kafka + Spark Streaming)
  * Large-scale screening in materials informatics

\--- 

## Exercises

**Question 1:** Explain the differences between Data Parallelism and Model Parallelism from the perspectives of memory usage and communication patterns.

**Question 2:** For training with 8 GPUs, if a single GPU uses batch size 32 and learning rate 0.1, calculate the appropriate batch size and learning rate for distributed training.

**Question 3:** You want to achieve an effective batch size of 256 using Gradient Accumulation. If GPU memory constraints only allow batch size 32, what should accumulation_steps be set to?

**Question 4:** Explain the mechanism of "loss scaling" used by Mixed Precision Training (AMP) to maintain numerical stability.

**Question 5:** Discuss why DeepSpeed ZeRO-3 has higher memory efficiency compared to conventional Data Parallelism from the perspective of distributing optimizer states, gradients, and parameters (within 500 characters).

\--- 

## References

  1. Goyal, P. et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." _arXiv:1706.02677_ (2017).
  2. Sergeev, A. & Del Balso, M. "Horovod: fast and easy distributed deep learning in TensorFlow." _arXiv:1802.05799_ (2018).
  3. Li, S. et al. "PyTorch Distributed: Experiences on Accelerating Data Parallel Training." _VLDB_ (2020).
  4. Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." _SC'20_ (2020).
  5. Huang, Y. et al. "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." _NeurIPS_ (2019).
  6. You, Y. et al. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." _ICLR_ (2020).
  7. Micikevicius, P. et al. "Mixed Precision Training." _ICLR_ (2018).
  8. Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." _arXiv:1909.08053_ (2019).

\--- 

**Next chapter** : [Chapter 5: Real-World Large-Scale Data Processing Applications](<chapter5-large-scale-ml-pipeline.html>)

**License** : This content is provided under CC BY 4.0 license.
