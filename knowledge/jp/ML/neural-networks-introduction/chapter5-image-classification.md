---
title: 第5章：画像分類プロジェクト
chapter_title: 第5章：画像分類プロジェクト
subtitle: 実践的なCNNプロジェクト - MNIST から CIFAR-10 まで
reading_time: 30-35分
difficulty: 中級〜上級
code_examples: 17
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ MNIST手書き数字認識の完全な実装パイプライン
  * ✅ CNN設計の実践的なアプローチとベストプラクティス
  * ✅ CIFAR-10カラー画像分類の高度なテクニック
  * ✅ データ拡張と正則化の効果的な活用
  * ✅ 転移学習とモデルアンサンブルの実装
  * ✅ 本番環境へのデプロイメント戦略

* * *

## 5.1 プロジェクト1: MNIST手書き数字認識

### データセットの準備と前処理

**MNIST（Modified National Institute of Standards and Technology）** は、手書き数字（0-9）の画像分類タスクで、機械学習の"Hello World"とも呼ばれます。

特徴 | 詳細  
---|---  
**画像サイズ** | 28×28 ピクセル（グレースケール）  
**クラス数** | 10クラス（数字 0-9）  
**訓練データ** | 60,000枚  
**テストデータ** | 10,000枚  
**難易度** | 入門レベル（最高精度: 99.8%+）  
  
#### データのロードと可視化
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # データ変換の定義
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] → [0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST平均と標準偏差
    ])
    
    # データセットのロード
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f'訓練データ: {len(train_dataset)}枚')
    print(f'テストデータ: {len(test_dataset)}枚')
    
    # サンプル画像の可視化
    def visualize_mnist_samples(dataset, n_samples=10):
        """MNISTサンプル画像の表示"""
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.ravel()
    
        for i in range(n_samples):
            image, label = dataset[i]
            # 正規化を戻す
            image = image.squeeze() * 0.3081 + 0.1307
    
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    visualize_mnist_samples(train_dataset)
    

### CNNモデルの設計

MNIST用の効率的なCNNアーキテクチャを設計します。
    
    
    ```mermaid
    graph LR
        A[入力28×28×1] --> B[Conv124×24×32]
        B --> C[Pool112×12×32]
        C --> D[Conv28×8×64]
        D --> E[Pool24×4×64]
        E --> F[Flatten1024]
        F --> G[FC1128]
        G --> H[Dropout0.5]
        H --> I[FC210]
    
        style A fill:#e3f2fd
        style I fill:#e8f5e9
    ```
    
    
    class MNISTNet(nn.Module):
        """MNIST用CNNモデル"""
    
        def __init__(self):
            super(MNISTNet, self).__init__()
    
            # 畳み込み層
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # 28×28×1 → 24×24×32
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # 12×12×32 → 8×8×64
    
            # プーリング層
            self.pool = nn.MaxPool2d(2, 2)
    
            # 全結合層
            self.fc1 = nn.Linear(64 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, 10)
    
            # ドロップアウト
            self.dropout = nn.Dropout(0.5)
    
        def forward(self, x):
            # 第1ブロック: Conv → ReLU → Pool
            x = self.pool(F.relu(self.conv1(x)))  # → 12×12×32
    
            # 第2ブロック: Conv → ReLU → Pool
            x = self.pool(F.relu(self.conv2(x)))  # → 4×4×64
    
            # 平坦化
            x = x.view(-1, 64 * 4 * 4)  # → 1024
    
            # 全結合層
            x = F.relu(self.fc1(x))  # → 128
            x = self.dropout(x)
            x = self.fc2(x)  # → 10
    
            return x
    
    # モデルのインスタンス化
    model = MNISTNet().to(device)
    
    # モデル構造の表示
    print(model)
    
    # パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n総パラメータ数: {total_params:,}')
    print(f'訓練可能パラメータ数: {trainable_params:,}')
    

**出力例** ：
    
    
    MNISTNet(
      (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=1024, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
      (dropout): Dropout(p=0.5, inplace=False)
    )
    
    総パラメータ数: 163,978
    訓練可能パラメータ数: 163,978
    

### 学習と評価
    
    
    def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
        """1エポックの訓練"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
    
            # 勾配をゼロクリア
            optimizer.zero_grad()
    
            # 順伝播
            output = model(data)
            loss = criterion(output, target)
    
            # 逆伝播
            loss.backward()
            optimizer.step()
    
            # 統計情報
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
            # 進捗表示
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
    
        return epoch_loss, epoch_acc
    
    def evaluate(model, device, test_loader, criterion):
        """テストデータで評価"""
        model.eval()
        test_loss = 0
        correct = 0
    
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader)
        test_acc = 100. * correct / len(test_loader.dataset)
    
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n')
    
        return test_loss, test_acc
    
    # 訓練設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練ループ
    num_epochs = 10
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    # モデルの保存
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print('モデルを保存しました: mnist_cnn.pth')
    

### エラー分析と可視化
    
    
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    def plot_confusion_matrix(model, device, test_loader):
        """混同行列の可視化"""
        model.eval()
        all_preds = []
        all_targets = []
    
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
        # 混同行列の計算
        cm = confusion_matrix(all_targets, all_preds)
    
        # 可視化
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - MNIST')
        plt.show()
    
        # 分類レポート
        print('\nClassification Report:')
        print(classification_report(all_targets, all_preds,
                                    target_names=[str(i) for i in range(10)]))
    
    plot_confusion_matrix(model, device, test_loader)
    
    def visualize_misclassified(model, device, test_loader, n_samples=10):
        """誤分類されたサンプルの表示"""
        model.eval()
        misclassified = []
    
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
    
                # 誤分類を見つける
                mask = pred != target
                if mask.sum() > 0:
                    for i in range(len(mask)):
                        if mask[i]:
                            misclassified.append({
                                'image': data[i].cpu(),
                                'true': target[i].item(),
                                'pred': pred[i].item(),
                                'confidence': F.softmax(output[i], dim=0)[pred[i]].item()
                            })
    
                            if len(misclassified) >= n_samples:
                                break
    
                if len(misclassified) >= n_samples:
                    break
    
        # 可視化
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.ravel()
    
        for i, item in enumerate(misclassified[:n_samples]):
            image = item['image'].squeeze() * 0.3081 + 0.1307
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"True: {item['true']}, Pred: {item['pred']}\n"
                             f"Conf: {item['confidence']:.2%}")
            axes[i].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    visualize_misclassified(model, device, test_loader)
    
    def plot_training_history(train_losses, train_accs, test_losses, test_accs):
        """学習曲線の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        epochs = range(1, len(train_losses) + 1)
    
        # Loss曲線
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # Accuracy曲線
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    

* * *

## 5.2 プロジェクト2: CIFAR-10カラー画像分類

### データセットの概要

**CIFAR-10（Canadian Institute for Advanced Research）** は、MNISTより難易度の高い実世界画像の分類タスクです。

特徴 | 詳細  
---|---  
**画像サイズ** | 32×32 ピクセル（RGB カラー）  
**クラス数** | 10クラス（飛行機、車、鳥、猫、鹿、犬、蛙、馬、船、トラック）  
**訓練データ** | 50,000枚（各クラス5,000枚）  
**テストデータ** | 10,000枚（各クラス1,000枚）  
**難易度** | 中級（最高精度: 99%+、標準的には 90-95%）  
  
### データ拡張の重要性

CIFAR-10では**データ拡張（Data Augmentation）** が精度向上の鍵となります。
    
    
    ```mermaid
    graph LR
        A[元画像] --> B[水平反転]
        A --> C[ランダムクロップ]
        A --> D[色調変換]
        A --> E[回転]
        B --> F[訓練データ]
        C --> F
        D --> F
        E --> F
    
        style A fill:#e3f2fd
        style F fill:#e8f5e9
    ```
    
    
    from torchvision import datasets, transforms
    
    # CIFAR-10のクラス名
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # データ拡張を含む訓練用変換
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # ランダムクロップ
        transforms.RandomHorizontalFlip(),  # 水平反転（確率50%）
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 色調変換
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # テスト用変換（拡張なし）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # データセットのロード
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    def visualize_augmentation(dataset, idx=0, n_augments=8):
        """データ拡張の効果を可視化"""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
    
        # 元画像
        original_img, label = dataset[idx]
    
        for i in range(n_augments):
            # 拡張後の画像を取得
            img, _ = dataset[idx]
    
            # 正規化を戻す
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            img = std * img + mean
            img = np.clip(img, 0, 1)
    
            axes[i].imshow(img)
            axes[i].set_title(f'Augmentation {i+1}')
            axes[i].axis('off')
    
        plt.suptitle(f'Class: {classes[label]}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    visualize_augmentation(train_dataset)
    

### より深いネットワーク設計

CIFAR-10にはVGGスタイルのより深いCNNが効果的です。
    
    
    class CIFAR10Net(nn.Module):
        """CIFAR-10用の深いCNN (VGG-style)"""
    
        def __init__(self, num_classes=10):
            super(CIFAR10Net, self).__init__()
    
            # ブロック1: 32×32 → 16×16
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1_1 = nn.BatchNorm2d(64)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn1_2 = nn.BatchNorm2d(64)
    
            # ブロック2: 16×16 → 8×8
            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2_1 = nn.BatchNorm2d(128)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn2_2 = nn.BatchNorm2d(128)
    
            # ブロック3: 8×8 → 4×4
            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3_1 = nn.BatchNorm2d(256)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn3_2 = nn.BatchNorm2d(256)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn3_3 = nn.BatchNorm2d(256)
    
            # プーリング
            self.pool = nn.MaxPool2d(2, 2)
    
            # 全結合層
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.bn_fc1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, num_classes)
    
            # ドロップアウト
            self.dropout = nn.Dropout(0.5)
    
        def forward(self, x):
            # ブロック1
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))
            x = self.pool(x)
    
            # ブロック2
            x = F.relu(self.bn2_1(self.conv2_1(x)))
            x = F.relu(self.bn2_2(self.conv2_2(x)))
            x = self.pool(x)
    
            # ブロック3
            x = F.relu(self.bn3_1(self.conv3_1(x)))
            x = F.relu(self.bn3_2(self.conv3_2(x)))
            x = F.relu(self.bn3_3(self.conv3_3(x)))
            x = self.pool(x)
    
            # 平坦化
            x = x.view(-1, 256 * 4 * 4)
    
            # 全結合層
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)
    
            return x
    
    # モデルのインスタンス化
    model = CIFAR10Net().to(device)
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'総パラメータ数: {total_params:,}')
    

### 正則化テクニック（Dropout、Batch Normalization）

> **Batch Normalization** : 各層の出力を正規化し、学習を安定化・高速化します。
> 
> **Dropout** : 訓練中にランダムにニューロンを無効化し、過学習を防ぎます。
    
    
    def train_with_scheduler(model, device, train_loader, test_loader, num_epochs=50):
        """Learning Rate Schedulerを使った訓練"""
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []
    
        best_acc = 0.0
    
        for epoch in range(1, num_epochs + 1):
            # 訓練
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
    
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
    
            # 評価
            model.eval()
            test_loss = 0
            correct = 0
    
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
    
            test_loss /= len(test_loader)
            test_acc = 100. * correct / len(test_loader.dataset)
    
            # スケジューラのステップ
            scheduler.step()
    
            # 記録
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
    
            print(f'Epoch {epoch}/{num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% - '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
            # ベストモデルの保存
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'cifar10_best.pth')
                print(f'  → Best model saved! (Acc: {best_acc:.2f}%)')
    
        return train_losses, train_accs, test_losses, test_accs
    
    # 訓練実行
    train_losses, train_accs, test_losses, test_accs = train_with_scheduler(
        model, device, train_loader, test_loader, num_epochs=50
    )
    

### Early Stopping実装
    
    
    class EarlyStopping:
        """Early Stoppingの実装"""
    
        def __init__(self, patience=7, min_delta=0, path='checkpoint.pth'):
            """
            Args:
                patience: 改善が見られない最大エポック数
                min_delta: 改善と見なす最小変化量
                path: モデル保存先のパス
            """
            self.patience = patience
            self.min_delta = min_delta
            self.path = path
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.best_loss = np.Inf
    
        def __call__(self, val_loss, model):
            score = -val_loss
    
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.min_delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
    
        def save_checkpoint(self, val_loss, model):
            """モデルを保存"""
            torch.save(model.state_dict(), self.path)
            self.best_loss = val_loss
    
    # 使用例
    def train_with_early_stopping(model, device, train_loader, test_loader,
                                  num_epochs=100, patience=10):
        """Early Stoppingを使った訓練"""
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=patience, path='cifar10_checkpoint.pth')
    
        for epoch in range(1, num_epochs + 1):
            # 訓練ループ（省略）
            train_loss = 0.0  # 実際には訓練を実行
    
            # 検証
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
    
            val_loss /= len(test_loader)
    
            # Early Stoppingチェック
            early_stopping(val_loss, model)
    
            if early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch}')
                break
    
        # ベストモデルのロード
        model.load_state_dict(torch.load('cifar10_checkpoint.pth'))
        return model
    

* * *

## 5.3 高度なテクニック

### Transfer Learning（転移学習）

事前学習済みモデルを使うことで、少ないデータでも高精度を実現できます。
    
    
    ```mermaid
    graph LR
        A[ImageNet事前学習] --> B[特徴抽出器固定]
        B --> C[分類器訓練]
        C --> D[CIFAR-10分類]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```
    
    
    import torchvision.models as models
    
    def create_transfer_model(num_classes=10, freeze_features=True):
        """ResNet18をベースにした転移学習モデル"""
    
        # ImageNetで事前学習済みのResNet18をロード
        model = models.resnet18(pretrained=True)
    
        # 特徴抽出器を固定
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
    
        # 最終層を置き換え（CIFAR-10用に10クラス分類）
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
        return model
    
    # モデルの作成
    transfer_model = create_transfer_model(num_classes=10, freeze_features=True).to(device)
    
    # 訓練可能なパラメータのみを最適化
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()),
                           lr=0.001)
    
    print("転移学習モデル:")
    print(f"全パラメータ数: {sum(p.numel() for p in transfer_model.parameters()):,}")
    print(f"訓練可能パラメータ数: {sum(p.numel() for p in transfer_model.parameters() if p.requires_grad):,}")
    
    def finetune_transfer_model(model, device, train_loader, test_loader, num_epochs=20):
        """転移学習モデルのファインチューニング"""
    
        criterion = nn.CrossEntropyLoss()
    
        # Phase 1: 分類器のみ訓練（5エポック）
        print("\nPhase 1: Training classifier only...")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
        for epoch in range(1, 6):
            model.train()
            running_loss = 0.0
    
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    
            print(f"Epoch {epoch}/5 - Loss: {running_loss/len(train_loader):.4f}")
    
        # Phase 2: 全層をファインチューニング（15エポック）
        print("\nPhase 2: Fine-tuning all layers...")
        for param in model.parameters():
            param.requires_grad = True
    
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # より小さい学習率
    
        for epoch in range(1, 16):
            model.train()
            running_loss = 0.0
    
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    
            # 評価
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
    
            test_acc = 100. * correct / len(test_loader.dataset)
            print(f"Epoch {epoch}/15 - Loss: {running_loss/len(train_loader):.4f}, "
                  f"Test Acc: {test_acc:.2f}%")
    
        return model
    
    # ファインチューニング実行
    transfer_model = finetune_transfer_model(transfer_model, device, train_loader, test_loader)
    

### Learning Rate Finder

最適な学習率を自動的に見つけるテクニック（Fast.aiで有名）。
    
    
    import matplotlib.pyplot as plt
    
    class LRFinder:
        """Learning Rate Finderの実装"""
    
        def __init__(self, model, optimizer, criterion, device):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.history = {'lr': [], 'loss': []}
            self.best_loss = 1e9
    
        def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
            """学習率の範囲をテスト"""
    
            # 学習率のスケジュール
            lr_schedule = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
    
            # モデルの初期状態を保存
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()
    
            self.model.train()
            iter_count = 0
    
            for batch_idx, (data, target) in enumerate(train_loader):
                if iter_count >= num_iter:
                    break
    
                data, target = data.to(self.device), target.to(self.device)
    
                # 学習率を更新
                lr = lr_schedule[iter_count]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
    
                # 順伝播と逆伝播
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
    
                # 記録
                self.history['lr'].append(lr)
                self.history['loss'].append(loss.item())
    
                # 発散チェック
                if loss.item() > 4 * self.best_loss:
                    break
    
                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
    
                iter_count += 1
    
            # モデルを初期状態に戻す
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
    
        def plot(self):
            """結果のプロット"""
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['lr'], self.history['loss'])
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.grid(True, alpha=0.3)
            plt.show()
    
            # 推奨学習率
            min_loss_idx = np.argmin(self.history['loss'])
            suggested_lr = self.history['lr'][min_loss_idx] / 10
            print(f"推奨学習率: {suggested_lr:.2e}")
    
    # 使用例
    model = CIFAR10Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1, num_iter=100)
    lr_finder.plot()
    

### モデルアンサンブル

複数のモデルの予測を組み合わせることで、精度を向上させます。
    
    
    class ModelEnsemble:
        """モデルアンサンブルの実装"""
    
        def __init__(self, models, device):
            """
            Args:
                models: モデルのリスト
                device: 実行デバイス
            """
            self.models = models
            self.device = device
    
            # 全モデルを評価モードに
            for model in self.models:
                model.eval()
    
        def predict(self, data):
            """アンサンブル予測（平均）"""
            predictions = []
    
            with torch.no_grad():
                for model in self.models:
                    output = model(data)
                    predictions.append(F.softmax(output, dim=1))
    
            # 平均を取る
            ensemble_pred = torch.stack(predictions).mean(dim=0)
            return ensemble_pred
    
        def evaluate(self, test_loader):
            """テストデータで評価"""
            correct = 0
            total = 0
    
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
    
                    # アンサンブル予測
                    ensemble_pred = self.predict(data)
                    pred = ensemble_pred.argmax(dim=1)
    
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
    
            accuracy = 100. * correct / total
            print(f'Ensemble Accuracy: {correct}/{total} ({accuracy:.2f}%)')
            return accuracy
    
    # 複数のモデルを訓練（異なる初期化や設定で）
    def train_multiple_models(n_models=5):
        """複数のモデルを訓練"""
        models = []
    
        for i in range(n_models):
            print(f"\nTraining model {i+1}/{n_models}...")
    
            # モデルの初期化
            model = CIFAR10Net().to(device)
    
            # 訓練（簡略版）
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
    
            # 訓練ループ（省略）
            # ... train_epoch(model, device, train_loader, optimizer, criterion)
    
            models.append(model)
    
        return models
    
    # アンサンブルの使用例
    # models = train_multiple_models(n_models=5)
    # ensemble = ModelEnsemble(models, device)
    # ensemble_acc = ensemble.evaluate(test_loader)
    

### Grad-CAM可視化

モデルが画像のどこに注目しているかを可視化します。
    
    
    class GradCAM:
        """Gradient-weighted Class Activation Mappingの実装"""
    
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
    
            # フックを登録
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_backward_hook(self.save_gradient)
    
        def save_activation(self, module, input, output):
            """順伝播時の活性化を保存"""
            self.activations = output.detach()
    
        def save_gradient(self, module, grad_input, grad_output):
            """逆伝播時の勾配を保存"""
            self.gradients = grad_output[0].detach()
    
        def generate_cam(self, input_image, target_class):
            """CAMを生成"""
    
            # 順伝播
            output = self.model(input_image)
    
            # ターゲットクラスのスコアに対する勾配
            self.model.zero_grad()
            class_loss = output[0, target_class]
            class_loss.backward()
    
            # 勾配と活性化を取得
            gradients = self.gradients[0]  # [C, H, W]
            activations = self.activations[0]  # [C, H, W]
    
            # 勾配の平均（重み）
            weights = gradients.mean(dim=(1, 2))  # [C]
    
            # 重み付き和
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
    
            # ReLU適用
            cam = F.relu(cam)
    
            # 正規化
            cam = cam - cam.min()
            cam = cam / cam.max()
    
            return cam.cpu().numpy()
    
    def visualize_gradcam(model, image, label, device):
        """Grad-CAMの可視化"""
    
        # Grad-CAMの準備（最後の畳み込み層をターゲット）
        target_layer = model.conv3_3  # CIFAR10Netの場合
        gradcam = GradCAM(model, target_layer)
    
        # CAMの生成
        model.eval()
        image_input = image.unsqueeze(0).to(device)
        cam = gradcam.generate_cam(image_input, label)
    
        # 元画像の準備
        img = image.cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)
    
        # CAMをリサイズ
        from scipy.ndimage import zoom
        cam_resized = zoom(cam, (img.shape[0]/cam.shape[0], img.shape[1]/cam.shape[1]))
    
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
    
        axes[2].imshow(img)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    # 使用例
    # image, label = test_dataset[0]
    # visualize_gradcam(model, image, label, device)
    

* * *

## 5.4 実践的なヒントとベストプラクティス

### ハイパーパラメータチューニング
    
    
    ```mermaid
    graph TD
        A[初期設定] --> B{検証精度}
        B -->|低い| C[学習率調整]
        B -->|過学習| D[正則化強化]
        B -->|学習不足| E[エポック数増加]
        C --> F[再訓練]
        D --> F
        E --> F
        F --> B
        B -->|満足| G[最終評価]
    
        style A fill:#e3f2fd
        style G fill:#e8f5e9
    ```

ハイパーパラメータ | 推奨範囲 | 調整のヒント  
---|---|---  
**Learning Rate** | 1e-4 ~ 1e-1 | LR Finderを使用、大きすぎると発散  
**Batch Size** | 32 ~ 256 | GPU メモリに依存、大きいほど安定  
**Weight Decay** | 1e-5 ~ 1e-3 | 過学習対策、L2正則化  
**Dropout Rate** | 0.3 ~ 0.5 | 過学習が強い場合は増やす  
**Optimizer** | Adam, SGD+Momentum | Adamは汎用的、SGDは収束が良い  
  
### デバッグテクニック
    
    
    class ModelDebugger:
        """モデルのデバッグツール"""
    
        @staticmethod
        def check_gradients(model):
            """勾配の確認"""
            print("\n=== Gradient Check ===")
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    grad_max = param.grad.abs().max().item()
                    print(f"{name:30s} - Mean: {grad_mean:8.6f}, "
                          f"Std: {grad_std:8.6f}, Max: {grad_max:8.6f}")
    
                    # 勾配消失・爆発の警告
                    if grad_max < 1e-6:
                        print(f"  ⚠️  WARNING: Vanishing gradient!")
                    if grad_max > 100:
                        print(f"  ⚠️  WARNING: Exploding gradient!")
    
        @staticmethod
        def check_weights(model):
            """重みの統計を確認"""
            print("\n=== Weight Statistics ===")
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    print(f"{name:30s} - Mean: {weight_mean:8.6f}, Std: {weight_std:8.6f}")
    
        @staticmethod
        def check_nan_inf(model):
            """NaN/Infの検出"""
            print("\n=== NaN/Inf Check ===")
            has_issue = False
            for name, param in model.named_parameters():
                if torch.isnan(param.data).any():
                    print(f"  ❌ NaN detected in {name}")
                    has_issue = True
                if torch.isinf(param.data).any():
                    print(f"  ❌ Inf detected in {name}")
                    has_issue = True
    
            if not has_issue:
                print("  ✅ No NaN/Inf detected")
    
        @staticmethod
        def visualize_activation_distribution(model, data, device):
            """活性化の分布を可視化"""
            activations = {}
    
            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output.detach().cpu()
                return hook
    
            # フックを登録
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
    
            # 順伝播
            model.eval()
            with torch.no_grad():
                _ = model(data.to(device))
    
            # フックを削除
            for hook in hooks:
                hook.remove()
    
            # 可視化
            fig, axes = plt.subplots(len(activations), 1, figsize=(10, 3*len(activations)))
            if len(activations) == 1:
                axes = [axes]
    
            for ax, (name, activation) in zip(axes, activations.items()):
                activation_flat = activation.flatten().numpy()
                ax.hist(activation_flat, bins=50, alpha=0.7)
                ax.set_title(f'Activation Distribution: {name}')
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # 使用例
    debugger = ModelDebugger()
    
    # 訓練ループ内で使用
    for epoch in range(num_epochs):
        # 訓練
        # ...
    
        # デバッグチェック
        debugger.check_gradients(model)
        debugger.check_nan_inf(model)
    

### GPU活用とメモリ最適化
    
    
    import torch.cuda as cuda
    
    class GPUOptimizer:
        """GPU最適化のユーティリティ"""
    
        @staticmethod
        def get_gpu_info():
            """GPU情報の取得"""
            if not torch.cuda.is_available():
                print("CUDA is not available")
                return
    
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
            print(f"Cached Memory: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    
        @staticmethod
        def clear_cache():
            """GPUキャッシュのクリア"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU cache cleared")
    
        @staticmethod
        def mixed_precision_training_example(model, train_loader, device):
            """Mixed Precision Training（FP16）の例"""
            from torch.cuda.amp import autocast, GradScaler
    
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scaler = GradScaler()
    
            model.train()
    
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
    
                # Mixed Precision
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
    
                # Scalerを使用した逆伝播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    
            print("Mixed precision training completed")
    
    # GPU情報の表示
    GPUOptimizer.get_gpu_info()
    
    # メモリ最適化のTips
    def optimize_memory_usage():
        """メモリ使用量の最適化"""
    
        tips = [
            "1. バッチサイズを小さくする（64 → 32 → 16）",
            "2. Gradient Accumulation を使用（複数ステップで勾配を累積）",
            "3. Mixed Precision Training（FP16）を使用",
            "4. 不要な中間結果は torch.no_grad() で包む",
            "5. del 文で不要な変数を明示的に削除",
            "6. torch.cuda.empty_cache() でキャッシュをクリア",
            "7. DataLoader の num_workers を調整（CPU/GPU バランス）",
            "8. Inplace 操作を活用（x = x + 1 → x += 1）"
        ]
    
        print("\n=== メモリ最適化のTips ===")
        for tip in tips:
            print(tip)
    
    optimize_memory_usage()
    

### 本番環境へのデプロイ
    
    
    import torch.jit
    
    class ModelDeployment:
        """モデルのデプロイメント"""
    
        @staticmethod
        def export_to_torchscript(model, example_input, save_path='model_scripted.pt'):
            """TorchScriptへのエクスポート"""
            model.eval()
    
            # Scriptモード
            scripted_model = torch.jit.script(model)
            scripted_model.save(save_path)
            print(f"Model exported to TorchScript: {save_path}")
    
            return scripted_model
    
        @staticmethod
        def export_to_onnx(model, example_input, save_path='model.onnx'):
            """ONNXフォーマットへのエクスポート"""
            model.eval()
    
            torch.onnx.export(
                model,
                example_input,
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"Model exported to ONNX: {save_path}")
    
        @staticmethod
        def optimize_for_inference(model):
            """推論用の最適化"""
            model.eval()
    
            # 推論モードに設定
            for param in model.parameters():
                param.requires_grad = False
    
            # Batchnorm と Dropout を固定
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                if isinstance(module, nn.Dropout):
                    module.p = 0
    
            return model
    
        @staticmethod
        def benchmark_inference(model, example_input, device, num_runs=100):
            """推論速度のベンチマーク"""
            import time
    
            model.eval()
            model = model.to(device)
            example_input = example_input.to(device)
    
            # ウォームアップ
            with torch.no_grad():
                for _ in range(10):
                    _ = model(example_input)
    
            # ベンチマーク
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
    
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(example_input)
    
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
    
            avg_time = (end_time - start_time) / num_runs
            fps = 1 / avg_time
    
            print(f"\n=== Inference Benchmark ===")
            print(f"Average inference time: {avg_time*1000:.2f} ms")
            print(f"Throughput: {fps:.2f} FPS")
            print(f"Total runs: {num_runs}")
    
    # 使用例
    deployment = ModelDeployment()
    
    # モデルのロード
    model = CIFAR10Net().to(device)
    model.load_state_dict(torch.load('cifar10_best.pth'))
    
    # 推論用に最適化
    model = deployment.optimize_for_inference(model)
    
    # エクスポート
    example_input = torch.randn(1, 3, 32, 32).to(device)
    # deployment.export_to_torchscript(model, example_input)
    # deployment.export_to_onnx(model, example_input)
    
    # ベンチマーク
    deployment.benchmark_inference(model, example_input, device)
    

* * *

## 本章のまとめ

### 学んだこと

  1. **MNIST手書き数字認識**

     * 完全なデータ準備パイプライン
     * 効率的なCNNアーキテクチャ設計
     * 混同行列とエラー分析
  2. **CIFAR-10カラー画像分類**

     * データ拡張の実践的な活用
     * VGGスタイルの深いネットワーク
     * Batch NormalizationとDropout
  3. **高度なテクニック**

     * 転移学習とファインチューニング
     * Learning Rate Finder
     * モデルアンサンブル
     * Grad-CAM可視化
  4. **実践的なスキル**

     * ハイパーパラメータチューニング
     * デバッグテクニック
     * GPU最適化
     * 本番デプロイメント

### ベストプラクティスチェックリスト

項目 | 重要度 | 説明  
---|---|---  
✅ データ正規化 | 必須 | 平均0、標準偏差1に正規化  
✅ データ拡張 | 高 | 訓練データの多様性を増やす  
✅ Batch Normalization | 高 | 学習の安定化と高速化  
✅ Dropout | 高 | 過学習の防止  
✅ Learning Rate Scheduling | 中 | 学習率の動的調整  
✅ Early Stopping | 中 | 過学習の早期検出  
✅ モデル保存 | 必須 | ベストモデルのチェックポイント  
✅ 評価指標の記録 | 必須 | 訓練過程の可視化  
  
* * *

## 演習問題

### 問題1（難易度：medium）

MNISTモデルに以下の改善を加えて、精度を99%以上にしてください：

  * Batch Normalizationを追加
  * より深いネットワーク（3層以上の畳み込み層）
  * データ拡張（回転、平行移動）

ヒント

  * 各畳み込み層の後にBatch Normalizationを追加
  * RandomRotation と RandomAffine を使用
  * 学習率を適切に調整（0.001前後）

解答例
    
    
    class ImprovedMNISTNet(nn.Module):
        """改善版MNISTネットワーク"""
    
        def __init__(self):
            super(ImprovedMNISTNet, self).__init__()
    
            # 畳み込み層（3ブロック）
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
    
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
    
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
    
            self.pool = nn.MaxPool2d(2, 2)
    
            # 全結合層
            self.fc1 = nn.Linear(128 * 3 * 3, 256)
            self.bn_fc = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 10)
    
            self.dropout = nn.Dropout(0.5)
    
        def forward(self, x):
            # ブロック1: 28×28 → 14×14
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
    
            # ブロック2: 14×14 → 7×7
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
    
            # ブロック3: 7×7 → 3×3
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
    
            # 平坦化と全結合層
            x = x.view(-1, 128 * 3 * 3)
            x = F.relu(self.bn_fc(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)
    
            return x
    
    # データ拡張
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 訓練
    # model = ImprovedMNISTNet().to(device)
    # 期待される精度: 99.2%以上
    

### 問題2（難易度：hard）

CIFAR-10で転移学習を使って、20エポック以内に90%以上の精度を達成してください。ResNet50を使用し、適切なファインチューニング戦略を実装してください。

ヒント

  * Phase 1: 分類器のみ訓練（5エポック、lr=0.001）
  * Phase 2: 最後の残差ブロックを解凍（10エポック、lr=0.0001）
  * Phase 3: 全層ファインチューニング（5エポック、lr=0.00001）

### 問題3（難易度：medium）

Learning Rate Finderを実装し、最適な学習率を見つけてください。その学習率を使ってモデルを訓練し、結果を比較してください。

解答例
    
    
    # LRFinderを使用
    model = CIFAR10Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1, num_iter=200)
    lr_finder.plot()
    
    # 推奨学習率を使用
    # 出力例: "推奨学習率: 1.2e-02"
    suggested_lr = 0.012
    
    # 新しいモデルで訓練
    model = CIFAR10Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=suggested_lr, momentum=0.9)
    # 訓練実行...
    

### 問題4（難易度：hard）

3つの異なるモデル（異なるアーキテクチャまたは異なる初期化）を訓練し、アンサンブルで精度を向上させてください。個々のモデルより2%以上の改善を目指してください。

ヒント

  * モデル1: CIFAR10Net（VGGスタイル）
  * モデル2: ResNet18（転移学習）
  * モデル3: DenseNet（転移学習）
  * アンサンブル方法: 予測確率の平均、または多数決

### 問題5（難易度：hard）

Mixed Precision Training（FP16）を実装し、通常の訓練と比較してください。速度とメモリ使用量、精度の差を測定してください。

解答例
    
    
    from torch.cuda.amp import autocast, GradScaler
    import time
    
    def train_with_mixed_precision(model, train_loader, num_epochs=10):
        """Mixed Precision Trainingの実装"""
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()
    
        start_time = time.time()
    
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
    
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
    
                # Mixed Precision
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
    
                # Scaled backpropagation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    
                running_loss += loss.item()
    
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f}")
    
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds")
    
        # メモリ使用量
        if torch.cuda.is_available():
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # 通常訓練と比較
    model_fp32 = CIFAR10Net().to(device)
    model_fp16 = CIFAR10Net().to(device)
    
    print("=== FP32 Training ===")
    # 通常訓練...
    
    print("\n=== FP16 Training ===")
    train_with_mixed_precision(model_fp16, train_loader)
    
    # 期待される結果:
    # - FP16は1.5-2倍高速
    # - メモリ使用量は30-40%削減
    # - 精度はほぼ同等（±0.5%以内）
    

* * *

## 参考文献

  1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." _Proceedings of the IEEE_ , 86(11), 2278-2324.
  2. Krizhevsky, A., & Hinton, G. (2009). "Learning multiple layers of features from tiny images." _Technical Report_ , University of Toronto.
  3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." _CVPR_.
  4. Smith, L. N. (2017). "Cyclical learning rates for training neural networks." _WACV_.
  5. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual explanations from deep networks via gradient-based localization." _ICCV_.
  6. Ioffe, S., & Szegedy, C. (2015). "Batch normalization: Accelerating deep network training by reducing internal covariate shift." _ICML_.
