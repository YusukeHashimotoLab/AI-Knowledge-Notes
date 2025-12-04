---
title: 第4章：転移学習
chapter_title: 第4章：転移学習
subtitle: 事前学習モデルを活用した効率的な学習とドメイン適応技術
reading_time: 30分
difficulty: 中級〜上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 転移学習の基礎概念と事前学習の効果を理解できる
  * ✅ ファインチューニング戦略を実装し、最適化できる
  * ✅ Domain Adaptationによる分布シフト問題を解決できる
  * ✅ 知識蒸留による軽量化とメタ学習の組み合わせができる
  * ✅ 実践的な転移学習プロジェクトを構築できる

## 1\. 転移学習の基礎

### 1.1 転移学習とは

転移学習は、あるタスク（ソースタスク）で学習した知識を別のタスク（ターゲットタスク）に活用する機械学習の手法です。メタ学習が「学習の学習」を目指すのに対し、転移学習は「知識の再利用」を目指します。
    
    
    ```mermaid
    graph LR
        A[Source DomainImageNet] -->|事前学習| B[Pre-trained Model]
        B -->|転移| C[Target DomainMedical Images]
        C -->|ファインチューニング| D[Specialized Model]
        style A fill:#e3f2fd
        style D fill:#c8e6c9
    ```

### 1.2 転移学習の種類

転移タイプ | 説明 | 具体例  
---|---|---  
**ドメイン転移** | 異なるデータ分布間での転移 | 自然画像 → 医療画像  
**タスク転移** | 異なるタスク間での転移 | 分類 → 物体検出  
**モデル転移** | モデル構造の再利用 | ResNet → カスタムアーキテクチャ  
  
### 1.3 転移可能性の評価

転移学習の成功は、ソースタスクとターゲットタスクの関連性に依存します：
    
    
    import torch
    import torch.nn as nn
    from torchvision import models
    from scipy.stats import spearmanr
    
    def compute_transferability_score(source_features, target_features):
        """
        転移可能性スコアの計算
    
        Args:
            source_features: ソースドメインの特徴量
            target_features: ターゲットドメインの特徴量
    
        Returns:
            transferability_score: 転移可能性スコア
        """
        # 特徴量の相関係数を計算
        correlation, _ = spearmanr(
            source_features.flatten(),
            target_features.flatten()
        )
    
        # 特徴量分布の距離（MMD）を計算
        def compute_mmd(x, y):
            xx = torch.mm(x, x.t())
            yy = torch.mm(y, y.t())
            xy = torch.mm(x, y.t())
            return xx.mean() + yy.mean() - 2 * xy.mean()
    
        mmd_distance = compute_mmd(
            torch.tensor(source_features),
            torch.tensor(target_features)
        )
    
        # 総合スコア（高いほど転移に適している）
        transferability_score = correlation - 0.1 * mmd_distance.item()
    
        return transferability_score
    
    # 使用例
    source_feats = torch.randn(100, 512)
    target_feats = torch.randn(100, 512)
    score = compute_transferability_score(source_feats, target_feats)
    print(f"Transferability Score: {score:.4f}")
    

## 2\. ファインチューニング戦略

### 2.1 全層 vs 部分層の更新

事前学習モデルのどの層を更新するかは、データ量とタスクの類似性によって決定します：
    
    
    import torch.nn as nn
    from torchvision import models
    
    class TransferLearningModel(nn.Module):
        def __init__(self, num_classes, freeze_strategy='partial'):
            """
            転移学習モデル
    
            Args:
                num_classes: ターゲットタスクのクラス数
                freeze_strategy: 'all', 'partial', 'none'
            """
            super().__init__()
            # ResNet50事前学習モデルをロード
            self.backbone = models.resnet50(pretrained=True)
    
            # 凍結戦略の適用
            if freeze_strategy == 'all':
                # 全層を凍結（分類層以外）
                for param in self.backbone.parameters():
                    param.requires_grad = False
    
            elif freeze_strategy == 'partial':
                # 初期層のみ凍結（特徴抽出器として使用）
                for name, param in self.backbone.named_parameters():
                    if 'layer4' not in name and 'fc' not in name:
                        param.requires_grad = False
    
            # 分類層を置き換え
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
        def forward(self, x):
            return self.backbone(x)
    
    # 戦略別のモデル作成
    model_frozen = TransferLearningModel(num_classes=10, freeze_strategy='all')
    model_partial = TransferLearningModel(num_classes=10, freeze_strategy='partial')
    model_full = TransferLearningModel(num_classes=10, freeze_strategy='none')
    
    # パラメータ数の確認
    for name, model in [('Frozen', model_frozen), ('Partial', model_partial), ('Full', model_full)]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{name}: {trainable:,} / {total:,} parameters trainable")
    

### 2.2 Discriminative Fine-Tuning

層ごとに異なる学習率を設定することで、より効果的なファインチューニングが可能です：
    
    
    def get_discriminative_params(model, base_lr=1e-4, multiplier=2.6):
        """
        層ごとに異なる学習率を設定
    
        Args:
            model: ニューラルネットワークモデル
            base_lr: 最終層の学習率
            multiplier: 層間の学習率倍率
    
        Returns:
            param_groups: 層別パラメータグループ
        """
        param_groups = []
    
        # ResNetの層グループ
        layer_groups = [
            ('layer1', model.backbone.layer1),
            ('layer2', model.backbone.layer2),
            ('layer3', model.backbone.layer3),
            ('layer4', model.backbone.layer4),
            ('fc', model.backbone.fc)
        ]
    
        # 層ごとに学習率を設定（深い層ほど高い学習率）
        for i, (name, layer) in enumerate(layer_groups):
            lr = base_lr / (multiplier ** (len(layer_groups) - i - 1))
            param_groups.append({
                'params': layer.parameters(),
                'lr': lr,
                'name': name
            })
            print(f"{name}: lr = {lr:.2e}")
    
        return param_groups
    
    # Discriminative Fine-Tuningの適用
    model = TransferLearningModel(num_classes=10, freeze_strategy='none')
    param_groups = get_discriminative_params(model, base_lr=1e-3)
    optimizer = torch.optim.Adam(param_groups)
    
    # 学習率スケジューラ
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    

## 3\. Domain Adaptation

### 3.1 Domain Shiftの問題

ソースドメインとターゲットドメインの分布が異なる場合、モデルの性能が低下します。Domain Adaptationは、この分布シフトを緩和する技術です。
    
    
    ```mermaid
    graph TB
        A[Source DomainP_s(X,Y)] -->|Training Data| B[Model]
        C[Target DomainP_t(X,Y)] -.->|No Labels| B
        B -->|Adapt| D[Domain-InvariantFeatures]
        D -->|Predict| E[Target Predictions]
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#c8e6c9
    ```

### 3.2 Domain Adversarial Neural Networks (DANN)

DANNは、特徴抽出器がドメインに不変な表現を学習するよう、敵対的訓練を使用します：
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class GradientReversalLayer(torch.autograd.Function):
        """勾配反転層（DANN用）"""
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)
    
        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.alpha * grad_output, None
    
    class DANN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
    
            # 特徴抽出器（ドメイン不変）
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 5 * 5, 1024), nn.ReLU(),
                nn.Dropout(0.5)
            )
    
            # クラス分類器
            self.class_classifier = nn.Sequential(
                nn.Linear(1024, 256), nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
    
            # ドメイン分類器
            self.domain_classifier = nn.Sequential(
                nn.Linear(1024, 256), nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)  # Source vs Target
            )
    
        def forward(self, x, alpha=1.0):
            features = self.feature_extractor(x)
    
            # クラス予測
            class_output = self.class_classifier(features)
    
            # ドメイン予測（勾配反転）
            reversed_features = GradientReversalLayer.apply(features, alpha)
            domain_output = self.domain_classifier(reversed_features)
    
            return class_output, domain_output
    
    # DANNの訓練
    def train_dann(model, source_loader, target_loader, epochs=50):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
        for epoch in range(epochs):
            model.train()
            # アルファ値を徐々に増加（勾配反転の強度）
            alpha = 2 / (1 + np.exp(-10 * epoch / epochs)) - 1
    
            for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
                # ソースドメインの損失
                source_class, source_domain = model(source_data, alpha)
                class_loss = F.cross_entropy(source_class, source_labels)
                source_domain_loss = F.cross_entropy(
                    source_domain,
                    torch.zeros(len(source_data), dtype=torch.long)
                )
    
                # ターゲットドメインの損失
                _, target_domain = model(target_data, alpha)
                target_domain_loss = F.cross_entropy(
                    target_domain,
                    torch.ones(len(target_data), dtype=torch.long)
                )
    
                # 総合損失
                loss = class_loss + source_domain_loss + target_domain_loss
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Alpha = {alpha:.4f}")
    
    # 使用例
    model = DANN(num_classes=10)
    print("DANN model created with gradient reversal layer")
    

### 3.3 Maximum Mean Discrepancy (MMD)

MMDは、ソースとターゲットの分布間の距離を最小化することでドメイン適応を実現します：
    
    
    def compute_mmd_loss(source_features, target_features, kernel='rbf'):
        """
        Maximum Mean Discrepancy損失の計算
    
        Args:
            source_features: ソース特徴量 (N_s, D)
            target_features: ターゲット特徴量 (N_t, D)
            kernel: カーネルタイプ ('rbf', 'linear')
    
        Returns:
            mmd_loss: MMD損失
        """
        def gaussian_kernel(x, y, sigma=1.0):
            x_size = x.size(0)
            y_size = y.size(0)
            dim = x.size(1)
    
            x = x.unsqueeze(1)  # (N_s, 1, D)
            y = y.unsqueeze(0)  # (1, N_t, D)
    
            diff = x - y  # (N_s, N_t, D)
            dist_sq = torch.sum(diff ** 2, dim=2)  # (N_s, N_t)
    
            return torch.exp(-dist_sq / (2 * sigma ** 2))
    
        if kernel == 'rbf':
            # RBFカーネル
            xx = gaussian_kernel(source_features, source_features).mean()
            yy = gaussian_kernel(target_features, target_features).mean()
            xy = gaussian_kernel(source_features, target_features).mean()
        else:
            # 線形カーネル
            xx = torch.mm(source_features, source_features.t()).mean()
            yy = torch.mm(target_features, target_features.t()).mean()
            xy = torch.mm(source_features, target_features.t()).mean()
    
        mmd_loss = xx + yy - 2 * xy
        return mmd_loss
    
    class MMDNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5)
            )
            self.classifier = nn.Linear(256, num_classes)
    
        def forward(self, x):
            features = self.feature_extractor(x)
            output = self.classifier(features)
            return output, features
    
    # MMDを使った訓練
    def train_with_mmd(model, source_loader, target_loader, lambda_mmd=0.1):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            # 特徴量と予測を取得
            source_pred, source_feat = model(source_data)
            _, target_feat = model(target_data)
    
            # 分類損失
            class_loss = F.cross_entropy(source_pred, source_labels)
    
            # MMD損失
            mmd_loss = compute_mmd_loss(source_feat, target_feat)
    
            # 総合損失
            total_loss = class_loss + lambda_mmd * mmd_loss
    
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
        return class_loss.item(), mmd_loss.item()
    

## 4\. 知識蒸留 (Knowledge Distillation)

### 4.1 Teacher-Student学習の基礎

知識蒸留は、大きなTeacherモデルの知識を小さなStudentモデルに転移する手法です。メタ学習と組み合わせることで、効率的な少数ショット学習が可能になります。
    
    
    ```mermaid
    graph LR
        A[Large Teacher ModelHigh Accuracy] -->|Soft Targets| B[KnowledgeDistillation]
        C[Training Data] --> B
        B -->|Transfer| D[Small Student ModelFast Inference]
        style A fill:#e3f2fd
        style D fill:#c8e6c9
    ```

### 4.2 温度付き蒸留の実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class DistillationLoss(nn.Module):
        def __init__(self, temperature=3.0, alpha=0.7):
            """
            知識蒸留の損失関数
    
            Args:
                temperature: ソフトマックスの温度パラメータ
                alpha: ハード損失とソフト損失のバランス
            """
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha
            self.ce_loss = nn.CrossEntropyLoss()
    
        def forward(self, student_logits, teacher_logits, targets):
            # ソフトターゲット損失（知識蒸留）
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
            distillation_loss = F.kl_div(
                soft_student, soft_targets, reduction='batchmean'
            ) * (self.temperature ** 2)
    
            # ハードターゲット損失（通常の分類）
            student_loss = self.ce_loss(student_logits, targets)
    
            # 総合損失
            total_loss = (
                self.alpha * distillation_loss +
                (1 - self.alpha) * student_loss
            )
    
            return total_loss, distillation_loss, student_loss
    
    # TeacherとStudentモデルの定義
    class TeacherModel(nn.Module):
        """大きなTeacherモデル"""
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 512), nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
        def forward(self, x):
            return self.features(x)
    
    class StudentModel(nn.Module):
        """軽量なStudentモデル"""
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
                nn.Linear(128, num_classes)
            )
    
        def forward(self, x):
            return self.features(x)
    
    # 知識蒸留の訓練
    def train_distillation(teacher, student, train_loader, epochs=50):
        # Teacherは評価モード
        teacher.eval()
        student.train()
    
        criterion = DistillationLoss(temperature=3.0, alpha=0.7)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                # Teacher予測（勾配なし）
                with torch.no_grad():
                    teacher_logits = teacher(images)
    
                # Student予測
                student_logits = student(images)
    
                # 蒸留損失
                loss, dist_loss, student_loss = criterion(
                    student_logits, teacher_logits, labels
                )
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # モデルサイズの比較
    teacher = TeacherModel()
    student = StudentModel()
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {teacher_params:,} parameters")
    print(f"Student: {student_params:,} parameters")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x")
    

### 4.3 メタ学習との組み合わせ
    
    
    class MetaDistillation(nn.Module):
        def __init__(self, teacher_model, student_model, inner_lr=0.01):
            """
            メタ学習と知識蒸留の組み合わせ
    
            Args:
                teacher_model: 大きなTeacherモデル
                student_model: 軽量なStudentモデル
                inner_lr: 内側ループの学習率
            """
            super().__init__()
            self.teacher = teacher_model
            self.student = student_model
            self.inner_lr = inner_lr
            self.temperature = 3.0
    
        def inner_loop(self, support_x, support_y, steps=5):
            """
            内側ループ：Studentモデルのタスク適応
            """
            # Studentモデルのコピーを作成
            adapted_params = [p.clone() for p in self.student.parameters()]
    
            for _ in range(steps):
                # Teacher予測
                with torch.no_grad():
                    teacher_logits = self.teacher(support_x)
    
                # Student予測
                student_logits = self.student(support_x)
    
                # 蒸留損失
                soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
                soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
                loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
    
                # 勾配計算と更新
                grads = torch.autograd.grad(loss, self.student.parameters())
                adapted_params = [
                    p - self.inner_lr * g
                    for p, g in zip(adapted_params, grads)
                ]
    
            return adapted_params
    
        def forward(self, support_x, support_y, query_x, query_y):
            """
            メタ蒸留の順伝播
            """
            # 内側ループで適応
            adapted_params = self.inner_loop(support_x, support_y)
    
            # 適応したモデルでクエリセットを評価
            query_logits = self.student(query_x)
            loss = F.cross_entropy(query_logits, query_y)
    
            return loss
    
    # 使用例
    teacher = TeacherModel(num_classes=5)
    student = StudentModel(num_classes=5)
    meta_distill = MetaDistillation(teacher, student, inner_lr=0.01)
    print("Meta-Distillation model initialized")
    

## 5\. 実践：転移学習プロジェクト

### プロジェクト：ImageNetから医療画像分類への転移学習

**目標：** ImageNetで事前学習したモデルを使って、少数の医療画像データで高精度な診断モデルを構築する。

### 5.1 完全な転移学習パイプライン
    
    
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    
    class MedicalImageDataset(Dataset):
        """医療画像データセット"""
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
    
        def __len__(self):
            return len(self.images)
    
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
    
            if self.transform:
                image = self.transform(image)
    
            return image, label
    
    class TransferLearningPipeline:
        def __init__(self, num_classes, device='cuda'):
            self.device = device
            self.num_classes = num_classes
    
            # データ拡張（Domain Adaptation用）
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
            self.val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
            # 事前学習モデルのロード
            self.model = self._create_model()
    
        def _create_model(self):
            """事前学習モデルの作成とカスタマイズ"""
            # ResNet50（ImageNet事前学習済み）
            model = models.resnet50(pretrained=True)
    
            # 初期層を凍結
            for name, param in model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
    
            # 分類層をカスタマイズ
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes)
            )
    
            return model.to(self.device)
    
        def train(self, train_data, val_data, epochs=50, use_distillation=False):
            """
            訓練実行
    
            Args:
                train_data: 訓練データ (images, labels)
                val_data: 検証データ (images, labels)
                epochs: エポック数
                use_distillation: 知識蒸留を使用するか
            """
            # データローダー
            train_dataset = MedicalImageDataset(*train_data, self.train_transform)
            val_dataset = MedicalImageDataset(*val_data, self.val_transform)
    
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
            # Discriminative Learning Rates
            optimizer = torch.optim.Adam([
                {'params': self.model.layer4.parameters(), 'lr': 1e-3},
                {'params': self.model.fc.parameters(), 'lr': 1e-2}
            ])
    
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )
    
            criterion = nn.CrossEntropyLoss()
    
            best_val_acc = 0.0
            history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
            for epoch in range(epochs):
                # 訓練フェーズ
                self.model.train()
                train_loss = 0.0
    
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
    
                    train_loss += loss.item()
    
                # 検証フェーズ
                val_loss, val_acc = self._validate(val_loader, criterion)
    
                # 履歴保存
                history['train_loss'].append(train_loss / len(train_loader))
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
    
                # ベストモデル保存
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'best_model.pth')
    
                scheduler.step()
    
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
            return history
    
        def _validate(self, val_loader, criterion):
            """検証"""
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
    
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
    
            val_loss /= len(val_loader)
            val_acc = correct / total
    
            return val_loss, val_acc
    
        def evaluate_transferability(self, source_data, target_data):
            """転移可能性の評価"""
            self.model.eval()
    
            def extract_features(data):
                features = []
                with torch.no_grad():
                    for images, _ in DataLoader(data, batch_size=32):
                        images = images.to(self.device)
                        # 最終層の前の特徴量を抽出
                        feat = self.model.layer4(
                            self.model.layer3(
                                self.model.layer2(
                                    self.model.layer1(
                                        self.model.conv1(images)
                                    )
                                )
                            )
                        )
                        features.append(feat.cpu().flatten(1))
                return torch.cat(features, dim=0)
    
            source_feats = extract_features(source_data)
            target_feats = extract_features(target_data)
    
            # MMDスコア計算
            score = compute_transferability_score(source_feats, target_feats)
            print(f"Transferability Score: {score:.4f}")
    
            return score
    
    # 使用例
    if __name__ == "__main__":
        # ダミーデータ（実際は医療画像データを使用）
        num_samples = 1000
        train_images = np.random.randint(0, 255, (num_samples, 224, 224, 3), dtype=np.uint8)
        train_labels = np.random.randint(0, 3, num_samples)
    
        val_images = np.random.randint(0, 255, (200, 224, 224, 3), dtype=np.uint8)
        val_labels = np.random.randint(0, 3, 200)
    
        # パイプライン実行
        pipeline = TransferLearningPipeline(num_classes=3, device='cpu')
    
        print("Starting transfer learning training...")
        history = pipeline.train(
            train_data=(train_images, train_labels),
            val_data=(val_images, val_labels),
            epochs=20
        )
    
        print("\nTraining completed!")
        print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    

## まとめ

この章では、転移学習の包括的な技術を学びました：

  * **転移学習の基礎：** 事前学習モデルの活用により、少数データでも高精度なモデル構築が可能
  * **ファインチューニング戦略：** 層別の学習率調整により効率的な知識転移を実現
  * **Domain Adaptation：** DANNやMMDによりドメインシフト問題を解決
  * **知識蒸留：** 大規模モデルの知識を軽量モデルに転移し、推論効率を向上
  * **実践プロジェクト：** ImageNetから医療画像への転移により実用的な応用を実現

> **重要ポイント：** 転移学習は、メタ学習と組み合わせることで、さらに強力な少数ショット学習システムを構築できます。事前学習で獲得した一般的な特徴表現と、メタ学習による高速適応能力を統合することが、実世界での成功の鍵となります。 

## 演習問題

**演習1：転移可能性の分析**

異なるソースドメイン（ImageNet、Places365、COCO）からターゲットドメイン（医療画像）への転移可能性を評価し、最適な事前学習モデルを選択してください。MMDスコアと実際の性能を比較してください。

**演習2：ファインチューニング戦略の比較**

以下の3つの戦略を比較実装してください：  
1) 全層凍結（分類層のみ訓練）  
2) 部分凍結（Layer4と分類層のみ訓練）  
3) Discriminative Fine-Tuning  
データ量を変化させ、各戦略の性能を評価してください。

**演習3：DANNの実装と評価**

ソースドメイン（MNIST）からターゲットドメイン（MNIST-M）へのDomain Adaptationを実装してください。勾配反転層のアルファ値を変化させ、ドメイン不変性と分類性能のトレードオフを分析してください。

**演習4：知識蒸留の最適化**

温度パラメータ（T=1, 3, 5, 10）とアルファ値（α=0.3, 0.5, 0.7, 0.9）を変化させて、Studentモデルの性能への影響を評価してください。最適なハイパーパラメータを見つけてください。

**演習5：メタ蒸留の実装**

MAMLと知識蒸留を組み合わせたメタ蒸留システムを実装してください。通常のMAML、通常の蒸留、メタ蒸留の3つの手法を比較し、少数ショット学習における効果を検証してください。

## 参考文献

  * Pan, S. J., & Yang, Q. (2009). "A Survey on Transfer Learning". IEEE Transactions on Knowledge and Data Engineering.
  * Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks". JMLR.
  * Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network". NIPS Deep Learning Workshop.
  * Yosinski, J., et al. (2014). "How transferable are features in deep neural networks?". NIPS.
  * Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification". ACL.
