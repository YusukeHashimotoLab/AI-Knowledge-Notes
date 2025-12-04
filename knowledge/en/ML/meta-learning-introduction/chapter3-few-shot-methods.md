---
title: "Chapter 3: Few-Shot Learning Methods"
chapter_title: "Chapter 3: Few-Shot Learning Methods"
subtitle: Metric Learning-Based Few-Sample Classification Architectures
reading_time: 32 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 4
---

This chapter covers Few. You will learn pair learning with Siamese Networks and learnable distance metrics in Relation Networks.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand pair learning with Siamese Networks and Contrastive Loss
  * ✅ Implement prototype-based classification with Prototypical Networks
  * ✅ Implement classification using the attention mechanism of Matching Networks
  * ✅ Understand learnable distance metrics in Relation Networks
  * ✅ Design and conduct comparative experiments of Few-Shot learning methods

## 1\. Siamese Networks

### 1.1 Principles of Pair Learning

Siamese Networks are architectures that process two inputs through the same network (with shared weights) and learn their similarity. In few-shot learning, this is a fundamental method for directly learning relationships between samples.
    
    
    ```mermaid
    graph LR
        A[Image 1] --> B[CNN]
        C[Image 2] --> D[CNN]
        B --> E[Embedding 1]
        D --> F[Embedding 2]
        E --> G[Distance Calculation]
        F --> G
        G --> H[Similarity Score]
    
        style B fill:#9d4edd
        style D fill:#9d4edd
        style G fill:#3182ce
    ```

**Key Characteristics:**

  * **Weight Sharing:** Learns a consistent feature space by applying the same network to both inputs
  * **Pairwise Learning:** Directly learns whether two samples belong to the same class or different classes
  * **Metric Learning:** Semantically similar samples are placed close together, while different samples are placed far apart

### 1.2 Contrastive Loss

Contrastive Loss is a loss function that learns to place pairs of the same class close together and pairs of different classes far apart.

**Mathematical Definition:**

$$ \mathcal{L}(x_1, x_2, y) = y \cdot d(x_1, x_2)^2 + (1-y) \cdot \max(0, m - d(x_1, x_2))^2 $$ 

Where:

  * $d(x_1, x_2)$ is the Euclidean distance
  * $y \in \\{0, 1\\}$ is the label (1=same class, 0=different class)
  * $m$ is the margin (minimum distance between different classes)

### 1.3 Similarity Learning with Image Pairs
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SiameseNetwork(nn.Module):
        """Siamese Network Implementation"""
    
        def __init__(self, input_channels=3, embedding_dim=128):
            super(SiameseNetwork, self).__init__()
    
            # Shared feature extractor
            self.encoder = nn.Sequential(
                # Conv Block 1
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
    
                # Conv Block 2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
    
                # Conv Block 3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
    
                nn.Flatten(),
            )
    
            # Fully connected layers to embedding space
            self.fc = nn.Sequential(
                nn.Linear(256 * 8 * 8, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, embedding_dim)
            )
    
        def forward_one(self, x):
            """Transform a single input into embedding space"""
            x = self.encoder(x)
            x = self.fc(x)
            return F.normalize(x, p=2, dim=1)  # L2 normalization
    
        def forward(self, x1, x2):
            """Process pair inputs"""
            emb1 = self.forward_one(x1)
            emb2 = self.forward_one(x2)
            return emb1, emb2
    
    class ContrastiveLoss(nn.Module):
        """Contrastive Loss Implementation"""
    
        def __init__(self, margin=1.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin
    
        def forward(self, emb1, emb2, label):
            """
            Args:
                emb1, emb2: Embedding vectors (batch_size, embedding_dim)
                label: Label (1=same class, 0=different class)
            """
            # Euclidean distance
            distance = F.pairwise_distance(emb1, emb2, p=2)
    
            # Contrastive Loss
            loss_positive = label * torch.pow(distance, 2)
            loss_negative = (1 - label) * torch.pow(
                torch.clamp(self.margin - distance, min=0.0), 2
            )
    
            loss = torch.mean(loss_positive + loss_negative)
            return loss
    
    # Training example
    def train_siamese(model, train_loader, num_epochs=10):
        """Train Siamese Network"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = ContrastiveLoss(margin=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
    
            for batch_idx, (img1, img2, labels) in enumerate(train_loader):
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
    
                # Forward
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, labels.float())
    
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Usage example
    model = SiameseNetwork(input_channels=3, embedding_dim=128)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    

## 2\. Prototypical Networks

### 2.1 Computing Prototypes (Class Centers)

Prototypical Networks compute a "prototype" (representative embedding vector) for each class and classify new samples to the class of the nearest prototype.
    
    
    ```mermaid
    graph TB
        subgraph Support Set
            A1[Class A Sample 1] --> E1[Encoder]
            A2[Class A Sample 2] --> E2[Encoder]
            B1[Class B Sample 1] --> E3[Encoder]
            B2[Class B Sample 2] --> E4[Encoder]
        end
    
        E1 --> PA[Prototype AAverage]
        E2 --> PA
        E3 --> PB[Prototype BAverage]
        E4 --> PB
    
        Q[Query] --> EQ[Encoder]
        EQ --> D[Distance Calculation]
        PA --> D
        PB --> D
        D --> C[Classification]
    
        style PA fill:#9d4edd
        style PB fill:#9d4edd
        style D fill:#3182ce
    ```

**Prototype Definition:**

$$ c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\theta(x_i) $$ 

Where:

  * $c_k$ is the prototype of class $k$
  * $S_k$ is the support set of class $k$
  * $f_\theta$ is the encoder network

### 2.2 Euclidean Distance-Based Classification

The class probability for a query sample $x$ is computed using softmax:

$$ P(y=k|x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'} \exp(-d(f_\theta(x), c_{k'}))} $$ 

### 2.3 PyTorch Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class PrototypicalNetwork(nn.Module):
        """Prototypical Network Implementation"""
    
        def __init__(self, input_channels=3, hidden_dim=64):
            super(PrototypicalNetwork, self).__init__()
    
            # Feature extractor (4-layer CNN blocks)
            def conv_block(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
    
            self.encoder = nn.Sequential(
                conv_block(input_channels, hidden_dim),
                conv_block(hidden_dim, hidden_dim),
                conv_block(hidden_dim, hidden_dim),
                conv_block(hidden_dim, hidden_dim),
                nn.Flatten()
            )
    
        def forward(self, support_images, support_labels, query_images, n_way, k_shot):
            """
            Args:
                support_images: (n_way * k_shot, C, H, W)
                support_labels: (n_way * k_shot,)
                query_images: (n_query, C, H, W)
                n_way: Number of classes
                k_shot: Number of samples per class
            """
            # Embeddings for support set and query set
            support_embeddings = self.encoder(support_images)
            query_embeddings = self.encoder(query_images)
    
            # Compute prototypes (average of each class)
            prototypes = self.compute_prototypes(
                support_embeddings, support_labels, n_way
            )
    
            # Calculate distances between queries and prototypes
            distances = self.euclidean_distance(query_embeddings, prototypes)
    
            # Use negative distance as logits
            logits = -distances
            return logits
    
        def compute_prototypes(self, embeddings, labels, n_way):
            """Compute prototype for each class"""
            prototypes = torch.zeros(n_way, embeddings.size(1), device=embeddings.device)
    
            for k in range(n_way):
                # Mask for samples belonging to class k
                mask = (labels == k)
                # Compute average of samples in class k
                prototypes[k] = embeddings[mask].mean(dim=0)
    
            return prototypes
    
        def euclidean_distance(self, x, y):
            """
            Calculate Euclidean distance
            Args:
                x: (n_query, d)
                y: (n_way, d)
            Returns:
                distances: (n_query, n_way)
            """
            n = x.size(0)
            m = y.size(0)
            d = x.size(1)
    
            # Efficiently compute using broadcasting
            x = x.unsqueeze(1).expand(n, m, d)  # (n, m, d)
            y = y.unsqueeze(0).expand(n, m, d)  # (n, m, d)
    
            return torch.pow(x - y, 2).sum(2)  # (n, m)
    
    def train_prototypical(model, train_loader, num_epochs=100, n_way=5, k_shot=1):
        """Train Prototypical Network"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_acc = 0
    
            for batch_idx, (support_imgs, support_labels, query_imgs, query_labels) in enumerate(train_loader):
                support_imgs = support_imgs.to(device)
                support_labels = support_labels.to(device)
                query_imgs = query_imgs.to(device)
                query_labels = query_labels.to(device)
    
                # Forward
                logits = model(support_imgs, support_labels, query_imgs, n_way, k_shot)
                loss = criterion(logits, query_labels)
    
                # Accuracy calculation
                pred = logits.argmax(dim=1)
                acc = (pred == query_labels).float().mean()
    
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
                total_acc += acc.item()
    
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    
    # Usage example
    model = PrototypicalNetwork(input_channels=3, hidden_dim=64)
    print(f"Model architecture:\n{model}")
    

## 3\. Matching Networks

### 3.1 Utilizing Attention Mechanism

Matching Networks use an attention mechanism between the query sample and each sample in the support set, computing class probabilities through a weighted average. This enables classification that considers the context of the entire support set.
    
    
    ```mermaid
    graph TB
        subgraph Support Set
            S1[Support 1] --> ES1[Embedding]
            S2[Support 2] --> ES2[Embedding]
            S3[Support 3] --> ES3[Embedding]
        end
    
        Q[Query] --> EQ[Embedding + LSTM]
    
        EQ --> A1[AttentionWeight 1]
        EQ --> A2[AttentionWeight 2]
        EQ --> A3[AttentionWeight 3]
    
        ES1 --> A1
        ES2 --> A2
        ES3 --> A3
    
        A1 --> W[Weighted Average]
        A2 --> W
        A3 --> W
    
        W --> P[Prediction]
    
        style EQ fill:#9d4edd
        style W fill:#3182ce
    ```

### 3.2 Full Context Embeddings

An important feature of Matching Networks is generating embeddings that consider the context of the entire support set. This is achieved using sequential models such as LSTMs.

**Attention Weight Calculation:**

$$ a(x, x_i) = \frac{\exp(c(\hat{x}, \hat{x}_i))}{\sum_j \exp(c(\hat{x}, \hat{x}_j))} $$ 

**Prediction Distribution:**

$$ P(y|x, S) = \sum_{i=1}^k a(x, x_i) y_i $$ 

### 3.3 Implementation and Evaluation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MatchingNetwork(nn.Module):
        """Matching Network Implementation"""
    
        def __init__(self, input_channels=3, hidden_dim=64, lstm_layers=1):
            super(MatchingNetwork, self).__init__()
    
            # Feature extractor (CNN encoder)
            self.encoder = nn.Sequential(
                self._conv_block(input_channels, hidden_dim),
                self._conv_block(hidden_dim, hidden_dim),
                self._conv_block(hidden_dim, hidden_dim),
                self._conv_block(hidden_dim, hidden_dim),
            )
    
            # Calculate embedding dimension
            self.embedding_dim = hidden_dim * 5 * 5
    
            # LSTM for Full Context Embeddings
            self.lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.embedding_dim,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True
            )
    
            # Transform bidirectional LSTM output to original dimension
            self.fc = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
    
        def _conv_block(self, in_channels, out_channels):
            """CNN block"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
    
        def encode(self, x):
            """Convert image to embedding vector"""
            batch_size = x.size(0)
            x = self.encoder(x)
            x = x.view(batch_size, -1)
            return x
    
        def full_context_embeddings(self, embeddings):
            """
            Consider the context of the entire support set with LSTM
            Args:
                embeddings: (batch_size, seq_len, embedding_dim)
            """
            output, _ = self.lstm(embeddings)
            output = self.fc(output)
            return output
    
        def attention(self, query_emb, support_emb):
            """
            Calculate attention weights
            Args:
                query_emb: (n_query, embedding_dim)
                support_emb: (n_support, embedding_dim)
            Returns:
                attention_weights: (n_query, n_support)
            """
            # Calculate cosine similarity
            query_norm = F.normalize(query_emb, p=2, dim=1)
            support_norm = F.normalize(support_emb, p=2, dim=1)
    
            similarities = torch.mm(query_norm, support_norm.t())
    
            # Convert to attention weights with softmax
            attention_weights = F.softmax(similarities, dim=1)
            return attention_weights
    
        def forward(self, support_images, support_labels, query_images, n_way):
            """
            Args:
                support_images: (n_way * k_shot, C, H, W)
                support_labels: (n_way * k_shot,) one-hot encoded
                query_images: (n_query, C, H, W)
            """
            # Compute embeddings
            support_emb = self.encode(support_images)  # (n_support, emb_dim)
            query_emb = self.encode(query_images)      # (n_query, emb_dim)
    
            # Full Context Embeddings (support set only)
            support_emb_context = self.full_context_embeddings(
                support_emb.unsqueeze(0)  # (1, n_support, emb_dim)
            ).squeeze(0)  # (n_support, emb_dim)
    
            # Calculate attention weights
            attention_weights = self.attention(query_emb, support_emb_context)
    
            # Convert to one-hot labels
            support_labels_one_hot = F.one_hot(support_labels, n_way).float()
    
            # Attention-weighted prediction
            predictions = torch.mm(attention_weights, support_labels_one_hot)
    
            return predictions
    
    # Training function
    def train_matching(model, train_loader, num_epochs=100, n_way=5):
        """Train Matching Network"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_acc = 0
    
            for batch_idx, (support_imgs, support_labels, query_imgs, query_labels) in enumerate(train_loader):
                support_imgs = support_imgs.to(device)
                support_labels = support_labels.to(device)
                query_imgs = query_imgs.to(device)
                query_labels = query_labels.to(device)
    
                # Forward
                predictions = model(support_imgs, support_labels, query_imgs, n_way)
                loss = criterion(predictions, query_labels)
    
                # Accuracy calculation
                pred = predictions.argmax(dim=1)
                acc = (pred == query_labels).float().mean()
    
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
                total_acc += acc.item()
    
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    
    # Usage example
    model = MatchingNetwork(input_channels=3, hidden_dim=64)
    

## 4\. Relation Networks

### 4.1 Learnable Distance Metrics

Relation Networks compute similarity using a learnable neural network instead of fixed Euclidean distance or cosine similarity. This enables learning task-specific optimal distance functions.
    
    
    ```mermaid
    graph TB
        S[Support] --> ES[Feature Extractor]
        Q[Query] --> EQ[Feature Extractor]
    
        ES --> C[Concatenation]
        EQ --> C
    
        C --> R[Relation ModuleCNN]
        R --> SC[Similarity Score]
    
        style ES fill:#9d4edd
        style EQ fill:#9d4edd
        style R fill:#3182ce
    ```

**Relation Score Calculation:**

$$ r_{i,j} = g_\phi(\text{concat}(f_\theta(x_i), f_\theta(x_j))) $$ 

Where:

  * $f_\theta$ is the feature extractor
  * $g_\phi$ is the relation module (learnable CNN)
  * $r_{i,j}$ is the similarity score between samples $i$ and $j$

### 4.2 CNN-Based Relation Module

The relation module is a convolutional network that outputs a similarity score from concatenated feature vectors.

### 4.3 Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class RelationNetwork(nn.Module):
        """Relation Network Implementation"""
    
        def __init__(self, input_channels=3, feature_dim=64):
            super(RelationNetwork, self).__init__()
    
            # Feature extractor (encoder)
            self.encoder = nn.Sequential(
                self._conv_block(input_channels, feature_dim),
                self._conv_block(feature_dim, feature_dim),
                self._conv_block(feature_dim, feature_dim),
                self._conv_block(feature_dim, feature_dim),
            )
    
            # Relation module (calculates similarity from concatenated features)
            self.relation_module = nn.Sequential(
                self._conv_block(feature_dim * 2, feature_dim),
                self._conv_block(feature_dim, feature_dim),
                nn.Flatten(),
                nn.Linear(feature_dim * 5 * 5, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()  # Normalize similarity score to [0, 1]
            )
    
        def _conv_block(self, in_channels, out_channels):
            """CNN block"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
    
        def forward(self, support_images, query_images, n_way, k_shot):
            """
            Args:
                support_images: (n_way * k_shot, C, H, W)
                query_images: (n_query, C, H, W)
            """
            # Feature extraction
            support_features = self.encoder(support_images)  # (n_support, D, H, W)
            query_features = self.encoder(query_images)      # (n_query, D, H, W)
    
            n_support = support_features.size(0)
            n_query = query_features.size(0)
            D, H, W = support_features.size(1), support_features.size(2), support_features.size(3)
    
            # Compute prototypes for support set (average of each class)
            support_features_proto = support_features.view(n_way, k_shot, D, H, W).mean(dim=1)
    
            # Create query-prototype pairs
            # Expand query features: (n_query, n_way, D, H, W)
            query_features_ext = query_features.unsqueeze(1).repeat(1, n_way, 1, 1, 1)
    
            # Expand prototype features: (n_query, n_way, D, H, W)
            support_features_ext = support_features_proto.unsqueeze(0).repeat(n_query, 1, 1, 1, 1)
    
            # Concatenate features
            relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
            relation_pairs = relation_pairs.view(-1, D * 2, H, W)
    
            # Calculate relation scores
            relation_scores = self.relation_module(relation_pairs).view(n_query, n_way)
    
            return relation_scores
    
    class MSELoss4RelationNetwork(nn.Module):
        """MSE Loss for Relation Network"""
    
        def forward(self, relation_scores, labels, n_way):
            """
            Args:
                relation_scores: (n_query, n_way)
                labels: (n_query,)
            """
            # Create one-hot labels
            one_hot_labels = F.one_hot(labels, n_way).float()
    
            # MSE Loss
            loss = F.mse_loss(relation_scores, one_hot_labels)
            return loss
    
    # Training function
    def train_relation(model, train_loader, num_epochs=100, n_way=5, k_shot=1):
        """Train Relation Network"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = MSELoss4RelationNetwork()
    
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_acc = 0
    
            for batch_idx, (support_imgs, support_labels, query_imgs, query_labels) in enumerate(train_loader):
                support_imgs = support_imgs.to(device)
                query_imgs = query_imgs.to(device)
                query_labels = query_labels.to(device)
    
                # Forward
                relation_scores = model(support_imgs, query_imgs, n_way, k_shot)
                loss = criterion(relation_scores, query_labels, n_way)
    
                # Accuracy calculation
                pred = relation_scores.argmax(dim=1)
                acc = (pred == query_labels).float().mean()
    
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
                total_acc += acc.item()
    
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    
    # Usage example
    model = RelationNetwork(input_channels=3, feature_dim=64)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    

## 5\. Practice: Comparative Experiments of Methods

### 5.1 miniImageNet Dataset

miniImageNet is a subset of ImageNet widely used as a benchmark for few-shot learning.

**Dataset Structure:**

Split | Number of Classes | Samples per Class | Purpose  
---|---|---|---  
Train | 64 | 600 | Meta-learning training  
Validation | 16 | 600 | Hyperparameter tuning  
Test | 20 | 600 | Final performance evaluation  
  
### 5.2 5-way 1-shot/5-shot Evaluation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    
    def evaluate_few_shot(model, test_loader, n_way=5, k_shot=1, n_query=15, n_episodes=600):
        """
        Evaluate Few-Shot learning model
    
        Args:
            model: Model to evaluate (Prototypical, Matching, or Relation)
            test_loader: Test data loader
            n_way: Number of classes
            k_shot: Number of samples in support set
            n_query: Number of samples in query set
            n_episodes: Number of evaluation episodes
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
    
        accuracies = []
    
        with torch.no_grad():
            for episode in range(n_episodes):
                # Sample episode data
                support_imgs, support_labels, query_imgs, query_labels = next(iter(test_loader))
    
                support_imgs = support_imgs.to(device)
                support_labels = support_labels.to(device)
                query_imgs = query_imgs.to(device)
                query_labels = query_labels.to(device)
    
                # Different inference methods depending on model
                if hasattr(model, 'relation_module'):  # Relation Network
                    predictions = model(support_imgs, query_imgs, n_way, k_shot)
                    pred_labels = predictions.argmax(dim=1)
                else:  # Prototypical or Matching Network
                    logits = model(support_imgs, support_labels, query_imgs, n_way, k_shot)
                    pred_labels = logits.argmax(dim=1)
    
                # Calculate accuracy
                acc = (pred_labels == query_labels).float().mean().item()
                accuracies.append(acc)
    
                if (episode + 1) % 100 == 0:
                    current_avg = np.mean(accuracies)
                    current_std = np.std(accuracies)
                    print(f"Episode [{episode+1}/{n_episodes}], "
                          f"Acc: {current_avg:.4f} ± {1.96 * current_std / np.sqrt(len(accuracies)):.4f}")
    
        # Final results (95% confidence interval)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        confidence_interval = 1.96 * std_acc / np.sqrt(n_episodes)
    
        return mean_acc, confidence_interval
    
    # Example data loader setup
    class FewShotDataLoader:
        """Data loader for Few-Shot learning"""
    
        def __init__(self, dataset, n_way=5, k_shot=1, n_query=15):
            self.dataset = dataset
            self.n_way = n_way
            self.k_shot = k_shot
            self.n_query = n_query
    
        def sample_episode(self):
            """Sample data for one episode"""
            # Randomly select n_way classes
            classes = np.random.choice(len(self.dataset.classes), self.n_way, replace=False)
    
            support_imgs = []
            support_labels = []
            query_imgs = []
            query_labels = []
    
            for i, cls in enumerate(classes):
                # Select k_shot + n_query samples from class
                cls_samples = self.dataset.get_samples_by_class(cls)
                indices = np.random.choice(len(cls_samples), self.k_shot + self.n_query, replace=False)
    
                # Support set
                support_imgs.extend([cls_samples[idx] for idx in indices[:self.k_shot]])
                support_labels.extend([i] * self.k_shot)
    
                # Query set
                query_imgs.extend([cls_samples[idx] for idx in indices[self.k_shot:]])
                query_labels.extend([i] * self.n_query)
    
            return (torch.stack(support_imgs), torch.tensor(support_labels),
                    torch.stack(query_imgs), torch.tensor(query_labels))
    

### 5.3 Accuracy Comparison and Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    # Comparative experiment for each method
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def compare_few_shot_methods(test_loader, n_way=5, k_shot_list=[1, 5]):
        """Compare multiple Few-Shot learning methods"""
    
        results = []
    
        for k_shot in k_shot_list:
            print(f"\n{'='*50}")
            print(f"{n_way}-way {k_shot}-shot evaluation")
            print(f"{'='*50}\n")
    
            # Prototypical Network
            print("Evaluating Prototypical Network...")
            proto_model = PrototypicalNetwork(input_channels=3, hidden_dim=64)
            proto_acc, proto_ci = evaluate_few_shot(proto_model, test_loader, n_way, k_shot)
            results.append({
                'Method': 'Prototypical',
                'Setting': f'{n_way}-way {k_shot}-shot',
                'Accuracy': proto_acc,
                'CI': proto_ci
            })
            print(f"Prototypical Network: {proto_acc:.4f} ± {proto_ci:.4f}\n")
    
            # Matching Network
            print("Evaluating Matching Network...")
            match_model = MatchingNetwork(input_channels=3, hidden_dim=64)
            match_acc, match_ci = evaluate_few_shot(match_model, test_loader, n_way, k_shot)
            results.append({
                'Method': 'Matching',
                'Setting': f'{n_way}-way {k_shot}-shot',
                'Accuracy': match_acc,
                'CI': match_ci
            })
            print(f"Matching Network: {match_acc:.4f} ± {match_ci:.4f}\n")
    
            # Relation Network
            print("Evaluating Relation Network...")
            relation_model = RelationNetwork(input_channels=3, feature_dim=64)
            relation_acc, relation_ci = evaluate_few_shot(relation_model, test_loader, n_way, k_shot)
            results.append({
                'Method': 'Relation',
                'Setting': f'{n_way}-way {k_shot}-shot',
                'Accuracy': relation_acc,
                'CI': relation_ci
            })
            print(f"Relation Network: {relation_acc:.4f} ± {relation_ci:.4f}\n")
    
        return pd.DataFrame(results)
    
    # Visualize results
    def plot_comparison(results_df):
        """Visualize comparison results"""
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Separate 1-shot and 5-shot
        settings = results_df['Setting'].unique()
        x = np.arange(len(results_df['Method'].unique()))
        width = 0.35
    
        for i, setting in enumerate(settings):
            data = results_df[results_df['Setting'] == setting]
            accuracies = data['Accuracy'].values
            cis = data['CI'].values
    
            ax.bar(x + i * width, accuracies, width,
                   yerr=cis, label=setting, capsize=5)
    
        ax.set_xlabel('Method')
        ax.set_ylabel('Accuracy')
        ax.set_title('Few-Shot Learning Methods Comparison on miniImageNet')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(results_df['Method'].unique())
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('few_shot_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Execution example
    # results_df = compare_few_shot_methods(test_loader, n_way=5, k_shot_list=[1, 5])
    # plot_comparison(results_df)
    

**Typical Results (miniImageNet):**

Method | 5-way 1-shot | 5-way 5-shot | Key Features  
---|---|---|---  
Prototypical Networks | 49.42% ± 0.78% | 68.20% ± 0.66% | Simple and efficient  
Matching Networks | 46.60% ± 0.78% | 60.00% ± 0.71% | Attention mechanism  
Relation Networks | 50.44% ± 0.82% | 65.32% ± 0.70% | Learnable distance  
  
### 5.4 Analysis and Method Selection Guidelines

**Prototypical Networks**

  * **Advantages:** Simple implementation, computationally efficient, high accuracy on many tasks
  * **Disadvantages:** Depends on fixed Euclidean distance
  * **Recommended Use:** As a baseline method, in resource-constrained environments

**Matching Networks**

  * **Advantages:** Considers entire support set through attention mechanism
  * **Disadvantages:** High computational cost due to LSTM
  * **Recommended Use:** Tasks where relationships between support set samples are important

**Relation Networks**

  * **Advantages:** Can learn task-specific optimal distance functions
  * **Disadvantages:** More parameters, longer training time
  * **Recommended Use:** Tasks requiring complex similarity, when sufficient data is available

> **Practical Advice:** For new tasks, it is efficient to first try Prototypical Networks as a baseline, and consider Relation Networks if performance is insufficient.

## Exercises

**Exercise 1: Improving Siamese Networks**

Implement Triplet Loss for the provided Siamese Network and compare its performance with Contrastive Loss. Triplet Loss uses three samples: anchor, positive, and negative.
    
    
    class TripletLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(TripletLoss, self).__init__()
            self.margin = margin
    
        def forward(self, anchor, positive, negative):
            # Exercise: Implement Triplet Loss
            # Hint: L = max(0, d(a,p) - d(a,n) + margin)
            pass
    

**Exercise 2: Extending Prototypical Networks**

Extend Prototypical Networks to compute class prototypes using a weighted average with an attention mechanism instead of a simple average. This can reduce the impact of noisy samples.
    
    
    def compute_prototypes_with_attention(self, embeddings, labels, n_way):
        """Prototype computation using attention mechanism"""
        # Exercise: Implement
        # Hint: Calculate attention weights based on similarity between samples
        pass
    

**Exercise 3: Multimodal Few-Shot Learning**

Design a multimodal Prototypical Network that takes both images and text as input. Use a CNN for images and a Transformer for text, and combine both embeddings.
    
    
    class MultimodalPrototypicalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            pass
    
        def forward(self, images, texts):
            pass
    

**Exercise 4: Real Application of Few-Shot Learning**

Assuming a medical image diagnosis scenario, design a system to classify new diseases from limited case images (about 5 per disease). Explain which method would be most suitable and why. Also consider data augmentation and domain adaptation strategies.

## Summary

In this chapter, we learned about the major methods of few-shot learning:

  * **Siamese Networks:** Foundations of similarity learning through pair learning and Contrastive Loss
  * **Prototypical Networks:** Simple and effective prototype-based classification
  * **Matching Networks:** Context-aware classification using attention mechanism
  * **Relation Networks:** Flexible similarity computation using learnable distance metrics

Each of these methods has different strengths and can be selected according to the task and available resources. In the next chapter, we will learn how to combine these methods with more advanced optimization algorithms (such as MAML).
