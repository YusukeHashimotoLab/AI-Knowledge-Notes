---
title: 第4章：深層学習による推薦システム
chapter_title: 第4章：深層学習による推薦システム
subtitle: Neural Collaborative Filtering、Two-Tower Models、Sequence-based推薦
reading_time: 40-50分
difficulty: 中級〜上級
code_examples: 8
exercises: 6
---

## 学習目標

  * Neural Collaborative Filtering (NCF) の理論と実装を理解する
  * Factorization MachinesとDeepFMの仕組みを学ぶ
  * Two-Tower Modelsによる大規模推薦システムを構築する
  * Sequence-based推薦（RNN、Transformer）を実装する
  * 実践的な推薦システムパイプラインを設計する

## 4.1 Neural Collaborative Filtering (NCF)

### 4.1.1 NCFの動機

**従来の協調フィルタリングの限界：**

  * Matrix Factorization: 線形な内積のみで相互作用を表現
  * $\hat{r}_{ui} = p_u^T q_i$ （ユーザー埋め込み × アイテム埋め込み）
  * 問題：非線形な嗜好パターンを捉えられない

**NCFの提案：**

  * Multi-Layer Perceptron (MLP) で非線形性を導入
  * 論文: He et al. "Neural Collaborative Filtering" (WWW 2017)
  * 引用数: 4,000+（2024年時点）

### 4.1.2 NCFアーキテクチャ

**構成要素：**
    
    
    Input Layer
    ├─ User ID → User Embedding (dim=K)
    └─ Item ID → Item Embedding (dim=K)
        ↓
    Interaction Layer (複数の選択肢)
    ├─ (1) GMF: p_u ⊙ q_i (要素積)
    ├─ (2) MLP: f(concat(p_u, q_i))
    └─ (3) NeuMF: GMF + MLP の融合
        ↓
    Output Layer: σ(weighted sum) → 予測評価値
    

**数式表現：**

**GMF (Generalized Matrix Factorization):**

$$ \hat{y}_{ui}^{GMF} = \sigma(h^T (p_u \odot q_i)) $$ 

**MLP:**

$$ \begin{aligned} z_1 &= \phi_1(p_u, q_i) = \text{concat}(p_u, q_i) \\\ z_2 &= \sigma(W_2^T z_1 + b_2) \\\ &\vdots \\\ z_L &= \sigma(W_L^T z_{L-1} + b_L) \\\ \hat{y}_{ui}^{MLP} &= \sigma(h^T z_L) \end{aligned} $$ 

**NeuMF (Neural Matrix Factorization):**

$$ \hat{y}_{ui} = \sigma(h^T [\text{GMF output} \,||\, \text{MLP output}]) $$ 

### 4.1.3 PyTorch実装
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import pandas as pd
    
    class NCFDataset(Dataset):
        """NCF用のデータセット"""
        def __init__(self, user_ids, item_ids, ratings):
            self.user_ids = torch.LongTensor(user_ids)
            self.item_ids = torch.LongTensor(item_ids)
            self.ratings = torch.FloatTensor(ratings)
    
        def __len__(self):
            return len(self.user_ids)
    
        def __getitem__(self, idx):
            return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]
    
    
    class GMF(nn.Module):
        """Generalized Matrix Factorization"""
        def __init__(self, n_users, n_items, embedding_dim=32):
            super(GMF, self).__init__()
    
            # Embeddings
            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_embedding = nn.Embedding(n_items, embedding_dim)
    
            # Output layer
            self.fc = nn.Linear(embedding_dim, 1)
    
            # Initialize weights
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.item_embedding.weight, std=0.01)
    
        def forward(self, user_ids, item_ids):
            # Get embeddings
            user_emb = self.user_embedding(user_ids)  # (batch, dim)
            item_emb = self.item_embedding(item_ids)  # (batch, dim)
    
            # Element-wise product
            element_product = user_emb * item_emb  # (batch, dim)
    
            # Output
            output = self.fc(element_product)  # (batch, 1)
            return torch.sigmoid(output.squeeze())
    
    
    class MLPModel(nn.Module):
        """MLP-based Collaborative Filtering"""
        def __init__(self, n_users, n_items, embedding_dim=32, layers=[64, 32, 16]):
            super(MLPModel, self).__init__()
    
            # Embeddings
            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_embedding = nn.Embedding(n_items, embedding_dim)
    
            # MLP layers
            mlp_modules = []
            input_size = embedding_dim * 2
            for layer_size in layers:
                mlp_modules.append(nn.Linear(input_size, layer_size))
                mlp_modules.append(nn.ReLU())
                mlp_modules.append(nn.Dropout(0.2))
                input_size = layer_size
    
            self.mlp = nn.Sequential(*mlp_modules)
            self.fc = nn.Linear(layers[-1], 1)
    
            # Initialize weights
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.item_embedding.weight, std=0.01)
    
        def forward(self, user_ids, item_ids):
            # Get embeddings
            user_emb = self.user_embedding(user_ids)
            item_emb = self.item_embedding(item_ids)
    
            # Concatenate
            concat = torch.cat([user_emb, item_emb], dim=-1)  # (batch, dim*2)
    
            # MLP
            mlp_output = self.mlp(concat)
            output = self.fc(mlp_output)
            return torch.sigmoid(output.squeeze())
    
    
    class NeuMF(nn.Module):
        """Neural Matrix Factorization (GMF + MLP)"""
        def __init__(self, n_users, n_items,
                     gmf_dim=32, mlp_dim=32, layers=[64, 32, 16]):
            super(NeuMF, self).__init__()
    
            # GMF Embeddings
            self.gmf_user_embedding = nn.Embedding(n_users, gmf_dim)
            self.gmf_item_embedding = nn.Embedding(n_items, gmf_dim)
    
            # MLP Embeddings
            self.mlp_user_embedding = nn.Embedding(n_users, mlp_dim)
            self.mlp_item_embedding = nn.Embedding(n_items, mlp_dim)
    
            # MLP layers
            mlp_modules = []
            input_size = mlp_dim * 2
            for layer_size in layers:
                mlp_modules.append(nn.Linear(input_size, layer_size))
                mlp_modules.append(nn.ReLU())
                mlp_modules.append(nn.Dropout(0.2))
                input_size = layer_size
    
            self.mlp = nn.Sequential(*mlp_modules)
    
            # Final prediction layer
            self.fc = nn.Linear(gmf_dim + layers[-1], 1)
    
            # Initialize
            self._init_weights()
    
        def _init_weights(self):
            nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
            nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
            nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
            nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
    
        def forward(self, user_ids, item_ids):
            # GMF part
            gmf_user_emb = self.gmf_user_embedding(user_ids)
            gmf_item_emb = self.gmf_item_embedding(item_ids)
            gmf_output = gmf_user_emb * gmf_item_emb  # (batch, gmf_dim)
    
            # MLP part
            mlp_user_emb = self.mlp_user_embedding(user_ids)
            mlp_item_emb = self.mlp_item_embedding(item_ids)
            mlp_concat = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
            mlp_output = self.mlp(mlp_concat)  # (batch, layers[-1])
    
            # Concatenate GMF and MLP
            concat = torch.cat([gmf_output, mlp_output], dim=-1)
            output = self.fc(concat)
            return torch.sigmoid(output.squeeze())
    
    
    def train_ncf(model, train_loader, n_epochs=10, lr=0.001):
        """NCFモデルの訓練"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
    
            for user_ids, item_ids, ratings in train_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
    
                # Forward pass
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
        return model
    
    
    # 使用例
    if __name__ == "__main__":
        # サンプルデータ生成
        np.random.seed(42)
        n_users = 1000
        n_items = 500
        n_samples = 10000
    
        user_ids = np.random.randint(0, n_users, n_samples)
        item_ids = np.random.randint(0, n_items, n_samples)
        ratings = np.random.randint(0, 2, n_samples)  # Binary ratings
    
        # Dataset and DataLoader
        dataset = NCFDataset(user_ids, item_ids, ratings)
        train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
        # NeuMFモデルの訓練
        model = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32, layers=[64, 32, 16])
        trained_model = train_ncf(model, train_loader, n_epochs=5, lr=0.001)
    
        print("\nNeuMFモデルの訓練完了")
    

### 4.1.4 NCFの性能

**実験結果（MovieLens-1M）：**

モデル | HR@10 | NDCG@10 | 訓練時間  
---|---|---|---  
Matrix Factorization | 0.692 | 0.425 | 5分  
GMF | 0.704 | 0.432 | 8分  
MLP | 0.718 | 0.445 | 12分  
NeuMF | 0.726 | 0.463 | 15分  
  
**結論：** NeuMFが最高性能、MLPとGMFの組み合わせが効果的

* * *

## 4.2 Factorization Machines

### 4.2.1 FMの動機

**問題設定：**

  * スパースな特徴量（categorical features）の処理
  * 例：ユーザーID、アイテムID、カテゴリ、時刻、デバイス
  * One-hot encodingで数万〜数百万次元

**従来手法の限界：**

  * 線形回帰：特徴量間の相互作用を捉えられない
  * 多項式回帰：パラメータ数が爆発（$O(n^2)$）
  * SVMなど：計算コスト高、スパースデータで性能低下

**FMの提案：**

  * 論文: Rendle "Factorization Machines" (ICDM 2010)
  * 特徴量間の2次相互作用を効率的にモデル化
  * 時間計算量：$O(kn)$（kは埋め込み次元、nは特徴量数）

### 4.2.2 FM数式

**予測式：**

$$ \hat{y}(x) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j $$ 

ここで：

  * $w_0$: バイアス項
  * $w_i$: 1次係数（線形項）
  * $\langle v_i, v_j \rangle = \sum_{f=1}^k v_{i,f} \cdot v_{j,f}$: 2次係数（相互作用項）
  * $v_i \in \mathbb{R}^k$: 特徴量$i$の埋め込みベクトル（$k$次元）

**計算量削減のトリック：**

$$ \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^n v_{i,f} x_i \right)^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2 \right] $$ 

これにより計算量が $O(n^2) \to O(kn)$ に削減

### 4.2.3 Field-aware FM (FFM)

**FMの拡張：**

  * 各特徴量が複数のフィールド（field）に分類される
  * 例：User field（user_id, age, gender）、Item field（item_id, category）
  * 異なるfieldペアで異なる埋め込みを使用

**FFM予測式：**

$$ \hat{y}(x) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_{i,f_j}, v_{j,f_i} \rangle x_i x_j $$ 

  * $v_{i,f_j}$: 特徴量$i$がfield $f_j$と相互作用するときの埋め込み
  * パラメータ数：$nfk$（$f$はフィールド数）

### 4.2.4 DeepFM

**FM + Deep Neural Networkの融合：**

  * 論文: Guo et al. "DeepFM" (IJCAI 2017)
  * 低次相互作用（FM）+ 高次相互作用（DNN）

**アーキテクチャ：**
    
    
    Input: Sparse Features (one-hot)
        ↓
    Embedding Layer (共有)
        ├─ FM Component
        │   └─ 1次項 + 2次項（相互作用）
        └─ Deep Component
            └─ MLP (複数層、ReLU)
        ↓
    Output: σ(FM output + Deep output)
    

**数式：**

$$ \hat{y} = \sigma(y_{FM} + y_{DNN}) $$ 

### 4.2.5 DeepFM実装
    
    
    import torch
    import torch.nn as nn
    
    class FMLayer(nn.Module):
        """Factorization Machine Layer"""
        def __init__(self, n_features, embedding_dim=10):
            super(FMLayer, self).__init__()
    
            # 1次項
            self.linear = nn.Embedding(n_features, 1)
    
            # 2次項（埋め込み）
            self.embedding = nn.Embedding(n_features, embedding_dim)
    
            # Initialize
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.xavier_uniform_(self.embedding.weight)
    
        def forward(self, x):
            """
            Args:
                x: (batch, n_fields) - feature indices
            Returns:
                fm_output: (batch,) - FM prediction
            """
            # 1次項
            linear_part = self.linear(x).sum(dim=1).squeeze()  # (batch,)
    
            # 2次項
            embeddings = self.embedding(x)  # (batch, n_fields, dim)
    
            # (sum of squares) - (square of sum)
            square_of_sum = torch.sum(embeddings, dim=1) ** 2  # (batch, dim)
            sum_of_square = torch.sum(embeddings ** 2, dim=1)  # (batch, dim)
    
            interaction = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)  # (batch,)
    
            return linear_part + interaction
    
    
    class DeepFM(nn.Module):
        """DeepFM: FM + Deep Neural Network"""
        def __init__(self, n_features, n_fields, embedding_dim=10,
                     deep_layers=[256, 128, 64]):
            super(DeepFM, self).__init__()
    
            self.n_fields = n_fields
    
            # 共有Embedding layer
            self.embedding = nn.Embedding(n_features, embedding_dim)
    
            # FM component
            self.linear = nn.Embedding(n_features, 1)
            self.bias = nn.Parameter(torch.zeros(1))
    
            # Deep component
            deep_input_dim = n_fields * embedding_dim
            deep_modules = []
    
            for i, layer_size in enumerate(deep_layers):
                if i == 0:
                    deep_modules.append(nn.Linear(deep_input_dim, layer_size))
                else:
                    deep_modules.append(nn.Linear(deep_layers[i-1], layer_size))
                deep_modules.append(nn.BatchNorm1d(layer_size))
                deep_modules.append(nn.ReLU())
                deep_modules.append(nn.Dropout(0.5))
    
            self.deep = nn.Sequential(*deep_modules)
            self.deep_fc = nn.Linear(deep_layers[-1], 1)
    
            # Initialize
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.linear.weight)
    
        def forward(self, x):
            """
            Args:
                x: (batch, n_fields) - feature indices
            Returns:
                output: (batch,) - prediction
            """
            # Embeddings
            embeddings = self.embedding(x)  # (batch, n_fields, dim)
    
            # FM part
            # 1次項
            linear_part = self.linear(x).sum(dim=1).squeeze() + self.bias
    
            # 2次項
            square_of_sum = torch.sum(embeddings, dim=1) ** 2
            sum_of_square = torch.sum(embeddings ** 2, dim=1)
            fm_interaction = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
    
            fm_output = linear_part + fm_interaction
    
            # Deep part
            deep_input = embeddings.view(-1, self.n_fields * embeddings.size(2))
            deep_output = self.deep(deep_input)
            deep_output = self.deep_fc(deep_output).squeeze()
    
            # Combine
            output = torch.sigmoid(fm_output + deep_output)
            return output
    
    
    # 使用例
    if __name__ == "__main__":
        # サンプルデータ
        batch_size = 128
        n_features = 10000  # Total unique features
        n_fields = 20       # Number of feature fields
    
        # ランダムな特徴インデックス
        x = torch.randint(0, n_features, (batch_size, n_fields))
    
        # モデル
        model = DeepFM(n_features, n_fields, embedding_dim=10,
                       deep_layers=[256, 128, 64])
    
        # Forward pass
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample outputs: {output[:5]}")
    

* * *

## 4.3 Two-Tower Models

### 4.3.1 Two-Towerアーキテクチャ

**動機：**

  * 大規模推薦システム（数億ユーザー × 数千万アイテム）
  * リアルタイム推論の必要性
  * User TowerとItem Towerを分離してスケーラブルに

**アーキテクチャ：**
    
    
    User Features          Item Features
        ↓                      ↓
    User Tower            Item Tower
    (DNN)                 (DNN)
        ↓                      ↓
    User Embedding        Item Embedding
    (dim=128)             (dim=128)
        ↓                      ↓
        └──────── dot product ────────┘
                    ↓
              Similarity Score
    

**特徴：**

  * User/Item埋め込みを事前計算可能
  * 推論時：最近傍探索（ANN: Approximate Nearest Neighbor）
  * 計算量：$O(\log N)$ with ANN index (FAISS, ScaNN)

### 4.3.2 Contrastive Learning

**訓練目標：**

  * Positive pair（ユーザーが実際に閲覧したアイテム）の類似度を最大化
  * Negative pair（ランダムサンプル）の類似度を最小化

**損失関数（Triplet Loss）：**

$$ L = \sum_{(u,i^+,i^-)} \max(0, \alpha - \text{sim}(u, i^+) + \text{sim}(u, i^-)) $$ 

または**InfoNCE Loss** :

$$ L = -\log \frac{\exp(\text{sim}(u, i^+) / \tau)}{\sum_{j \in \mathcal{N}} \exp(\text{sim}(u, i_j) / \tau)} $$ 

  * $\tau$: temperature parameter
  * $\mathcal{N}$: negative samples

### 4.3.3 Two-Tower実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class TowerNetwork(nn.Module):
        """Single Tower (User or Item)"""
        def __init__(self, input_dim, embedding_dim=128, hidden_dims=[256, 128]):
            super(TowerNetwork, self).__init__()
    
            layers = []
            prev_dim = input_dim
    
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                prev_dim = hidden_dim
    
            # Final embedding layer
            layers.append(nn.Linear(prev_dim, embedding_dim))
    
            self.network = nn.Sequential(*layers)
    
        def forward(self, x):
            embedding = self.network(x)
            # L2 normalization
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
    
    
    class TwoTowerModel(nn.Module):
        """Two-Tower Recommendation Model"""
        def __init__(self, user_feature_dim, item_feature_dim,
                     embedding_dim=128, hidden_dims=[256, 128]):
            super(TwoTowerModel, self).__init__()
    
            self.user_tower = TowerNetwork(user_feature_dim, embedding_dim, hidden_dims)
            self.item_tower = TowerNetwork(item_feature_dim, embedding_dim, hidden_dims)
    
            self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
        def forward(self, user_features, item_features):
            """
            Args:
                user_features: (batch, user_dim)
                item_features: (batch, item_dim)
            Returns:
                similarity: (batch,)
            """
            user_emb = self.user_tower(user_features)  # (batch, emb_dim)
            item_emb = self.item_tower(item_features)  # (batch, emb_dim)
    
            # Cosine similarity (already L2 normalized)
            similarity = (user_emb * item_emb).sum(dim=1)
            return similarity
    
        def get_user_embedding(self, user_features):
            """Get user embedding for indexing"""
            return self.user_tower(user_features)
    
        def get_item_embedding(self, item_features):
            """Get item embedding for indexing"""
            return self.item_tower(item_features)
    
    
    class InfoNCELoss(nn.Module):
        """InfoNCE Loss for contrastive learning"""
        def __init__(self, temperature=0.07):
            super(InfoNCELoss, self).__init__()
            self.temperature = temperature
    
        def forward(self, user_emb, pos_item_emb, neg_item_embs):
            """
            Args:
                user_emb: (batch, dim)
                pos_item_emb: (batch, dim) - positive items
                neg_item_embs: (batch, n_neg, dim) - negative items
            Returns:
                loss: scalar
            """
            # Positive similarity
            pos_sim = (user_emb * pos_item_emb).sum(dim=1) / self.temperature  # (batch,)
    
            # Negative similarities
            neg_sim = torch.bmm(neg_item_embs, user_emb.unsqueeze(2)).squeeze(2)  # (batch, n_neg)
            neg_sim = neg_sim / self.temperature
    
            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1+n_neg)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
            loss = F.cross_entropy(logits, labels)
            return loss
    
    
    def train_two_tower(model, train_loader, n_epochs=10, lr=0.001, n_negatives=5):
        """Two-Towerモデルの訓練"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = InfoNCELoss(temperature=0.07)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
    
            for batch in train_loader:
                user_feat = batch['user_features'].to(device)
                pos_item_feat = batch['pos_item_features'].to(device)
                neg_item_feats = batch['neg_item_features'].to(device)  # (batch, n_neg, dim)
    
                # Get embeddings
                user_emb = model.get_user_embedding(user_feat)
                pos_item_emb = model.get_item_embedding(pos_item_feat)
    
                # Negative embeddings
                batch_size, n_neg, feat_dim = neg_item_feats.shape
                neg_item_feats_flat = neg_item_feats.view(-1, feat_dim)
                neg_item_embs = model.get_item_embedding(neg_item_feats_flat)
                neg_item_embs = neg_item_embs.view(batch_size, n_neg, -1)
    
                # Compute loss
                loss = criterion(user_emb, pos_item_emb, neg_item_embs)
    
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
        return model
    
    
    # 使用例
    if __name__ == "__main__":
        # モデル定義
        user_feature_dim = 50   # User features (age, gender, history, etc.)
        item_feature_dim = 100  # Item features (category, price, brand, etc.)
    
        model = TwoTowerModel(user_feature_dim, item_feature_dim,
                              embedding_dim=128, hidden_dims=[256, 128])
    
        # サンプル入力
        batch_size = 64
        user_features = torch.randn(batch_size, user_feature_dim)
        item_features = torch.randn(batch_size, item_feature_dim)
    
        # Forward
        similarity = model(user_features, item_features)
        print(f"Similarity scores: {similarity[:5]}")
    
        # Get embeddings for indexing
        user_emb = model.get_user_embedding(user_features)
        item_emb = model.get_item_embedding(item_features)
        print(f"User embedding shape: {user_emb.shape}")
        print(f"Item embedding shape: {item_emb.shape}")
    

### 4.3.4 効率的な推論（ANN）

**FAISS (Facebook AI Similarity Search) の利用：**
    
    
    import faiss
    import numpy as np
    
    def build_item_index(item_embeddings, use_gpu=False):
        """
        Build FAISS index for fast retrieval
    
        Args:
            item_embeddings: (n_items, embedding_dim) numpy array
            use_gpu: whether to use GPU for indexing
        Returns:
            index: FAISS index
        """
        n_items, dim = item_embeddings.shape
    
        # Normalize embeddings
        faiss.normalize_L2(item_embeddings)
    
        # Build index (Inner Product = Cosine Similarity for normalized vectors)
        index = faiss.IndexFlatIP(dim)
    
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    
        index.add(item_embeddings)
    
        print(f"Built FAISS index with {index.ntotal} items")
        return index
    
    
    def retrieve_top_k(user_embedding, item_index, k=10):
        """
        Retrieve top-k items for a user
    
        Args:
            user_embedding: (embedding_dim,) or (1, embedding_dim)
            item_index: FAISS index
            k: number of items to retrieve
        Returns:
            scores: (k,) similarity scores
            indices: (k,) item indices
        """
        if len(user_embedding.shape) == 1:
            user_embedding = user_embedding.reshape(1, -1)
    
        # Normalize
        faiss.normalize_L2(user_embedding)
    
        # Search
        scores, indices = item_index.search(user_embedding, k)
    
        return scores[0], indices[0]
    
    
    # 使用例
    if __name__ == "__main__":
        # 仮のアイテム埋め込み（100万アイテム）
        n_items = 1_000_000
        embedding_dim = 128
    
        item_embeddings = np.random.randn(n_items, embedding_dim).astype('float32')
    
        # Build index
        index = build_item_index(item_embeddings, use_gpu=False)
    
        # User embedding
        user_emb = np.random.randn(embedding_dim).astype('float32')
    
        # Retrieve top-10
        scores, indices = retrieve_top_k(user_emb, index, k=10)
    
        print(f"Top-10 item indices: {indices}")
        print(f"Top-10 scores: {scores}")
    

* * *

## 4.4 Sequence-based推薦

### 4.4.1 Session-based Recommendation

**問題設定：**

  * ユーザーの行動履歴（session）から次のアクションを予測
  * 例：E-commerce（閲覧履歴 → 次に閲覧/購入する商品）
  * 匿名ユーザーでも適用可能（user IDなし）

**データ形式：**
    
    
    Session 1: [item_5, item_12, item_3, item_8] → item_15
    Session 2: [item_1, item_6] → item_9
    Session 3: [item_12, item_15, item_2] → item_7
    

### 4.4.2 RNN/GRU for Sequential Recommendation

**GRU4Rec (Session-based Recommendations with RNNs):**

  * 論文: Hidasi et al. (ICLR 2016)
  * GRU (Gated Recurrent Unit) でセッションをモデル化

**アーキテクチャ：**
    
    
    Input: item sequence [i_1, i_2, ..., i_t]
        ↓
    Embedding Layer: [e_1, e_2, ..., e_t]
        ↓
    GRU Layer: h_t = GRU(e_t, h_{t-1})
        ↓
    Output Layer: softmax(W h_t + b) → P(next item)
    

### 4.4.3 GRU4Rec実装
    
    
    import torch
    import torch.nn as nn
    
    class GRU4Rec(nn.Module):
        """GRU-based Session Recommendation"""
        def __init__(self, n_items, embedding_dim=100, hidden_dim=100, n_layers=1):
            super(GRU4Rec, self).__init__()
    
            self.n_items = n_items
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
    
            # Embedding layer
            self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
    
            # GRU layer
            self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=0.2 if n_layers > 1 else 0)
    
            # Output layer
            self.fc = nn.Linear(hidden_dim, n_items)
    
            # Initialize
            nn.init.xavier_uniform_(self.item_embedding.weight)
            nn.init.xavier_uniform_(self.fc.weight)
    
        def forward(self, item_seq, lengths=None):
            """
            Args:
                item_seq: (batch, seq_len) - item ID sequence
                lengths: (batch,) - actual sequence lengths
            Returns:
                logits: (batch, n_items) - prediction for next item
            """
            # Embedding
            emb = self.item_embedding(item_seq)  # (batch, seq_len, emb_dim)
    
            # GRU
            if lengths is not None:
                # Pack sequence for efficiency
                packed = nn.utils.rnn.pack_padded_sequence(
                    emb, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                gru_out, hidden = self.gru(packed)
                gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
            else:
                gru_out, hidden = self.gru(emb)  # (batch, seq_len, hidden)
    
            # Get last hidden state
            if lengths is not None:
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, gru_out.size(2))
                last_hidden = gru_out.gather(1, idx).squeeze(1)  # (batch, hidden)
            else:
                last_hidden = gru_out[:, -1, :]  # (batch, hidden)
    
            # Output
            logits = self.fc(last_hidden)  # (batch, n_items)
            return logits
    
    
    def train_gru4rec(model, train_loader, n_epochs=10, lr=0.001):
        """GRU4Recの訓練"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
    
            for batch in train_loader:
                item_seq = batch['item_seq'].to(device)  # (batch, seq_len)
                target = batch['target'].to(device)      # (batch,)
                lengths = batch['lengths'].to(device)    # (batch,)
    
                # Forward
                logits = model(item_seq, lengths)
                loss = criterion(logits, target)
    
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
        return model
    
    
    # 使用例
    if __name__ == "__main__":
        n_items = 10000
        model = GRU4Rec(n_items, embedding_dim=100, hidden_dim=100, n_layers=2)
    
        # サンプル入力
        batch_size = 32
        seq_len = 10
        item_seq = torch.randint(1, n_items, (batch_size, seq_len))
        lengths = torch.randint(5, seq_len+1, (batch_size,))
    
        logits = model(item_seq, lengths)
        print(f"Input shape: {item_seq.shape}")
        print(f"Output shape: {logits.shape}")
    
        # Top-k prediction
        k = 10
        _, top_k_items = torch.topk(logits, k, dim=1)
        print(f"Top-{k} predicted items (first sample): {top_k_items[0]}")
    

### 4.4.4 Self-Attention for Sequential Recommendation (SASRec)

**Self-Attentionの利点：**

  * RNN/GRUの逐次処理の制約を克服
  * 並列計算可能、長距離依存関係を捉えやすい
  * 論文: Kang & McAuley "SASRec" (ICDM 2018)

**アーキテクチャ：**
    
    
    Input: [i_1, i_2, ..., i_t]
        ↓
    Embedding + Positional Encoding
        ↓
    Self-Attention Blocks (×L layers)
    ├─ Multi-Head Self-Attention
    ├─ Feed-Forward Network
    └─ Layer Normalization + Residual
        ↓
    Output Layer: Predict i_{t+1}
    

### 4.4.5 SASRec実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    
    class SelfAttention(nn.Module):
        """Multi-Head Self-Attention"""
        def __init__(self, hidden_dim, n_heads, dropout=0.1):
            super(SelfAttention, self).__init__()
    
            assert hidden_dim % n_heads == 0
    
            self.hidden_dim = hidden_dim
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
    
            self.q_linear = nn.Linear(hidden_dim, hidden_dim)
            self.k_linear = nn.Linear(hidden_dim, hidden_dim)
            self.v_linear = nn.Linear(hidden_dim, hidden_dim)
    
            self.out_linear = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x, mask=None):
            """
            Args:
                x: (batch, seq_len, hidden_dim)
                mask: (batch, seq_len, seq_len) - causal mask
            Returns:
                out: (batch, seq_len, hidden_dim)
            """
            batch_size, seq_len, _ = x.shape
    
            # Linear projections
            Q = self.q_linear(x)  # (batch, seq_len, hidden)
            K = self.k_linear(x)
            V = self.v_linear(x)
    
            # Reshape for multi-head
            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            # (batch, n_heads, seq_len, head_dim)
    
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # (batch, n_heads, seq_len, seq_len)
    
            # Apply mask (causal)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
    
            # Softmax
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
    
            # Apply attention to values
            out = torch.matmul(attn, V)  # (batch, n_heads, seq_len, head_dim)
    
            # Reshape back
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
    
            # Final linear
            out = self.out_linear(out)
            return out
    
    
    class FeedForward(nn.Module):
        """Position-wise Feed-Forward Network"""
        def __init__(self, hidden_dim, ff_dim, dropout=0.1):
            super(FeedForward, self).__init__()
    
            self.linear1 = nn.Linear(hidden_dim, ff_dim)
            self.linear2 = nn.Linear(ff_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
    
    class SASRecBlock(nn.Module):
        """Single SASRec Transformer Block"""
        def __init__(self, hidden_dim, n_heads, ff_dim, dropout=0.1):
            super(SASRecBlock, self).__init__()
    
            self.attn = SelfAttention(hidden_dim, n_heads, dropout)
            self.ff = FeedForward(hidden_dim, ff_dim, dropout)
    
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x, mask):
            # Self-attention + residual
            attn_out = self.attn(self.ln1(x), mask)
            x = x + self.dropout(attn_out)
    
            # Feed-forward + residual
            ff_out = self.ff(self.ln2(x))
            x = x + self.dropout(ff_out)
    
            return x
    
    
    class SASRec(nn.Module):
        """Self-Attentive Sequential Recommendation"""
        def __init__(self, n_items, max_len=50, hidden_dim=100,
                     n_heads=2, n_blocks=2, dropout=0.1):
            super(SASRec, self).__init__()
    
            self.n_items = n_items
            self.max_len = max_len
    
            # Embeddings
            self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
            self.pos_embedding = nn.Embedding(max_len, hidden_dim)
    
            # Transformer blocks
            self.blocks = nn.ModuleList([
                SASRecBlock(hidden_dim, n_heads, hidden_dim * 4, dropout)
                for _ in range(n_blocks)
            ])
    
            self.ln = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
    
            # Initialize
            nn.init.xavier_uniform_(self.item_embedding.weight)
            nn.init.xavier_uniform_(self.pos_embedding.weight)
    
        def forward(self, item_seq):
            """
            Args:
                item_seq: (batch, seq_len)
            Returns:
                logits: (batch, seq_len, n_items)
            """
            batch_size, seq_len = item_seq.shape
    
            # Item embeddings
            item_emb = self.item_embedding(item_seq)  # (batch, seq_len, hidden)
    
            # Positional embeddings
            positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)  # (1, seq_len, hidden)
    
            # Combine
            x = item_emb + pos_emb
            x = self.dropout(x)
    
            # Causal mask (prevent looking at future items)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=item_seq.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
            # Transformer blocks
            for block in self.blocks:
                x = block(x, mask)
    
            x = self.ln(x)
    
            # Predict next item (using item embeddings as output weights)
            logits = torch.matmul(x, self.item_embedding.weight.T)  # (batch, seq_len, n_items+1)
    
            return logits[:, :, 1:]  # Remove padding index
    
    
    # 使用例
    if __name__ == "__main__":
        n_items = 10000
        max_len = 50
    
        model = SASRec(n_items, max_len=max_len, hidden_dim=128,
                       n_heads=4, n_blocks=2, dropout=0.2)
    
        # サンプル入力
        batch_size = 16
        seq_len = 20
        item_seq = torch.randint(1, n_items, (batch_size, seq_len))
    
        logits = model(item_seq)
        print(f"Input shape: {item_seq.shape}")
        print(f"Output shape: {logits.shape}")
    
        # Predict next item (using last position)
        last_logits = logits[:, -1, :]  # (batch, n_items)
        _, top_k = torch.topk(last_logits, 10, dim=1)
        print(f"Top-10 predictions: {top_k[0]}")
    

### 4.4.6 BERT4Rec

**BERT for Sequential Recommendation:**

  * 論文: Sun et al. "BERT4Rec" (CIKM 2019)
  * 双方向Transformer（過去+未来の文脈を利用）
  * Masked Item Prediction (MLM風)

**訓練方法：**

  * 入力シーケンスの一部をマスク（[MASK]トークン）
  * マスクされたアイテムを予測
  * 双方向の文脈を利用可能

* * *

## 4.5 実践的な推薦システムパイプライン

### 4.5.1 End-to-Endパイプライン

**推薦システムの2段階アーキテクチャ：**
    
    
    Stage 1: Candidate Generation（候補生成）
    ├─ Input: User context
    ├─ Method: Two-Tower, Matrix Factorization, Graph-based
    ├─ Output: 数百〜数千の候補アイテム
    └─ 目的: Recall最大化、高速
    
    Stage 2: Ranking（ランキング）
    ├─ Input: User + 候補アイテム
    ├─ Method: DeepFM, Wide&Deep, DNN
    ├─ Output: Top-K推薦リスト
    └─ 目的: Precision最大化、精度重視
    

### 4.5.2 Candidate Generation例
    
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class CandidateGenerator:
        """候補生成モジュール"""
        def __init__(self, user_embeddings, item_embeddings):
            """
            Args:
                user_embeddings: (n_users, dim)
                item_embeddings: (n_items, dim)
            """
            self.user_embeddings = user_embeddings
            self.item_embeddings = item_embeddings
    
            # Build FAISS index for fast retrieval
            import faiss
            dim = item_embeddings.shape[1]
    
            # Normalize for cosine similarity
            faiss.normalize_L2(item_embeddings)
    
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(item_embeddings.astype('float32'))
    
        def generate_candidates(self, user_id, n_candidates=500, exclude_items=None):
            """
            Generate candidate items for a user
    
            Args:
                user_id: int
                n_candidates: number of candidates to retrieve
                exclude_items: set of item IDs to exclude
            Returns:
                candidate_items: list of item IDs
                scores: list of similarity scores
            """
            # Get user embedding
            user_emb = self.user_embeddings[user_id:user_id+1].astype('float32')
    
            # Normalize
            import faiss
            faiss.normalize_L2(user_emb)
    
            # Retrieve top candidates
            scores, indices = self.index.search(user_emb, n_candidates * 2)
            scores = scores[0]
            indices = indices[0]
    
            # Filter out excluded items
            if exclude_items is not None:
                mask = ~np.isin(indices, list(exclude_items))
                indices = indices[mask]
                scores = scores[mask]
    
            # Return top n_candidates
            return indices[:n_candidates].tolist(), scores[:n_candidates].tolist()
    
    
    class Ranker:
        """ランキングモジュール（DeepFMなど）"""
        def __init__(self, ranking_model):
            self.model = ranking_model
    
        def rank(self, user_features, candidate_item_features, top_k=10):
            """
            Rank candidate items
    
            Args:
                user_features: user feature vector
                candidate_item_features: (n_candidates, item_dim)
                top_k: number of items to return
            Returns:
                ranked_items: list of item indices
                scores: list of ranking scores
            """
            import torch
    
            n_candidates = len(candidate_item_features)
    
            # Replicate user features
            user_feat_batch = np.tile(user_features, (n_candidates, 1))
    
            # Combine features
            features = np.concatenate([user_feat_batch, candidate_item_features], axis=1)
            features_tensor = torch.FloatTensor(features)
    
            # Predict scores
            with torch.no_grad():
                scores = self.model(features_tensor).numpy()
    
            # Sort by score
            sorted_indices = np.argsort(scores)[::-1]
    
            return sorted_indices[:top_k].tolist(), scores[sorted_indices[:top_k]].tolist()
    
    
    class RecommendationPipeline:
        """推薦パイプライン（候補生成 + ランキング）"""
        def __init__(self, candidate_generator, ranker):
            self.candidate_generator = candidate_generator
            self.ranker = ranker
    
        def recommend(self, user_id, user_features, item_features_db,
                      n_candidates=500, top_k=10, exclude_items=None):
            """
            Generate recommendations for a user
    
            Args:
                user_id: int
                user_features: user feature vector
                item_features_db: dict {item_id: features}
                n_candidates: number of candidates
                top_k: number of final recommendations
                exclude_items: set of item IDs to exclude
            Returns:
                recommendations: list of (item_id, score) tuples
            """
            # Stage 1: Candidate Generation
            candidate_ids, _ = self.candidate_generator.generate_candidates(
                user_id, n_candidates, exclude_items
            )
    
            # Get item features for candidates
            candidate_features = np.array([item_features_db[iid] for iid in candidate_ids])
    
            # Stage 2: Ranking
            ranked_indices, scores = self.ranker.rank(user_features, candidate_features, top_k)
    
            # Map back to item IDs
            recommendations = [(candidate_ids[idx], scores[i])
                              for i, idx in enumerate(ranked_indices)]
    
            return recommendations
    
    
    # 使用例
    if __name__ == "__main__":
        # 仮のデータ
        n_users = 10000
        n_items = 50000
        emb_dim = 128
    
        user_embeddings = np.random.randn(n_users, emb_dim).astype('float32')
        item_embeddings = np.random.randn(n_items, emb_dim).astype('float32')
    
        # Candidate Generator
        candidate_gen = CandidateGenerator(user_embeddings, item_embeddings)
    
        # 仮のランキングモデル（実際にはDeepFMなど）
        class DummyRanker:
            def __call__(self, x):
                return torch.rand(len(x))
    
        ranker = Ranker(DummyRanker())
    
        # Pipeline
        pipeline = RecommendationPipeline(candidate_gen, ranker)
    
        # Recommend
        user_id = 42
        user_feat = np.random.randn(50)
        item_feat_db = {i: np.random.randn(100) for i in range(n_items)}
    
        recommendations = pipeline.recommend(
            user_id, user_feat, item_feat_db,
            n_candidates=500, top_k=10, exclude_items={1, 5, 10}
        )
    
        print("Top-10 Recommendations:")
        for item_id, score in recommendations:
            print(f"  Item {item_id}: {score:.4f}")
    

### 4.5.3 A/Bテスト

**実験設計：**

  * Control Group: 既存の推薦アルゴリズム
  * Treatment Group: 新しいアルゴリズム（DeepFM、Two-Towerなど）
  * 評価指標：CTR、Conversion Rate、Revenue、User Engagement

**統計的有意性の検定：**

  * 帰無仮説：$H_0$: 新アルゴリズムの効果なし
  * 対立仮説：$H_1$: 新アルゴリズムの効果あり
  * 有意水準：$\alpha = 0.05$（5%）
  * 検定方法：t検定、Welch's t-test、Mann-Whitney U検定

### 4.5.4 Production Deployment

**システム要件：**

  * レイテンシ：< 100ms（推論時間）
  * スループット：> 10,000 QPS（Queries Per Second）
  * スケーラビリティ：水平スケール可能

**技術スタック例：**

  * モデルサーブ：TensorFlow Serving、TorchServe、NVIDIA Triton
  * キャッシュ：Redis（ユーザー埋め込み、アイテム埋め込み）
  * ANN検索：FAISS、ScaNN、Milvus
  * ログ収集：Kafka、Fluentd
  * モニタリング：Prometheus、Grafana

* * *

## まとめ

本章で学んだこと：

  1. **Neural Collaborative Filtering (NCF):**
     * GMF、MLP、NeuMFの3つのアプローチ
     * 非線形な相互作用をMLPで表現
     * MovieLensでHR@10 0.726達成
  2. **Factorization Machines:**
     * 2次相互作用を$O(kn)$で計算
     * FFMでフィールド別埋め込み
     * DeepFMでFM+DNNを融合
  3. **Two-Tower Models:**
     * User TowerとItem Towerを分離
     * FAISS/ScaNNで高速推論（$O(\log N)$）
     * InfoNCE Lossで対照学習
  4. **Sequence-based推薦:**
     * GRU4Rec: セッションベース推薦
     * SASRec: Self-Attentionで長距離依存
     * BERT4Rec: 双方向Transformerでマスク予測
  5. **実践的パイプライン:**
     * 候補生成（Recall重視）+ ランキング（Precision重視）
     * A/Bテストで効果検証
     * Production deploymentの技術スタック

* * *

## 演習問題

**問1:** NeuMFがMatrix Factorizationより高性能な理由を、線形性と非線形性の観点から説明せよ。

**問2:** Factorization Machinesの2次相互作用の計算量が$O(n^2)$から$O(kn)$に削減される仕組みを数式を用いて示せ。

**問3:** Two-Tower Modelで、User TowerとItem Towerを分離するメリットとデメリットを3つずつ挙げよ。

**問4:** SASRecとGRU4Recの違いを、並列化可能性と長距離依存の捉え方の2つの観点から論じよ。

**問5:** 推薦システムのCandidate Generation段階でRecall@500 = 0.8、Ranking段階でPrecision@10 = 0.3の場合、最終的なTop-10推薦の期待的中数を計算せよ。

**問6:** 大規模推薦システム（1億ユーザー × 1000万アイテム）でレイテンシ < 100msを達成するための技術的工夫を5つ提案せよ。

* * *

## 参考文献

  1. He, X. et al. "Neural Collaborative Filtering." _Proceedings of WWW_ (2017).
  2. Rendle, S. "Factorization Machines." _Proceedings of ICDM_ (2010).
  3. Juan, Y. et al. "Field-aware Factorization Machines for CTR Prediction." _RecSys_ (2016).
  4. Guo, H. et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." _IJCAI_ (2017).
  5. Yi, X. et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." _RecSys_ (2019). [Two-Tower]
  6. Hidasi, B. et al. "Session-based Recommendations with Recurrent Neural Networks." _ICLR_ (2016).
  7. Kang, W. & McAuley, J. "Self-Attentive Sequential Recommendation." _ICDM_ (2018).
  8. Sun, F. et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer." _CIKM_ (2019).
  9. Covington, P. et al. "Deep Neural Networks for YouTube Recommendations." _RecSys_ (2016).
  10. Cheng, H. et al. "Wide & Deep Learning for Recommender Systems." _DLRS Workshop_ (2016).

* * *
