---
title: "Chapter 4: Deep Learning-Based Recommendation Systems"
chapter_title: "Chapter 4: Deep Learning-Based Recommendation Systems"
subtitle: Neural Collaborative Filtering, Two-Tower Models, Sequence-based Recommendation
reading_time: 40-50 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 6
version: 1.0
---

This chapter covers Deep Learning. You will learn mechanisms of Factorization Machines, sequence-based recommendation systems (RNN, and Design practical recommendation system pipelines.

## Learning Objectives

  * Understand the theory and implementation of Neural Collaborative Filtering (NCF)
  * Learn the mechanisms of Factorization Machines and DeepFM
  * Build large-scale recommendation systems using Two-Tower Models
  * Implement sequence-based recommendation systems (RNN, Transformer)
  * Design practical recommendation system pipelines

## 4.1 Neural Collaborative Filtering (NCF)

### 4.1.1 Motivation for NCF

**Limitations of Traditional Collaborative Filtering:**

  * Matrix Factorization: Represents interactions only through linear inner products
  * $\hat{r}_{ui} = p_u^T q_i$ (User embedding × Item embedding)
  * Problem: Cannot capture non-linear preference patterns

**NCF Proposal:**

  * Introduce non-linearity with Multi-Layer Perceptron (MLP)
  * Paper: He et al. "Neural Collaborative Filtering" (WWW 2017)
  * Citations: 4,000+ (as of 2024)

### 4.1.2 NCF Architecture

**Components:**
    
    
    Input Layer
    ├─ User ID → User Embedding (dim=K)
    └─ Item ID → Item Embedding (dim=K)
        ↓
    Interaction Layer (multiple options)
    ├─ (1) GMF: p_u ⊙ q_i (element-wise product)
    ├─ (2) MLP: f(concat(p_u, q_i))
    └─ (3) NeuMF: Fusion of GMF + MLP
        ↓
    Output Layer: σ(weighted sum) → Predicted rating
    

**Mathematical Formulation:**

**GMF (Generalized Matrix Factorization):**

$$ \hat{y}_{ui}^{GMF} = \sigma(h^T (p_u \odot q_i)) $$ 

**MLP:**

$$ \begin{aligned} z_1 &= \phi_1(p_u, q_i) = \text{concat}(p_u, q_i) \\\ z_2 &= \sigma(W_2^T z_1 + b_2) \\\ &\vdots \\\ z_L &= \sigma(W_L^T z_{L-1} + b_L) \\\ \hat{y}_{ui}^{MLP} &= \sigma(h^T z_L) \end{aligned} $$ 

**NeuMF (Neural Matrix Factorization):**

$$ \hat{y}_{ui} = \sigma(h^T [\text{GMF output} \,||\, \text{MLP output}]) $$ 

### 4.1.3 PyTorch Implementation
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import pandas as pd
    
    class NCFDataset(Dataset):
        """Dataset for NCF"""
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
        """Train NCF model"""
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
    
    
    # Usage example
    if __name__ == "__main__":
        # Generate sample data
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
    
        # Train NeuMF model
        model = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32, layers=[64, 32, 16])
        trained_model = train_ncf(model, train_loader, n_epochs=5, lr=0.001)
    
        print("\nNeuMF model training complete")
    

### 4.1.4 NCF Performance

**Experimental Results (MovieLens-1M):**

Model | HR@10 | NDCG@10 | Training Time  
---|---|---|---  
Matrix Factorization | 0.692 | 0.425 | 5 min  
GMF | 0.704 | 0.432 | 8 min  
MLP | 0.718 | 0.445 | 12 min  
NeuMF | 0.726 | 0.463 | 15 min  
  
**Conclusion:** NeuMF achieves the best performance, combining MLP and GMF is effective

* * *

## 4.2 Factorization Machines

### 4.2.1 Motivation for FM

**Problem Setting:**

  * Processing sparse features (categorical features)
  * Examples: User ID, Item ID, Category, Time, Device
  * Tens of thousands to millions of dimensions with one-hot encoding

**Limitations of Traditional Methods:**

  * Linear regression: Cannot capture feature interactions
  * Polynomial regression: Parameter explosion ($O(n^2)$)
  * SVM etc.: High computational cost, poor performance on sparse data

**FM Proposal:**

  * Paper: Rendle "Factorization Machines" (ICDM 2010)
  * Efficiently model second-order feature interactions
  * Time complexity: $O(kn)$ (k is embedding dimension, n is number of features)

### 4.2.2 FM Formula

**Prediction Formula:**

$$ \hat{y}(x) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j $$ 

Where:

  * $w_0$: Bias term
  * $w_i$: First-order coefficients (linear terms)
  * $\langle v_i, v_j \rangle = \sum_{f=1}^k v_{i,f} \cdot v_{j,f}$: Second-order coefficients (interaction terms)
  * $v_i \in \mathbb{R}^k$: Embedding vector for feature $i$ ($k$ dimensions)

**Computational Complexity Reduction Trick:**

$$ \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^n v_{i,f} x_i \right)^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2 \right] $$ 

This reduces complexity from $O(n^2) \to O(kn)$

### 4.2.3 Field-aware FM (FFM)

**FM Extension:**

  * Each feature is classified into multiple fields
  * Example: User field (user_id, age, gender), Item field (item_id, category)
  * Use different embeddings for different field pairs

**FFM Prediction Formula:**

$$ \hat{y}(x) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_{i,f_j}, v_{j,f_i} \rangle x_i x_j $$ 

  * $v_{i,f_j}$: Embedding for feature $i$ when interacting with field $f_j$
  * Number of parameters: $nfk$ ($f$ is number of fields)

### 4.2.4 DeepFM

**Fusion of FM + Deep Neural Network:**

  * Paper: Guo et al. "DeepFM" (IJCAI 2017)
  * Low-order interactions (FM) + High-order interactions (DNN)

**Architecture:**
    
    
    Input: Sparse Features (one-hot)
        ↓
    Embedding Layer (shared)
        ├─ FM Component
        │   └─ First-order + Second-order (interactions)
        └─ Deep Component
            └─ MLP (multiple layers, ReLU)
        ↓
    Output: σ(FM output + Deep output)
    

**Formula:**

$$ \hat{y} = \sigma(y_{FM} + y_{DNN}) $$ 

### 4.2.5 DeepFM Implementation
    
    
    import torch
    import torch.nn as nn
    
    class FMLayer(nn.Module):
        """Factorization Machine Layer"""
        def __init__(self, n_features, embedding_dim=10):
            super(FMLayer, self).__init__()
    
            # First-order term
            self.linear = nn.Embedding(n_features, 1)
    
            # Second-order term (embeddings)
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
            # First-order term
            linear_part = self.linear(x).sum(dim=1).squeeze()  # (batch,)
    
            # Second-order term
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
    
            # Shared Embedding layer
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
            # First-order term
            linear_part = self.linear(x).sum(dim=1).squeeze() + self.bias
    
            # Second-order term
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
    
    
    # Usage example
    if __name__ == "__main__":
        # Sample data
        batch_size = 128
        n_features = 10000  # Total unique features
        n_fields = 20       # Number of feature fields
    
        # Random feature indices
        x = torch.randint(0, n_features, (batch_size, n_fields))
    
        # Model
        model = DeepFM(n_features, n_fields, embedding_dim=10,
                       deep_layers=[256, 128, 64])
    
        # Forward pass
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample outputs: {output[:5]}")
    

* * *

## 4.3 Two-Tower Models

### 4.3.1 Two-Tower Architecture

**Motivation:**

  * Large-scale recommendation systems (hundreds of millions of users × tens of millions of items)
  * Need for real-time inference
  * Scalability through separation of User Tower and Item Tower

**Architecture:**
    
    
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
    

**Features:**

  * User/Item embeddings can be pre-computed
  * Inference: Approximate Nearest Neighbor search (ANN)
  * Complexity: $O(\log N)$ with ANN index (FAISS, ScaNN)

### 4.3.2 Contrastive Learning

**Training Objective:**

  * Maximize similarity of positive pairs (items actually viewed by users)
  * Minimize similarity of negative pairs (random samples)

**Loss Function (Triplet Loss):**

$$ L = \sum_{(u,i^+,i^-)} \max(0, \alpha - \text{sim}(u, i^+) + \text{sim}(u, i^-)) $$ 

Or **InfoNCE Loss** :

$$ L = -\log \frac{\exp(\text{sim}(u, i^+) / \tau)}{\sum_{j \in \mathcal{N}} \exp(\text{sim}(u, i_j) / \tau)} $$ 

  * $\tau$: temperature parameter
  * $\mathcal{N}$: negative samples

### 4.3.3 Two-Tower Implementation
    
    
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
        """Train Two-Tower model"""
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
    
    
    # Usage example
    if __name__ == "__main__":
        # Model definition
        user_feature_dim = 50   # User features (age, gender, history, etc.)
        item_feature_dim = 100  # Item features (category, price, brand, etc.)
    
        model = TwoTowerModel(user_feature_dim, item_feature_dim,
                              embedding_dim=128, hidden_dims=[256, 128])
    
        # Sample input
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
    

### 4.3.4 Efficient Inference (ANN)

**Using FAISS (Facebook AI Similarity Search):**
    
    
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
    
    
    # Usage example
    if __name__ == "__main__":
        # Sample item embeddings (1 million items)
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

## 4.4 Sequence-based Recommendation

### 4.4.1 Session-based Recommendation

**Problem Setting:**

  * Predict the next action from user behavior history (session)
  * Example: E-commerce (browsing history → next item to view/purchase)
  * Applicable to anonymous users (no user ID)

**Data Format:**
    
    
    Session 1: [item_5, item_12, item_3, item_8] → item_15
    Session 2: [item_1, item_6] → item_9
    Session 3: [item_12, item_15, item_2] → item_7
    

### 4.4.2 RNN/GRU for Sequential Recommendation

**GRU4Rec (Session-based Recommendations with RNNs):**

  * Paper: Hidasi et al. (ICLR 2016)
  * Model sessions with GRU (Gated Recurrent Unit)

**Architecture:**
    
    
    Input: item sequence [i_1, i_2, ..., i_t]
        ↓
    Embedding Layer: [e_1, e_2, ..., e_t]
        ↓
    GRU Layer: h_t = GRU(e_t, h_{t-1})
        ↓
    Output Layer: softmax(W h_t + b) → P(next item)
    

### 4.4.3 GRU4Rec Implementation
    
    
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
        """Train GRU4Rec"""
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
    
    
    # Usage example
    if __name__ == "__main__":
        n_items = 10000
        model = GRU4Rec(n_items, embedding_dim=100, hidden_dim=100, n_layers=2)
    
        # Sample input
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

**Advantages of Self-Attention:**

  * Overcomes sequential processing constraints of RNN/GRU
  * Enables parallel computation, easier to capture long-range dependencies
  * Paper: Kang & McAuley "SASRec" (ICDM 2018)

**Architecture:**
    
    
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
    

### 4.4.5 SASRec Implementation
    
    
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
    
    
    # Usage example
    if __name__ == "__main__":
        n_items = 10000
        max_len = 50
    
        model = SASRec(n_items, max_len=max_len, hidden_dim=128,
                       n_heads=4, n_blocks=2, dropout=0.2)
    
        # Sample input
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

  * Paper: Sun et al. "BERT4Rec" (CIKM 2019)
  * Bidirectional Transformer (utilizes past + future context)
  * Masked Item Prediction (MLM-style)

**Training Method:**

  * Mask portions of input sequence ([MASK] token)
  * Predict masked items
  * Can utilize bidirectional context

* * *

## 4.5 Practical Recommendation System Pipeline

### 4.5.1 End-to-End Pipeline

**Two-Stage Architecture for Recommendation Systems:**
    
    
    Stage 1: Candidate Generation
    ├─ Input: User context
    ├─ Method: Two-Tower, Matrix Factorization, Graph-based
    ├─ Output: Hundreds to thousands of candidate items
    └─ Goal: Maximize recall, high speed
    
    Stage 2: Ranking
    ├─ Input: User + candidate items
    ├─ Method: DeepFM, Wide&Deep, DNN
    ├─ Output: Top-K recommendation list
    └─ Goal: Maximize precision, focus on accuracy
    

### 4.5.2 Candidate Generation Example
    
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class CandidateGenerator:
        """Candidate generation module"""
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
        """Ranking module (DeepFM, etc.)"""
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
        """Recommendation pipeline (candidate generation + ranking)"""
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
    
    
    # Usage example
    if __name__ == "__main__":
        # Sample data
        n_users = 10000
        n_items = 50000
        emb_dim = 128
    
        user_embeddings = np.random.randn(n_users, emb_dim).astype('float32')
        item_embeddings = np.random.randn(n_items, emb_dim).astype('float32')
    
        # Candidate Generator
        candidate_gen = CandidateGenerator(user_embeddings, item_embeddings)
    
        # Sample ranking model (in practice, use DeepFM, etc.)
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
    

### 4.5.3 A/B Testing

**Experimental Design:**

  * Control Group: Existing recommendation algorithm
  * Treatment Group: New algorithm (DeepFM, Two-Tower, etc.)
  * Evaluation Metrics: CTR, Conversion Rate, Revenue, User Engagement

**Statistical Significance Testing:**

  * Null hypothesis: $H_0$: No effect of new algorithm
  * Alternative hypothesis: $H_1$: New algorithm has effect
  * Significance level: $\alpha = 0.05$ (5%)
  * Test methods: t-test, Welch's t-test, Mann-Whitney U test

### 4.5.4 Production Deployment

**System Requirements:**

  * Latency: < 100ms (inference time)
  * Throughput: > 10,000 QPS (Queries Per Second)
  * Scalability: Horizontally scalable

**Example Technology Stack:**

  * Model serving: TensorFlow Serving, TorchServe, NVIDIA Triton
  * Cache: Redis (user embeddings, item embeddings)
  * ANN search: FAISS, ScaNN, Milvus
  * Log collection: Kafka, Fluentd
  * Monitoring: Prometheus, Grafana

* * *

## Summary

What we learned in this chapter:

  1. **Neural Collaborative Filtering (NCF):**
     * Three approaches: GMF, MLP, NeuMF
     * Represent non-linear interactions with MLP
     * Achieved HR@10 0.726 on MovieLens
  2. **Factorization Machines:**
     * Compute second-order interactions in $O(kn)$
     * Field-specific embeddings with FFM
     * Fusion of FM + DNN in DeepFM
  3. **Two-Tower Models:**
     * Separate User Tower and Item Tower
     * Fast inference with FAISS/ScaNN ($O(\log N)$)
     * Contrastive learning with InfoNCE Loss
  4. **Sequence-based Recommendation:**
     * GRU4Rec: Session-based recommendation
     * SASRec: Long-range dependencies with Self-Attention
     * BERT4Rec: Masked prediction with bidirectional Transformer
  5. **Practical Pipeline:**
     * Candidate generation (recall-focused) + Ranking (precision-focused)
     * Effect validation with A/B testing
     * Technology stack for production deployment

* * *

## Exercises

**Question 1:** Explain why NeuMF outperforms Matrix Factorization from the perspectives of linearity and non-linearity.

**Question 2:** Using mathematical formulas, show the mechanism by which the computational complexity of second-order interactions in Factorization Machines is reduced from $O(n^2)$ to $O(kn)$.

**Question 3:** List three advantages and three disadvantages of separating User Tower and Item Tower in Two-Tower Models.

**Question 4:** Discuss the differences between SASRec and GRU4Rec from two perspectives: parallelization capability and capturing long-range dependencies.

**Question 5:** If Recall@500 = 0.8 in the Candidate Generation stage and Precision@10 = 0.3 in the Ranking stage of a recommendation system, calculate the expected number of hits in the final Top-10 recommendations.

**Question 6:** Propose five technical approaches to achieve latency < 100ms in a large-scale recommendation system (100 million users × 10 million items).

* * *

## References

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
