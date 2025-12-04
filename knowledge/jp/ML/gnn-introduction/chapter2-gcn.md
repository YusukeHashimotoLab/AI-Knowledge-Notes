---
title: 第2章：Graph Convolutional Networks (GCN)
chapter_title: 第2章：Graph Convolutional Networks (GCN)
subtitle: スペクトルグラフ理論から実装まで完全理解
reading_time: 30-35分
difficulty: 中級〜上級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ スペクトルグラフ理論の基礎（グラフラプラシアン、固有値分解）を理解する
  * ✅ CNNからGCNへの理論的な拡張の動機を説明できる
  * ✅ GCN層の数学的定式化と対称正規化の意味を理解する
  * ✅ PyTorchでGCN層を一から実装できる
  * ✅ PyTorch GeometricライブラリでGCNモデルを構築できる
  * ✅ Coraデータセットでノード分類タスクを実行できる

* * *

## 2.1 スペクトルグラフ理論の基礎

### グラフラプラシアン（Graph Laplacian）

**グラフラプラシアン** は、グラフの構造を行列で表現する基本的な手法です。グラフ $G = (V, E)$ に対して、以下のように定義されます：

$$ L = D - A $$ 

ここで：

  * $A$：隣接行列（Adjacency matrix）
  * $D$：次数行列（Degree matrix）、対角要素が $D_{ii} = \sum_j A_{ij}$
  * $L$：ラプラシアン行列（Laplacian matrix）

### 正規化ラプラシアン

GCNでは、**対称正規化ラプラシアン** が使用されます：

$$ L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2} $$ 

また、ランダムウォーク正規化ラプラシアンもあります：

$$ L_{\text{rw}} = D^{-1} L = I - D^{-1} A $$ 
    
    
    ```mermaid
    graph LR
        A["隣接行列 A"] --> L["ラプラシアン L = D - A"]
        D["次数行列 D"] --> L
        L --> Lsym["対称正規化 L_sym"]
        L --> Lrw["ランダムウォーク L_rw"]
    
        style A fill:#b3e5fc
        style D fill:#c5e1a5
        style L fill:#fff9c4
        style Lsym fill:#ffab91
    ```

### グラフラプラシアンの実装
    
    
    import numpy as np
    import torch
    import networkx as nx
    import matplotlib.pyplot as plt
    
    def compute_graph_laplacian(A):
        """
        グラフラプラシアンを計算
    
        Args:
            A: (N, N) 隣接行列
    
        Returns:
            L: (N, N) ラプラシアン行列
            D: (N, N) 次数行列
        """
        # 次数行列（対角行列）
        D = np.diag(A.sum(axis=1))
    
        # ラプラシアン L = D - A
        L = D - A
    
        return L, D
    
    
    def compute_normalized_laplacian(A, method='symmetric'):
        """
        正規化ラプラシアンを計算
    
        Args:
            A: (N, N) 隣接行列
            method: 'symmetric' or 'random_walk'
    
        Returns:
            L_norm: (N, N) 正規化ラプラシアン
        """
        # 次数行列
        D = np.diag(A.sum(axis=1))
    
        # D^{-1/2}（対称正規化用）
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    
        if method == 'symmetric':
            # L_sym = I - D^{-1/2} A D^{-1/2}
            L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
        elif method == 'random_walk':
            # L_rw = I - D^{-1} A
            D_inv = np.diag(1.0 / (np.diag(D) + 1e-10))
            L_norm = np.eye(len(A)) - D_inv @ A
        else:
            raise ValueError(f"Unknown method: {method}")
    
        return L_norm
    
    
    # 簡単なグラフの例
    print("=== グラフラプラシアンの計算例 ===")
    
    # 5ノードのグラフを作成
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    
    # 隣接行列
    A = nx.adjacency_matrix(G).toarray()
    print(f"隣接行列 A:\n{A}\n")
    
    # ラプラシアン行列
    L, D = compute_graph_laplacian(A)
    print(f"次数行列 D:\n{D}\n")
    print(f"ラプラシアン L = D - A:\n{L}\n")
    
    # 正規化ラプラシアン
    L_sym = compute_normalized_laplacian(A, method='symmetric')
    print(f"対称正規化ラプラシアン L_sym:\n{L_sym}\n")
    
    # ラプラシアンの性質を確認
    print("=== ラプラシアンの性質 ===")
    print(f"Lの各行の和: {L.sum(axis=1)}")
    print("→ すべて0（ラプラシアンの重要な性質）")
    
    # グラフの可視化
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=800, font_size=16, font_weight='bold')
    plt.title("Sample Graph (5 nodes)")
    plt.savefig('graph_example.png', dpi=150, bbox_inches='tight')
    print("\nグラフを保存: graph_example.png")
    plt.close()
    

### 固有値分解とスペクトル

ラプラシアン行列の**固有値分解** は、グラフの周波数成分を表します：

$$ L = U \Lambda U^T $$ 

ここで：

  * $U$：固有ベクトル行列（グラフフーリエ基底）
  * $\Lambda$：固有値の対角行列（周波数に対応）

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def spectral_analysis(L):
        """
        ラプラシアンのスペクトル分析
    
        Args:
            L: (N, N) ラプラシアン行列
    
        Returns:
            eigenvalues: (N,) 固有値（昇順）
            eigenvectors: (N, N) 固有ベクトル
        """
        # 固有値分解
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    
        return eigenvalues, eigenvectors
    
    
    print("\n=== スペクトル分析 ===")
    
    # 固有値分解
    eigenvalues, eigenvectors = spectral_analysis(L)
    
    print(f"固有値（昇順）:\n{eigenvalues}\n")
    print(f"固有ベクトル（最初の3つ）:\n{eigenvectors[:, :3]}\n")
    
    # 固有値の可視化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(eigenvalues)), eigenvalues, color='steelblue')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Laplacian Eigenvalues (Spectrum)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(eigenvectors[:, 0], 'o-', label='1st eigenvector', markersize=8)
    plt.plot(eigenvectors[:, 1], 's-', label='2nd eigenvector', markersize=8)
    plt.plot(eigenvectors[:, 2], '^-', label='3rd eigenvector', markersize=8)
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.title('First 3 Eigenvectors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laplacian_spectrum.png', dpi=150, bbox_inches='tight')
    print("スペクトルを保存: laplacian_spectrum.png")
    plt.close()
    
    # 重要な性質
    print("=== ラプラシアンの重要な性質 ===")
    print(f"最小固有値: {eigenvalues[0]:.6f}")
    print("→ 連結グラフでは常に0")
    print(f"最小固有値の固有ベクトル: {eigenvectors[:, 0]}")
    print("→ すべて同じ値（定数ベクトル）")
    

### スペクトル畳み込み

グラフ上の畳み込みは、**フーリエ領域** で定義されます：

$$ x \star_G g = U \left( (U^T g) \odot (U^T x) \right) $$ 

ここで：

  * $x$：ノード特徴ベクトル
  * $g$：フィルタ（周波数領域で定義）
  * $\odot$：要素ごとの積（Hadamard積）

> 「スペクトル畳み込みは、グラフフーリエ変換を用いてグラフ上の信号を周波数領域で処理します」

* * *

## 2.2 CNNからGCNへの拡張

### CNNの畳み込み演算

通常のCNN（Convolutional Neural Networks）では、**画像のグリッド構造** 上で畳み込みを行います：

$$ h_i^{(\ell+1)} = \sigma \left( W^{(\ell)} \sum_{j \in \mathcal{N}(i)} h_j^{(\ell)} + b^{(\ell)} \right) $$ 

ここで、$\mathcal{N}(i)$ は画素 $i$ の近傍（通常は3×3カーネル）です。

### グラフへの一般化の課題

課題 | 画像（グリッド） | グラフ  
---|---|---  
**近傍のサイズ** | 固定（3×3など） | ノードごとに異なる  
**順序** | 固定（上下左右） | 定義なし  
**距離** | ユークリッド距離 | グラフ上の距離  
**対称性** | 並進対称性 | 置換対称性  
      
    
    ```mermaid
    graph TB
        CNN["CNNグリッド構造"] --> Challenge["課題不規則なグラフ構造"]
        Challenge --> Spectral["アプローチ1スペクトル法"]
        Challenge --> Spatial["アプローチ2空間法"]
        Spectral --> GCN["GCN一次近似"]
    
        style CNN fill:#b3e5fc
        style Challenge fill:#fff9c4
        style Spectral fill:#c5e1a5
        style GCN fill:#ffab91
    ```

### GCNの基本アイデア

**Graph Convolutional Networks (GCN)** は、スペクトル畳み込みを**一次近似** することで、効率的なグラフ畳み込みを実現します：

$$ H^{(\ell+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(\ell)} W^{(\ell)} \right) $$ 

ここで：

  * $\tilde{A} = A + I$：自己ループを追加した隣接行列
  * $\tilde{D}$：$\tilde{A}$ の次数行列
  * $H^{(\ell)}$：$\ell$ 層目のノード特徴行列
  * $W^{(\ell)}$：学習可能な重み行列
  * $\sigma$：活性化関数

### 自己ループの追加とその意義
    
    
    import numpy as np
    import torch
    
    def add_self_loops(A):
        """
        隣接行列に自己ループを追加
    
        Args:
            A: (N, N) 隣接行列
    
        Returns:
            A_tilde: (N, N) 自己ループ付き隣接行列
        """
        # 単位行列を追加
        A_tilde = A + np.eye(len(A))
        return A_tilde
    
    
    print("\n=== 自己ループの追加 ===")
    
    # 元の隣接行列
    print(f"元の隣接行列 A:\n{A}\n")
    
    # 自己ループを追加
    A_tilde = add_self_loops(A)
    print(f"自己ループ付き Ã = A + I:\n{A_tilde}\n")
    
    print("=== 自己ループの意義 ===")
    print("1. ノード自身の特徴を保持")
    print("2. 次数が0のノードでも情報を持てる")
    print("3. 数値安定性の向上")
    

### 対称正規化の導出

対称正規化は、**ノードの次数の影響を正規化** します：

$$ \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} $$ 

これにより：

  1. 次数が大きいノードからの寄与を正規化
  2. 対称行列となり数値的に安定
  3. 固有値が [-1, 1] の範囲に収まる

    
    
    import numpy as np
    
    def symmetric_normalization(A):
        """
        対称正規化隣接行列を計算
    
        Args:
            A: (N, N) 隣接行列
    
        Returns:
            A_hat: (N, N) 正規化隣接行列
        """
        # 自己ループを追加
        A_tilde = A + np.eye(len(A))
    
        # 次数行列
        D_tilde = np.diag(A_tilde.sum(axis=1))
    
        # D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    
        # Â = D^{-1/2} Ã D^{-1/2}
        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
        return A_hat
    
    
    print("\n=== 対称正規化 ===")
    
    A_hat = symmetric_normalization(A)
    print(f"正規化隣接行列 Â:\n{A_hat}\n")
    
    # 各行の和を確認
    print(f"Âの各行の和:\n{A_hat.sum(axis=1)}\n")
    print("→ 必ずしも1ではないが、バランスが取れている")
    
    # 対称性を確認
    is_symmetric = np.allclose(A_hat, A_hat.T)
    print(f"Âは対称行列: {is_symmetric}")
    

* * *

## 2.3 GCN層の数学的定式化

### 単一GCN層の定義

1つのGCN層は、以下の演算を実行します：

$$ H^{(\ell+1)} = \sigma \left( \hat{A} H^{(\ell)} W^{(\ell)} \right) $$ 

ここで：

  * $H^{(\ell)} \in \mathbb{R}^{N \times d_\ell}$：$\ell$ 層目のノード特徴（$N$ はノード数、$d_\ell$ は特徴次元）
  * $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell+1}}$：学習可能な重み行列
  * $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$：正規化隣接行列
  * $\sigma$：活性化関数（ReLU、Tanhなど）

### 計算の流れ
    
    
    ```mermaid
    graph LR
        H0["H^(ℓ)(N × d_ℓ)"] --> Mult1["H^(ℓ) W^(ℓ)"]
        W["W^(ℓ)(d_ℓ × d_ℓ+1)"] --> Mult1
        Mult1 --> Aggregate["Â × (H^(ℓ) W^(ℓ))"]
        Ahat["Â(N × N)"] --> Aggregate
        Aggregate --> Activation["σ(...)"]
        Activation --> H1["H^(ℓ+1)(N × d_ℓ+1)"]
    
        style H0 fill:#b3e5fc
        style W fill:#c5e1a5
        style Aggregate fill:#fff59d
        style H1 fill:#ffab91
    ```

### ノードレベルでの解釈

各ノード $i$ に対して：

$$ h_i^{(\ell+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \\{i\\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} h_j^{(\ell)} W^{(\ell)} \right) $$ 

これは、**近傍ノードの特徴の重み付き和** を計算していることを意味します。

### バイアス項の追加

実用的には、バイアス項も追加します：

$$ H^{(\ell+1)} = \sigma \left( \hat{A} H^{(\ell)} W^{(\ell)} + b^{(\ell)} \right) $$ 

### 多層GCNの定式化

$L$ 層のGCNは、再帰的に定義されます：

$$ \begin{align} H^{(0)} &= X \quad \text{（入力特徴）} \\\ H^{(\ell+1)} &= \sigma \left( \hat{A} H^{(\ell)} W^{(\ell)} \right), \quad \ell = 0, 1, \ldots, L-1 \\\ Z &= H^{(L)} \quad \text{（出力）} \end{align} $$ 
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    def gcn_layer_forward(A_hat, H, W, bias=None, activation=None):
        """
        GCN層の順伝播
    
        Args:
            A_hat: (N, N) 正規化隣接行列
            H: (N, d_in) ノード特徴
            W: (d_in, d_out) 重み行列
            bias: (d_out,) バイアス（オプション）
            activation: 活性化関数（オプション）
    
        Returns:
            H_next: (N, d_out) 次の層の特徴
        """
        # 1. 特徴変換：H @ W
        H_transformed = H @ W
    
        # 2. 近傍集約：Â @ (H @ W)
        H_aggregated = A_hat @ H_transformed
    
        # 3. バイアス追加
        if bias is not None:
            H_aggregated = H_aggregated + bias
    
        # 4. 活性化
        if activation is not None:
            H_next = activation(H_aggregated)
        else:
            H_next = H_aggregated
    
        return H_next
    
    
    # 数値例
    print("\n=== GCN層の順伝播 ===")
    
    N = 5  # ノード数
    d_in = 4  # 入力特徴次元
    d_out = 8  # 出力特徴次元
    
    # ダミーデータ
    A_hat_torch = torch.FloatTensor(A_hat)
    H = torch.randn(N, d_in)
    W = torch.randn(d_in, d_out)
    b = torch.randn(d_out)
    
    print(f"入力特徴 H: {H.shape}")
    print(f"重み W: {W.shape}")
    print(f"正規化隣接行列 Â: {A_hat_torch.shape}")
    
    # GCN層の計算
    H_next = gcn_layer_forward(A_hat_torch, H, W, bias=b, activation=torch.relu)
    
    print(f"出力特徴 H^(ℓ+1): {H_next.shape}")
    print(f"\nサンプル出力（ノード0の特徴）:\n{H_next[0]}")
    

* * *

## 2.4 PyTorchでのGCN層実装

### GCNLayerクラスの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class GCNLayer(nn.Module):
        """
        単一のGCN層
        """
        def __init__(self, in_features, out_features, bias=True):
            """
            Args:
                in_features: 入力特徴次元
                out_features: 出力特徴次元
                bias: バイアスを使用するか
            """
            super(GCNLayer, self).__init__()
    
            # 重み行列
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
    
            # バイアス
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
    
            # パラメータの初期化
            self.reset_parameters()
    
        def reset_parameters(self):
            """Xavierの初期化"""
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
        def forward(self, x, adj):
            """
            順伝播
    
            Args:
                x: (N, in_features) ノード特徴
                adj: (N, N) 正規化隣接行列
    
            Returns:
                output: (N, out_features) 出力特徴
            """
            # 特徴変換：X @ W
            support = torch.mm(x, self.weight)
    
            # 近傍集約：Â @ (X @ W)
            output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
    
            # バイアス追加
            if self.bias is not None:
                output = output + self.bias
    
            return output
    
    
    # 動作確認
    print("\n=== GCNLayerクラスの動作確認 ===")
    
    N = 10  # ノード数
    in_features = 16
    out_features = 32
    
    # GCN層の作成
    gcn_layer = GCNLayer(in_features, out_features)
    
    # ダミーデータ
    x = torch.randn(N, in_features)
    adj = torch.randn(N, N)
    # 対称正規化（簡略版）
    adj = (adj + adj.T) / 2
    adj = adj / adj.sum(dim=1, keepdim=True)
    
    # Forward
    output = gcn_layer(x, adj)
    
    print(f"入力: {x.shape}")
    print(f"隣接行列: {adj.shape}")
    print(f"出力: {output.shape}")
    
    # パラメータ数
    num_params = sum(p.numel() for p in gcn_layer.parameters())
    print(f"パラメータ数: {num_params:,}")
    print(f"  重み: {gcn_layer.weight.numel():,}")
    print(f"  バイアス: {gcn_layer.bias.numel() if gcn_layer.bias is not None else 0}")
    

### 完全なGCNモデルの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class GCN(nn.Module):
        """
        多層Graph Convolutional Network
        """
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
            """
            Args:
                input_dim: 入力特徴次元
                hidden_dim: 隠れ層の次元
                output_dim: 出力次元（クラス数）
                num_layers: GCN層の数
                dropout: ドロップアウト率
            """
            super(GCN, self).__init__()
    
            self.num_layers = num_layers
            self.dropout = dropout
    
            # GCN層のリスト
            self.gcn_layers = nn.ModuleList()
    
            # 最初の層
            self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
    
            # 中間層
            for _ in range(num_layers - 2):
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
    
            # 最後の層
            if num_layers > 1:
                self.gcn_layers.append(GCNLayer(hidden_dim, output_dim))
            else:
                # 1層のみの場合
                self.gcn_layers[0] = GCNLayer(input_dim, output_dim)
    
        def forward(self, x, adj):
            """
            順伝播
    
            Args:
                x: (N, input_dim) ノード特徴
                adj: (N, N) 正規化隣接行列
    
            Returns:
                output: (N, output_dim) 出力（ロジット）
            """
            h = x
    
            # 中間層
            for i in range(self.num_layers - 1):
                h = self.gcn_layers[i](h, adj)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
    
            # 最後の層（活性化なし）
            output = self.gcn_layers[-1](h, adj)
    
            return output
    
    
    # モデル作成
    print("\n=== GCNモデルの作成 ===")
    
    input_dim = 1433  # Coraデータセットの特徴次元
    hidden_dim = 16
    output_dim = 7  # クラス数
    num_layers = 2
    
    model = GCN(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=0.5)
    
    # ダミーデータ
    N = 100
    x = torch.randn(N, input_dim)
    adj = torch.randn(N, N)
    adj = (adj + adj.T) / 2
    adj = adj / adj.sum(dim=1, keepdim=True)
    
    # Forward
    output = model(x, adj)
    
    print(f"入力: {x.shape}")
    print(f"出力: {output.shape}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    
    # 各層のパラメータ数
    for i, layer in enumerate(model.gcn_layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"  GCN層{i+1}: {layer_params:,}")
    

### 正規化隣接行列の前処理
    
    
    import torch
    import scipy.sparse as sp
    import numpy as np
    
    def preprocess_adjacency(adj):
        """
        隣接行列を対称正規化
    
        Args:
            adj: (N, N) 隣接行列（NumPy配列またはSciPy sparse）
    
        Returns:
            adj_normalized: (N, N) 正規化隣接行列（Tensor）
        """
        # NumPy配列に変換
        if sp.issparse(adj):
            adj = adj.toarray()
    
        # 自己ループを追加
        adj_tilde = adj + np.eye(adj.shape[0])
    
        # 次数行列
        degree = np.array(adj_tilde.sum(1))
    
        # D^{-1/2}
        degree_inv_sqrt = np.power(degree, -0.5).flatten()
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(degree_inv_sqrt)
    
        # Â = D^{-1/2} Ã D^{-1/2}
        if sp.issparse(adj):
            adj_normalized = D_inv_sqrt @ sp.csr_matrix(adj_tilde) @ D_inv_sqrt
            adj_normalized = torch.FloatTensor(adj_normalized.toarray())
        else:
            adj_normalized = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
            adj_normalized = torch.FloatTensor(adj_normalized)
    
        return adj_normalized
    
    
    # 使用例
    print("\n=== 隣接行列の前処理 ===")
    
    # サンプル隣接行列
    N = 5
    adj_raw = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.float32)
    
    print(f"元の隣接行列:\n{adj_raw}\n")
    
    # 正規化
    adj_norm = preprocess_adjacency(adj_raw)
    
    print(f"正規化隣接行列:\n{adj_norm}\n")
    
    # 性質の確認
    print("=== 正規化の効果 ===")
    print(f"最大値: {adj_norm.max().item():.4f}")
    print(f"最小値: {adj_norm.min().item():.4f}")
    print(f"対角要素（自己ループ）: {adj_norm.diag()}")
    

* * *

## 2.5 PyTorch Geometricの利用

### PyTorch Geometricとは

**PyTorch Geometric (PyG)** は、グラフニューラルネットワーク用の強力なライブラリです。以下の機能を提供します：

  * 効率的なグラフデータ構造（COO形式のエッジリスト）
  * 豊富なGNN層（GCN、GAT、GraphSAGEなど）
  * グラフデータセット（Cora、PubMed、CiteSeerなど）
  * ミニバッチ処理のサポート

### PyTorch GeometricでのGCN実装
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    
    class PyGGCN(torch.nn.Module):
        """
        PyTorch GeometricのGCNConvを使用したモデル
        """
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
            super(PyGGCN, self).__init__()
    
            # GCN層
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
    
            self.dropout = dropout
    
        def forward(self, x, edge_index):
            """
            Args:
                x: (N, input_dim) ノード特徴
                edge_index: (2, E) エッジインデックス（COO形式）
    
            Returns:
                output: (N, output_dim) 出力
            """
            # 第1層
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
            # 第2層
            x = self.conv2(x, edge_index)
    
            return x
    
    
    # モデル作成
    print("\n=== PyTorch Geometric GCN ===")
    
    input_dim = 1433
    hidden_dim = 16
    output_dim = 7
    
    pyg_model = PyGGCN(input_dim, hidden_dim, output_dim)
    
    # ダミーデータ（PyG形式）
    N = 100
    E = 200
    
    x = torch.randn(N, input_dim)
    edge_index = torch.randint(0, N, (2, E))  # (2, E)
    
    # Forward
    output = pyg_model(x, edge_index)
    
    print(f"入力: {x.shape}")
    print(f"エッジインデックス: {edge_index.shape}")
    print(f"出力: {output.shape}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in pyg_model.parameters())
    print(f"総パラメータ数: {total_params:,}")
    

### グラフデータの構築
    
    
    import torch
    from torch_geometric.data import Data
    import networkx as nx
    
    def networkx_to_pyg(G, node_features=None, labels=None):
        """
        NetworkXグラフをPyTorch Geometric形式に変換
    
        Args:
            G: NetworkXグラフ
            node_features: (N, d) ノード特徴（オプション）
            labels: (N,) ノードラベル（オプション）
    
        Returns:
            data: PyTorch GeometricのDataオブジェクト
        """
        # エッジインデックス（COO形式）
        edge_list = list(G.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
        # 無向グラフの場合、逆方向のエッジも追加
        if not G.is_directed():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
        # ノード特徴
        if node_features is None:
            # ダミー特徴（単位行列）
            x = torch.eye(G.number_of_nodes())
        else:
            x = torch.FloatTensor(node_features)
    
        # ラベル
        y = torch.LongTensor(labels) if labels is not None else None
    
        # Dataオブジェクト作成
        data = Data(x=x, edge_index=edge_index, y=y)
    
        return data
    
    
    # 例：簡単なグラフ
    print("\n=== NetworkX → PyTorch Geometric ===")
    
    G = nx.karate_club_graph()
    print(f"グラフ: {G.number_of_nodes()} ノード, {G.number_of_edges()} エッジ")
    
    # PyG形式に変換
    data = networkx_to_pyg(G)
    
    print(f"\nPyGデータ:")
    print(f"  ノード特徴: {data.x.shape}")
    print(f"  エッジインデックス: {data.edge_index.shape}")
    print(f"  エッジ数: {data.num_edges}")
    print(f"  ノード数: {data.num_nodes}")
    

* * *

## 2.6 実践：Coraデータセットでのノード分類

### Coraデータセットの概要

**Coraデータセット** は、論文の引用ネットワークです：

  * **ノード** ：2,708本の論文
  * **エッジ** ：5,429の引用関係
  * **特徴** ：1,433次元のbag-of-words特徴（単語の有無）
  * **クラス** ：7つの研究分野（Case_Based、Genetic_Algorithms、Neural_Networks、Probabilistic_Methods、Reinforcement_Learning、Rule_Learning、Theory）

### データセットの読み込みと可視化
    
    
    import torch
    from torch_geometric.datasets import Planetoid
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Coraデータセットの読み込み
    print("=== Coraデータセットの読み込み ===")
    
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    
    print(f"データセット: {dataset}")
    print(f"グラフ数: {len(dataset)}")
    print(f"\nグラフ情報:")
    print(f"  ノード数: {data.num_nodes}")
    print(f"  エッジ数: {data.num_edges}")
    print(f"  ノード特徴次元: {data.num_node_features}")
    print(f"  クラス数: {dataset.num_classes}")
    print(f"\nデータ分割:")
    print(f"  訓練ノード: {data.train_mask.sum().item()}")
    print(f"  検証ノード: {data.val_mask.sum().item()}")
    print(f"  テストノード: {data.test_mask.sum().item()}")
    
    # クラス分布
    print(f"\nクラス分布:")
    for i in range(dataset.num_classes):
        count = (data.y == i).sum().item()
        print(f"  クラス {i}: {count} ノード")
    
    # データの一部を確認
    print(f"\n最初の5ノードの特徴（非ゼロ要素のみ）:")
    for i in range(5):
        nonzero = data.x[i].nonzero().squeeze()
        print(f"  ノード {i}: {len(nonzero)} 個の単語, ラベル={data.y[i].item()}")
    

### GCNモデルの訓練
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    
    class CoraGCN(torch.nn.Module):
        """Cora用のGCNモデル"""
        def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
            super(CoraGCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)
            self.dropout = dropout
    
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    
    def train(model, data, optimizer):
        """1エポックの訓練"""
        model.train()
        optimizer.zero_grad()
    
        # Forward
        out = model(data.x, data.edge_index)
    
        # 訓練ノードのみで損失計算
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
        # Backward
        loss.backward()
        optimizer.step()
    
        return loss.item()
    
    
    def evaluate(model, data, mask):
        """評価"""
        model.eval()
    
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
    
            # 精度計算
            correct = (pred[mask] == data.y[mask]).sum().item()
            accuracy = correct / mask.sum().item()
    
        return accuracy
    
    
    # モデル作成
    print("\n=== GCNモデルの訓練 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")
    
    model = CoraGCN(
        num_features=dataset.num_node_features,
        hidden_dim=16,
        num_classes=dataset.num_classes,
        dropout=0.5
    ).to(device)
    
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 訓練ループ
    num_epochs = 200
    best_val_acc = 0
    best_test_acc = 0
    
    print("\nエポック | 損失   | 訓練精度 | 検証精度 | テスト精度")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data, optimizer)
    
        if epoch % 10 == 0:
            train_acc = evaluate(model, data, data.train_mask)
            val_acc = evaluate(model, data, data.val_mask)
            test_acc = evaluate(model, data, data.test_mask)
    
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    
            print(f"{epoch:7d} | {loss:.4f} | {train_acc:.4f}   | {val_acc:.4f}   | {test_acc:.4f}")
    
    print(f"\n最良の検証精度: {best_val_acc:.4f}")
    print(f"対応するテスト精度: {best_test_acc:.4f}")
    

### 学習曲線の可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def train_with_history(model, data, optimizer, num_epochs=200):
        """訓練履歴を記録"""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }
    
        for epoch in range(1, num_epochs + 1):
            # 訓練
            loss = train(model, data, optimizer)
    
            # 評価
            train_acc = evaluate(model, data, data.train_mask)
            val_acc = evaluate(model, data, data.val_mask)
            test_acc = evaluate(model, data, data.test_mask)
    
            history['train_loss'].append(loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
    
        return history
    
    
    # 新しいモデルで訓練
    print("\n=== 学習曲線の記録 ===")
    
    model_new = CoraGCN(
        num_features=dataset.num_node_features,
        hidden_dim=16,
        num_classes=dataset.num_classes,
        dropout=0.5
    ).to(device)
    
    optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.01, weight_decay=5e-4)
    
    history = train_with_history(model_new, data, optimizer_new, num_epochs=200)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 損失
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 精度
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].plot(history['test_acc'], label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cora_training_curves.png', dpi=150, bbox_inches='tight')
    print("学習曲線を保存: cora_training_curves.png")
    plt.close()
    
    print(f"\n最終精度:")
    print(f"  訓練: {history['train_acc'][-1]:.4f}")
    print(f"  検証: {history['val_acc'][-1]:.4f}")
    print(f"  テスト: {history['test_acc'][-1]:.4f}")
    

### ノード埋め込みの可視化
    
    
    import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    def visualize_embeddings(model, data, layer='layer1'):
        """ノード埋め込みをt-SNEで可視化"""
        model.eval()
    
        with torch.no_grad():
            # 第1層の出力を取得
            x = model.conv1(data.x, data.edge_index)
            x = F.relu(x)
    
            if layer == 'layer2':
                x = model.conv2(x, data.edge_index)
    
            embeddings = x.cpu().numpy()
    
        # t-SNE
        print(f"\n=== t-SNE埋め込み（{embeddings.shape[1]}次元 → 2次元）===")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    
        # 可視化
        plt.figure(figsize=(12, 10))
    
        # クラスごとに色分け
        labels = data.y.cpu().numpy()
        for i in range(dataset.num_classes):
            mask = labels == i
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       label=f'Class {i}', alpha=0.6, s=30)
    
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(f'Node Embeddings Visualization ({layer})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        filename = f'cora_embeddings_{layer}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"埋め込みを保存: {filename}")
        plt.close()
    
    
    # 第1層と第2層の埋め込みを可視化
    visualize_embeddings(model_new, data, layer='layer1')
    visualize_embeddings(model_new, data, layer='layer2')
    
    print("\n→ 同じクラスのノードが近くに配置されていることを確認")
    

* * *

## 演習問題

**演習1：GCN層数の影響調査**

GCNの層数（1層、2層、3層、4層）を変えて、Coraデータセットでの性能を比較してください。過平滑化（over-smoothing）の問題を観察できますか？
    
    
    import torch
    
    # TODO: 異なる層数のGCNモデルを訓練
    # TODO: テスト精度をプロット
    # TODO: 層数が増えると性能が低下する理由を分析
    # ヒント: 層が深すぎるとノード表現が似通ってしまう（過平滑化）
    

**演習2：ドロップアウト率の最適化**

ドロップアウト率（0.0, 0.2, 0.5, 0.7, 0.9）を変えて、訓練精度と検証精度の変化を調査してください。
    
    
    import torch
    
    # TODO: 異なるドロップアウト率でモデルを訓練
    # TODO: 訓練精度 vs 検証精度のグラフを作成
    # TODO: 最適なドロップアウト率を見つける
    # 期待: 適度なドロップアウトで過学習を防ぐ
    

**演習3：隠れ層の次元数の影響**

隠れ層の次元数（4, 8, 16, 32, 64, 128）を変えて、性能と計算時間のトレードオフを調査してください。
    
    
    import torch
    import time
    
    # TODO: 異なる隠れ層次元でモデルを訓練
    # TODO: テスト精度とエポックあたりの時間を記録
    # TODO: 性能 vs 計算コストのグラフを作成
    # 分析: 次元が大きいほど表現力が高いが、過学習しやすい
    

**演習4：正規化手法の比較**

対称正規化、ランダムウォーク正規化、正規化なしの3つの手法でGCNを訓練し、性能を比較してください。
    
    
    import torch
    import numpy as np
    
    # TODO: 3種類の正規化手法を実装
    # TODO: それぞれでGCNを訓練
    # TODO: テスト精度を比較
    # 期待: 対称正規化が最も安定して高性能
    

**演習5：他のデータセットでの実験**

CiteSeerまたはPubMedデータセットでGCNを訓練し、Coraとの違いを分析してください。
    
    
    from torch_geometric.datasets import Planetoid
    
    # TODO: CiteSeerまたはPubMedデータセットを読み込む
    # TODO: 同じGCNモデルで訓練
    # TODO: データセット間の性能差を分析
    # TODO: グラフ構造の違い（密度、次数分布など）を調査
    

* * *

## まとめ

この章では、Graph Convolutional Networks (GCN)の理論と実装を学びました。

### 重要ポイント

  * **グラフラプラシアン** ：グラフ構造を行列で表現し、固有値分解でスペクトル解析
  * **スペクトル畳み込み** ：フーリエ領域でのグラフ信号処理
  * **GCNの動機** ：CNNの畳み込みをグラフ構造に拡張
  * **対称正規化** ：次数の影響を正規化し、数値安定性を向上
  * **GCN層** ：近傍ノードの特徴を集約して新しい表現を学習
  * **実装** ：PyTorchでの一から実装とPyTorch Geometricの活用
  * **ノード分類** ：Coraデータセットで高精度な論文分類を実現
  * **過平滑化** ：層が深すぎるとノード表現が似通う問題

### GCNの利点と限界

項目 | 利点 | 限界  
---|---|---  
**計算効率** | 線形計算量 $O(|E|)$ | 大規模グラフでメモリ制約  
**表現力** | グラフ構造を活用 | 過平滑化問題  
**一般化性** | 様々なグラフタスクに適用可能 | 動的グラフには不向き  
**解釈性** | 近傍集約の直感的な理解 | アテンション機構なし  
  
### 次のステップ

次章では、**Graph Attention Networks (GAT)** について学びます。アテンション機構によるより柔軟な近傍集約、マルチヘッドアテンション、GCNとの比較など、GNNの表現力をさらに向上させる手法を習得します。
