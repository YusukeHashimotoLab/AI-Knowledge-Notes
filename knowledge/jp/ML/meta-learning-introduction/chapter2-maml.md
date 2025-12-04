---
title: 第2章：MAML - Model-Agnostic Meta-Learning
chapter_title: 第2章：MAML - Model-Agnostic Meta-Learning
subtitle: 勾配ベースのメタ学習の最重要手法
reading_time: 30-35分
difficulty: 中級-上級
code_examples: 8
exercises: 3
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ MAMLの原理と二段階最適化を理解する
  * ✅ 勾配の勾配（Second-order derivatives）の計算方法を習得する
  * ✅ PyTorchとhigherライブラリでMAMLを実装できる
  * ✅ First-order MAML (FOMAML)の効率化を理解する
  * ✅ Reptileアルゴリズムの実装と比較ができる
  * ✅ Omniglot Few-Shot分類でMAMLを実践できる

* * *

## 2.1 MAMLの原理

### Model-Agnostic Meta-Learning概要

**MAML (Model-Agnostic Meta-Learning)** は、Chelsea Finnらによって2017年に提案された勾配ベースのメタ学習アルゴリズムです。

> 「少数のデータからの勾配更新で素早く適応できる、良い初期パラメータを学習する」

### MAMLの核心的アイデア

MAMLは以下の問いに答えます：

  * **問い** ：どのような初期パラメータ $\theta$ を選べば、新しいタスク $\mathcal{T}_i$ に対して、わずかな勾配ステップで高い性能を達成できるか？
  * **答え** ：複数のタスクで「勾配降下後の性能」を最大化するパラメータを学習する

### 二段階最適化（Inner/Outer Loop）

MAMLは以下の2つのループからなります：

ループ | 目的 | データ | 更新対象  
---|---|---|---  
**Inner Loop** | タスク適応 | サポートセット $\mathcal{D}^{tr}_i$ | タスク固有パラメータ $\theta'_i$  
**Outer Loop** | メタ学習 | クエリセット $\mathcal{D}^{test}_i$ | メタパラメータ $\theta$  
  
### MAMLの動作フロー
    
    
    ```mermaid
    graph TD
        A[メタパラメータ θ] --> B[タスク1: θ → θ'₁]
        A --> C[タスク2: θ → θ'₂]
        A --> D[タスクN: θ → θ'ₙ]
    
        B --> E[クエリセットで評価 L₁]
        C --> F[クエリセットで評価 L₂]
        D --> G[クエリセットで評価 Lₙ]
    
        E --> H[メタ損失 = 平均]
        F --> H
        G --> H
    
        H --> I[θを更新]
        I --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style H fill:#ffebee
        style I fill:#c8e6c9
    ```

### 勾配の勾配（Second-order derivatives）

MAMLの特徴は**勾配の勾配** を計算することです。

**Inner Loop（一次勾配）** ：

$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta) $$

**Outer Loop（二次勾配）** ：

$$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

ここで、$\theta'_i$ は $\theta$ の関数なので、以下のように展開されます：

$$ \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) \cdot \nabla_\theta \theta'_i $$

> **重要** ：この二次微分が計算コストの主な要因ですが、適応能力を高めます。

### MAMLの視覚的理解
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 2次元パラメータ空間でのMAMLのイメージ
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：通常の学習
    ax1 = axes[0]
    theta_init = np.array([0, 0])
    task1_opt = np.array([3, 1])
    task2_opt = np.array([1, 3])
    task3_opt = np.array([-2, 2])
    
    ax1.scatter(*theta_init, s=200, c='red', marker='X', label='ランダム初期化', zorder=5)
    ax1.scatter(*task1_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax1.scatter(*task2_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax1.scatter(*task3_opt, s=100, c='blue', marker='o', alpha=0.7, label='タスク最適解')
    
    for opt in [task1_opt, task2_opt, task3_opt]:
        ax1.arrow(theta_init[0], theta_init[1],
                  opt[0]-theta_init[0]*0.9, opt[1]-theta_init[1]*0.9,
                  head_width=0.2, head_length=0.2, fc='gray', ec='gray', alpha=0.5)
    
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_title('通常の学習：タスクごとにゼロから学習', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右図：MAML
    ax2 = axes[1]
    maml_init = np.array([0.7, 2])
    
    ax2.scatter(*maml_init, s=200, c='green', marker='X', label='MAML初期化', zorder=5)
    ax2.scatter(*task1_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax2.scatter(*task2_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax2.scatter(*task3_opt, s=100, c='blue', marker='o', alpha=0.7, label='タスク最適解')
    
    for opt in [task1_opt, task2_opt, task3_opt]:
        ax2.arrow(maml_init[0], maml_init[1],
                  opt[0]-maml_init[0]*0.7, opt[1]-maml_init[1]*0.7,
                  head_width=0.2, head_length=0.2, fc='green', ec='green', alpha=0.5)
    
    # 中心領域を強調
    circle = plt.Circle(maml_init, 1.5, color='green', fill=False, linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-1, 4)
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('MAML：全タスクに近い位置から開始', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== MAML の直感的理解 ===")
    print("✓ 通常の学習: 各タスクをゼロから学習（遠い）")
    print("✓ MAML: 全タスクの「中心」に位置する初期化を学習")
    print("✓ 結果: わずかな勾配ステップで各タスクに適応可能")
    

* * *

## 2.2 MAMLアルゴリズム

### 数式による定義

**メタ学習の目的** ：

$$ \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

ここで、$\theta'_i$ はタスク $\mathcal{T}_i$ に対する適応後パラメータ：

$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta) $$

**記号の意味** ：

  * $\theta$: メタパラメータ（学習対象）
  * $\theta'_i$: タスク $i$ に適応したパラメータ
  * $\alpha$: Inner Loop学習率（タスク適応）
  * $\beta$: Outer Loop学習率（メタ学習）
  * $\mathcal{L}_{\mathcal{T}_i}^{tr}$: タスク $i$ のサポートセット損失
  * $\mathcal{L}_{\mathcal{T}_i}^{test}$: タスク $i$ のクエリセット損失

### Inner Loop: タスク適応

各タスク $\mathcal{T}_i$ に対して、サポートセット $\mathcal{D}^{tr}_i$ を使って勾配降下：

$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta) $$

複数ステップの場合（$K$ ステップ）：

$$ \begin{aligned} \theta_i^{(0)} &= \theta \\\ \theta_i^{(k+1)} &= \theta_i^{(k)} - \alpha \nabla_{\theta_i^{(k)}} \mathcal{L}_{\mathcal{T}_i}^{tr}(f_{\theta_i^{(k)}}) \\\ \theta'_i &= \theta_i^{(K)} \end{aligned} $$

### Outer Loop: メタパラメータ更新

適応後のパラメータ $\theta'_i$ でクエリセット $\mathcal{D}^{test}_i$ を評価し、メタ損失を計算：

$$ \mathcal{L}_{\text{meta}}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

メタパラメータの更新：

$$ \theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\text{meta}}(\theta) $$

### アルゴリズムの擬似コード
    
    
    Algorithm: MAML
    
    Require: p(T): タスク分布
    Require: α, β: Inner/Outer Loop 学習率
    
    1: ランダムに θ を初期化
    2: while not converged do
    3:     B ← Sample batch of tasks {T_i} ~ p(T)
    4:     for all T_i ∈ B do
    5:         # Inner Loop: タスク適応
    6:         D_i^tr, D_i^test ← Sample support/query sets from T_i
    7:         θ'_i ← θ - α ∇_θ L_{T_i}^tr(f_θ)
    8:
    9:         # Query セットで損失を計算
    10:        L_i ← L_{T_i}^test(f_{θ'_i})
    11:    end for
    12:
    13:    # Outer Loop: メタ学習
    14:    θ ← θ - β ∇_θ Σ L_i
    15: end while
    16: return θ
    

### First-order MAML (FOMAML)

**課題** ：二次微分の計算コストが高い

**解決策** ：二次微分項を無視する近似

FOMAML では、以下のように近似します：

$$ \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) \approx \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

つまり、$\nabla_\theta \theta'_i$ の項を無視します。

比較項目 | MAML | FOMAML  
---|---|---  
**勾配計算** | 二次微分 | 一次微分のみ  
**計算コスト** | 高い | 低い（約50%削減）  
**メモリ使用量** | 多い | 少ない  
**性能** | 最高 | わずかに劣る（実用的）  
  
> **実践的には** 、FOMAMLで十分な性能が得られることが多く、広く使われています。

* * *

## 2.3 PyTorchによるMAML実装

### higher ライブラリの活用

**higher** は、PyTorchで高次微分を扱うための便利なライブラリです。MAMLの実装に最適です。
    
    
    # higherのインストール
    # pip install higher
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import higher
    import numpy as np
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

### シンプルなモデル定義
    
    
    class SimpleMLP(nn.Module):
        """Few-Shot学習用のシンプルなMLP"""
        def __init__(self, input_size=1, hidden_size=40, output_size=1):
            super(SimpleMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
    
        def forward(self, x):
            return self.net(x)
    
    # モデルのインスタンス化
    model = SimpleMLP().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    

### タスク生成関数
    
    
    def generate_sinusoid_task(amplitude=None, phase=None, n_samples=10):
        """
        正弦波回帰タスクを生成
    
        Args:
            amplitude: 振幅（Noneの場合はランダム）
            phase: 位相（Noneの場合はランダム）
            n_samples: サンプル数
    
        Returns:
            x, y: 入力と出力のペア
        """
        if amplitude is None:
            amplitude = np.random.uniform(0.1, 5.0)
        if phase is None:
            phase = np.random.uniform(0, np.pi)
    
        x = np.random.uniform(-5, 5, n_samples)
        y = amplitude * np.sin(x + phase)
    
        x = torch.FloatTensor(x).unsqueeze(1).to(device)
        y = torch.FloatTensor(y).unsqueeze(1).to(device)
    
        return x, y, amplitude, phase
    
    # タスク例の可視化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        x_train, y_train, amp, ph = generate_sinusoid_task(n_samples=10)
        x_test = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
        y_test = amp * np.sin(x_test.cpu().numpy() + ph)
    
        ax.scatter(x_train.cpu(), y_train.cpu(), label='Training samples', s=50, alpha=0.7)
        ax.plot(x_test.cpu(), y_test, 'r--', label='True function', alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Task {i+1}: A={amp:.2f}, φ={ph:.2f}', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== タスク分布 ===")
    print("各タスクは異なる振幅と位相を持つ正弦波")
    print("目標: 少数のサンプルから新しい正弦波に適応")
    

### Inner Loop実装
    
    
    def inner_loop(model, x_support, y_support, inner_lr=0.01, inner_steps=1):
        """
        Inner Loop: タスク適応
    
        Args:
            model: PyTorchモデル
            x_support: サポートセット入力
            y_support: サポートセット出力
            inner_lr: Inner Loop学習率
            inner_steps: 適応ステップ数
    
        Returns:
            task_loss: サポートセット損失
        """
        criterion = nn.MSELoss()
    
        # 予測と損失計算
        predictions = model(x_support)
        task_loss = criterion(predictions, y_support)
    
        # 勾配計算
        task_grad = torch.autograd.grad(
            task_loss,
            model.parameters(),
            create_graph=True  # 二次微分のために必要
        )
    
        # 手動で勾配降下（inner_stepsが1の場合）
        adapted_params = []
        for param, grad in zip(model.parameters(), task_grad):
            adapted_params.append(param - inner_lr * grad)
    
        return task_loss, adapted_params
    
    # Inner Loop の動作確認
    print("\n=== Inner Loop テスト ===")
    x_sup, y_sup, _, _ = generate_sinusoid_task(n_samples=5)
    loss, adapted = inner_loop(model, x_sup, y_sup)
    print(f"Support loss: {loss.item():.4f}")
    print(f"Adapted parameters: {len(adapted)} tensors")
    

### Outer Loop実装
    
    
    def outer_loop(model, tasks, inner_lr=0.01, inner_steps=1):
        """
        Outer Loop: メタ学習
    
        Args:
            model: PyTorchモデル
            tasks: タスクのリスト [(x_sup, y_sup, x_qry, y_qry), ...]
            inner_lr: Inner Loop学習率
            inner_steps: 適応ステップ数
    
        Returns:
            meta_loss: メタ損失（平均クエリ損失）
        """
        criterion = nn.MSELoss()
        meta_loss = 0.0
    
        for x_support, y_support, x_query, y_query in tasks:
            # Inner Loop: タスク適応（higherを使用）
            with higher.innerloop_ctx(
                model,
                optim.SGD(model.parameters(), lr=inner_lr),
                copy_initial_weights=False
            ) as (fmodel, diffopt):
    
                # Inner Loop更新
                for _ in range(inner_steps):
                    support_loss = criterion(fmodel(x_support), y_support)
                    diffopt.step(support_loss)
    
                # クエリセットで評価
                query_pred = fmodel(x_query)
                query_loss = criterion(query_pred, y_query)
    
                meta_loss += query_loss
    
        # タスク数で平均
        meta_loss = meta_loss / len(tasks)
    
        return meta_loss
    
    # Outer Loop の動作確認
    print("\n=== Outer Loop テスト ===")
    test_tasks = []
    for _ in range(4):
        x_s, y_s, _, _ = generate_sinusoid_task(n_samples=5)
        x_q, y_q, _, _ = generate_sinusoid_task(n_samples=10)
        test_tasks.append((x_s, y_s, x_q, y_q))
    
    meta_loss = outer_loop(model, test_tasks)
    print(f"Meta loss: {meta_loss.item():.4f}")
    

### エピソード学習ループ
    
    
    def train_maml(model, n_iterations=10000, tasks_per_batch=4,
                   k_shot=5, q_query=10, inner_lr=0.01, outer_lr=0.001,
                   inner_steps=1, eval_interval=500):
        """
        MAML学習ループ
    
        Args:
            model: PyTorchモデル
            n_iterations: 学習イテレーション数
            tasks_per_batch: バッチあたりのタスク数
            k_shot: サポートセットのサンプル数
            q_query: クエリセットのサンプル数
            inner_lr: Inner Loop学習率
            outer_lr: Outer Loop学習率
            inner_steps: Inner Loop更新ステップ数
            eval_interval: 評価間隔
    
        Returns:
            losses: 損失履歴
        """
        meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        criterion = nn.MSELoss()
    
        losses = []
    
        for iteration in range(n_iterations):
            meta_optimizer.zero_grad()
    
            # タスクバッチの生成
            tasks = []
            for _ in range(tasks_per_batch):
                # サポートセット
                x_support, y_support, amp, phase = generate_sinusoid_task(n_samples=k_shot)
                # クエリセット（同じタスクから）
                x_query, y_query, _, _ = generate_sinusoid_task(
                    amplitude=amp, phase=phase, n_samples=q_query
                )
                tasks.append((x_support, y_support, x_query, y_query))
    
            # Outer Loop
            meta_loss = outer_loop(model, tasks, inner_lr, inner_steps)
    
            # メタパラメータ更新
            meta_loss.backward()
            meta_optimizer.step()
    
            losses.append(meta_loss.item())
    
            # 定期的な評価
            if (iteration + 1) % eval_interval == 0:
                print(f"Iteration {iteration+1}/{n_iterations}, Meta Loss: {meta_loss.item():.4f}")
    
        return losses
    
    # MAML学習の実行
    print("\n=== MAML Training ===")
    model = SimpleMLP().to(device)
    
    losses = train_maml(
        model,
        n_iterations=5000,
        tasks_per_batch=4,
        k_shot=5,
        q_query=10,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=1,
        eval_interval=1000
    )
    
    # 損失曲線の可視化
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Meta Loss')
    plt.title('MAML Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    print("\n✓ MAML学習完了")
    print(f"✓ 最終メタ損失: {losses[-1]:.4f}")
    

### 学習済みモデルの評価
    
    
    def evaluate_maml(model, n_test_tasks=5, k_shot=5, inner_lr=0.01, inner_steps=5):
        """
        学習済みMAMLモデルを評価
    
        Args:
            model: 学習済みモデル
            n_test_tasks: テストタスク数
            k_shot: サポートセットのサンプル数
            inner_lr: 適応時の学習率
            inner_steps: 適応ステップ数
        """
        criterion = nn.MSELoss()
    
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
        for idx, ax in enumerate(axes.flat[:n_test_tasks]):
            # 新しいテストタスク生成
            x_support, y_support, amp, phase = generate_sinusoid_task(n_samples=k_shot)
            x_test = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
            y_test = amp * np.sin(x_test.cpu().numpy() + phase)
    
            # 適応前の予測
            with torch.no_grad():
                y_pred_before = model(x_test).cpu().numpy()
    
            # Inner Loop: タスク適応
            adapted_model = SimpleMLP().to(device)
            adapted_model.load_state_dict(model.state_dict())
            optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
            for step in range(inner_steps):
                optimizer.zero_grad()
                loss = criterion(adapted_model(x_support), y_support)
                loss.backward()
                optimizer.step()
    
            # 適応後の予測
            with torch.no_grad():
                y_pred_after = adapted_model(x_test).cpu().numpy()
    
            # 可視化
            ax.scatter(x_support.cpu(), y_support.cpu(),
                      label=f'{k_shot}-shot support', s=80, zorder=3, color='red')
            ax.plot(x_test.cpu(), y_test, 'k--',
                   label='True function', linewidth=2, alpha=0.7)
            ax.plot(x_test.cpu(), y_pred_before, 'b-',
                   label='Before adaptation', alpha=0.5, linewidth=2)
            ax.plot(x_test.cpu(), y_pred_after, 'g-',
                   label=f'After {inner_steps} steps', linewidth=2)
    
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Test Task {idx+1}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
        # 最後のサブプロットを削除
        if n_test_tasks < 6:
            fig.delaxes(axes.flat[5])
    
        plt.tight_layout()
        plt.show()
    
        print("\n=== MAML Evaluation ===")
        print(f"✓ {n_test_tasks}個の新しいタスクでテスト")
        print(f"✓ {k_shot}-shot learning with {inner_steps} gradient steps")
        print("✓ 青線: 適応前（メタ学習の初期化）")
        print("✓ 緑線: 適応後（わずかなデータで学習）")
    
    # 評価の実行
    evaluate_maml(model, n_test_tasks=5, k_shot=5, inner_steps=5)
    

* * *

## 2.4 Reptileアルゴリズム

### MAMLの簡略版

**Reptile** は、OpenAIによって提案されたMAMLの簡易版です。

> 「二次微分を使わずに、一次微分のみでメタ学習を実現」

### MAMLとReptileの違い

項目 | MAML | Reptile  
---|---|---  
**勾配計算** | 二次微分（勾配の勾配） | 一次微分のみ  
**更新式** | $\theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\text{meta}}$ | $\theta \leftarrow \theta + \epsilon (\theta' - \theta)$  
**計算コスト** | 高い | 低い（約70%削減）  
**実装の簡単さ** | 複雑（higherが必要） | シンプル  
**性能** | 理論的に最適 | 実用的に十分  
  
### Reptileの更新式

Reptileは以下のシンプルな更新を行います：

$$ \theta \leftarrow \theta + \epsilon (\theta'_i - \theta) $$

ここで：

  * $\theta$: メタパラメータ
  * $\theta'_i$: タスク $i$ で $K$ ステップ学習後のパラメータ
  * $\epsilon$: メタ学習率

**直感的理解** ：適応後のパラメータ方向にメタパラメータを移動

### Reptileアルゴリズム
    
    
    Algorithm: Reptile
    
    Require: p(T): タスク分布
    Require: α: Inner learning rate
    Require: ε: Meta learning rate (Outer)
    
    1: ランダムに θ を初期化
    2: while not converged do
    3:     T_i ~ p(T)  # タスクをサンプル
    4:     D_i ← Sample data from T_i
    5:
    6:     # タスクで通常の学習
    7:     θ' ← θ
    8:     for k = 1 to K do
    9:         θ' ← θ' - α ∇_{θ'} L_{T_i}(f_{θ'})
    10:    end for
    11:
    12:    # メタパラメータを適応方向に移動
    13:    θ ← θ + ε(θ' - θ)
    14: end while
    15: return θ
    

### Reptile実装
    
    
    def train_reptile(model, n_iterations=5000, k_shot=10,
                      inner_lr=0.01, meta_lr=0.1, inner_steps=5,
                      eval_interval=500):
        """
        Reptile アルゴリズム
    
        Args:
            model: PyTorchモデル
            n_iterations: 学習イテレーション数
            k_shot: タスクごとのサンプル数
            inner_lr: Inner Loop学習率
            meta_lr: メタ学習率
            inner_steps: タスクごとの学習ステップ数
            eval_interval: 評価間隔
    
        Returns:
            losses: 損失履歴
        """
        criterion = nn.MSELoss()
        losses = []
    
        for iteration in range(n_iterations):
            # メタパラメータのコピー
            meta_params = [p.clone() for p in model.parameters()]
    
            # 新しいタスクをサンプル
            x_task, y_task, _, _ = generate_sinusoid_task(n_samples=k_shot)
    
            # タスクで通常の学習（Inner Loop）
            optimizer = optim.SGD(model.parameters(), lr=inner_lr)
    
            for step in range(inner_steps):
                optimizer.zero_grad()
                predictions = model(x_task)
                loss = criterion(predictions, y_task)
                loss.backward()
                optimizer.step()
    
            losses.append(loss.item())
    
            # メタ更新: θ ← θ + ε(θ' - θ)
            with torch.no_grad():
                for meta_param, task_param in zip(meta_params, model.parameters()):
                    meta_param.add_(task_param - meta_param, alpha=meta_lr)
    
                # モデルパラメータを更新
                for param, meta_param in zip(model.parameters(), meta_params):
                    param.copy_(meta_param)
    
            # 定期的な評価
            if (iteration + 1) % eval_interval == 0:
                print(f"Iteration {iteration+1}/{n_iterations}, Loss: {loss.item():.4f}")
    
        return losses
    
    # Reptile学習の実行
    print("\n=== Reptile Training ===")
    reptile_model = SimpleMLP().to(device)
    
    reptile_losses = train_reptile(
        reptile_model,
        n_iterations=5000,
        k_shot=10,
        inner_lr=0.01,
        meta_lr=0.1,
        inner_steps=5,
        eval_interval=1000
    )
    
    # 損失曲線の可視化
    plt.figure(figsize=(10, 5))
    plt.plot(reptile_losses, alpha=0.7, color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Task Loss')
    plt.title('Reptile Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    print("\n✓ Reptile学習完了")
    print(f"✓ 最終損失: {reptile_losses[-1]:.4f}")
    

### MAMLとReptileの比較
    
    
    # MAMLとReptileの性能比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 損失曲線の比較
    ax1 = axes[0]
    ax1.plot(losses, label='MAML', alpha=0.7, linewidth=2)
    ax1.plot(reptile_losses, label='Reptile', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 適応速度の比較
    ax2 = axes[1]
    
    # テストタスク生成
    x_support, y_support, amp, phase = generate_sinusoid_task(n_samples=5)
    x_test = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
    y_test = amp * np.sin(x_test.cpu().numpy() + phase)
    
    # MAML適応
    maml_errors = []
    adapted_maml = SimpleMLP().to(device)
    adapted_maml.load_state_dict(model.state_dict())
    optimizer_maml = optim.SGD(adapted_maml.parameters(), lr=0.01)
    
    for step in range(10):
        with torch.no_grad():
            pred = adapted_maml(x_test)
            error = nn.MSELoss()(pred, torch.FloatTensor(y_test).to(device))
            maml_errors.append(error.item())
    
        optimizer_maml.zero_grad()
        loss = nn.MSELoss()(adapted_maml(x_support), y_support)
        loss.backward()
        optimizer_maml.step()
    
    # Reptile適応
    reptile_errors = []
    adapted_reptile = SimpleMLP().to(device)
    adapted_reptile.load_state_dict(reptile_model.state_dict())
    optimizer_reptile = optim.SGD(adapted_reptile.parameters(), lr=0.01)
    
    for step in range(10):
        with torch.no_grad():
            pred = adapted_reptile(x_test)
            error = nn.MSELoss()(pred, torch.FloatTensor(y_test).to(device))
            reptile_errors.append(error.item())
    
        optimizer_reptile.zero_grad()
        loss = nn.MSELoss()(adapted_reptile(x_support), y_support)
        loss.backward()
        optimizer_reptile.step()
    
    ax2.plot(maml_errors, 'o-', label='MAML', linewidth=2, markersize=6)
    ax2.plot(reptile_errors, 's-', label='Reptile', linewidth=2, markersize=6)
    ax2.set_xlabel('Adaptation Step')
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Adaptation Speed on New Task', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== MAML vs Reptile ===")
    print(f"MAML - 初期誤差: {maml_errors[0]:.4f}, 最終誤差: {maml_errors[-1]:.4f}")
    print(f"Reptile - 初期誤差: {reptile_errors[0]:.4f}, 最終誤差: {reptile_errors[-1]:.4f}")
    print("\n✓ 両手法とも高速適応が可能")
    print("✓ MAMLはわずかに良い初期化を提供")
    print("✓ Reptileは実装がシンプルで計算効率が高い")
    

* * *

## 2.5 実践: Omniglot Few-Shot分類

### Omniglotデータセット

**Omniglot** は、Few-Shot学習のベンチマークとして広く使われるデータセットです。

  * 50種類の言語から1,623種類の文字
  * 各文字につき20枚の手書き画像
  * 「MNISTの転置版」と呼ばれる

### 5-way 1-shot タスク

**タスク設定** ：

  * **5-way** : 5クラス分類
  * **1-shot** : 各クラス1枚のサンプルのみ
  * **目標** : サポートセット5枚から学習し、クエリセットを分類

### データ準備
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    import os
    
    # Omniglot用の畳み込みネットワーク
    class OmniglotCNN(nn.Module):
        """Omniglot用の4層CNN"""
        def __init__(self, n_way=5):
            super(OmniglotCNN, self).__init__()
            self.features = nn.Sequential(
                # Layer 1
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
    
                # Layer 2
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
    
                # Layer 3
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
    
                # Layer 4
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
    
            self.classifier = nn.Linear(64, n_way)
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # モデルのインスタンス化
    omniglot_model = OmniglotCNN(n_way=5).to(device)
    print(f"\n=== Omniglot Model ===")
    print(f"Parameters: {sum(p.numel() for p in omniglot_model.parameters()):,}")
    print(f"Architecture: 4-layer CNN + Linear classifier")
    

### Few-Shotタスク生成
    
    
    class OmniglotTaskGenerator:
        """Omniglot用のFew-Shotタスク生成"""
    
        def __init__(self, n_way=5, k_shot=1, q_query=15):
            """
            Args:
                n_way: クラス数
                k_shot: サポートセットのサンプル数
                q_query: クエリセットのサンプル数
            """
            self.n_way = n_way
            self.k_shot = k_shot
            self.q_query = q_query
    
            # ダミーデータ生成（実際にはOmniglotデータセットを使用）
            # ここでは28x28の画像を生成
            self.n_classes = 100  # 簡略化のため100クラス
            self.images_per_class = 20
    
        def generate_task(self):
            """
            N-way K-shot タスクを生成
    
            Returns:
                support_x, support_y, query_x, query_y
            """
            # ランダムにN個のクラスを選択
            selected_classes = np.random.choice(
                self.n_classes, self.n_way, replace=False
            )
    
            support_x, support_y = [], []
            query_x, query_y = [], []
    
            for class_idx, class_id in enumerate(selected_classes):
                # 各クラスから画像をサンプル
                n_samples = self.k_shot + self.q_query
    
                # ダミー画像生成（実際にはデータセットから読み込み）
                images = torch.randn(n_samples, 1, 28, 28)
    
                # サポートセット
                support_x.append(images[:self.k_shot])
                support_y.extend([class_idx] * self.k_shot)
    
                # クエリセット
                query_x.append(images[self.k_shot:])
                query_y.extend([class_idx] * self.q_query)
    
            # テンソルに変換
            support_x = torch.cat(support_x, dim=0).to(device)
            support_y = torch.LongTensor(support_y).to(device)
            query_x = torch.cat(query_x, dim=0).to(device)
            query_y = torch.LongTensor(query_y).to(device)
    
            return support_x, support_y, query_x, query_y
    
    # タスク生成器のテスト
    task_gen = OmniglotTaskGenerator(n_way=5, k_shot=1, q_query=15)
    sup_x, sup_y, qry_x, qry_y = task_gen.generate_task()
    
    print(f"\n=== Task Generation ===")
    print(f"Support set: {sup_x.shape}, labels: {sup_y.shape}")
    print(f"Query set: {qry_x.shape}, labels: {qry_y.shape}")
    print(f"Support labels: {sup_y.cpu().numpy()}")
    print(f"Query labels distribution: {np.bincount(qry_y.cpu().numpy())}")
    

### MAMLとReptileの比較実験
    
    
    def train_meta_learning(model, algorithm='maml', n_iterations=1000,
                           n_way=5, k_shot=1, q_query=15,
                           inner_lr=0.01, outer_lr=0.001, inner_steps=5,
                           eval_interval=100):
        """
        メタ学習トレーニング（MAMLまたはReptile）
    
        Args:
            model: PyTorchモデル
            algorithm: 'maml' または 'reptile'
            n_iterations: 学習イテレーション数
            n_way: N-way分類
            k_shot: K-shot学習
            q_query: クエリセットサイズ
            inner_lr: Inner Loop学習率
            outer_lr: Outer Loop学習率
            inner_steps: Inner Loop更新ステップ数
            eval_interval: 評価間隔
    
        Returns:
            train_accs, val_accs: 精度履歴
        """
        task_gen = OmniglotTaskGenerator(n_way=n_way, k_shot=k_shot, q_query=q_query)
        criterion = nn.CrossEntropyLoss()
    
        if algorithm == 'maml':
            meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    
        train_accs = []
    
        for iteration in range(n_iterations):
            # タスク生成
            support_x, support_y, query_x, query_y = task_gen.generate_task()
    
            if algorithm == 'maml':
                # MAML更新
                meta_optimizer.zero_grad()
    
                with higher.innerloop_ctx(
                    model,
                    optim.SGD(model.parameters(), lr=inner_lr),
                    copy_initial_weights=False
                ) as (fmodel, diffopt):
    
                    # Inner Loop
                    for _ in range(inner_steps):
                        support_loss = criterion(fmodel(support_x), support_y)
                        diffopt.step(support_loss)
    
                    # Query loss
                    query_pred = fmodel(query_x)
                    query_loss = criterion(query_pred, query_y)
    
                    # 精度計算
                    accuracy = (query_pred.argmax(1) == query_y).float().mean()
                    train_accs.append(accuracy.item())
    
                    # Outer Loop
                    query_loss.backward()
                    meta_optimizer.step()
    
            elif algorithm == 'reptile':
                # Reptile更新
                meta_params = [p.clone() for p in model.parameters()]
    
                # Inner Loop
                optimizer = optim.SGD(model.parameters(), lr=inner_lr)
                for _ in range(inner_steps):
                    optimizer.zero_grad()
                    loss = criterion(model(support_x), support_y)
                    loss.backward()
                    optimizer.step()
    
                # 精度計算
                with torch.no_grad():
                    query_pred = model(query_x)
                    accuracy = (query_pred.argmax(1) == query_y).float().mean()
                    train_accs.append(accuracy.item())
    
                # メタ更新
                with torch.no_grad():
                    for meta_param, task_param in zip(meta_params, model.parameters()):
                        meta_param.add_(task_param - meta_param, alpha=outer_lr)
    
                    for param, meta_param in zip(model.parameters(), meta_params):
                        param.copy_(meta_param)
    
            # 定期的な評価
            if (iteration + 1) % eval_interval == 0:
                avg_acc = np.mean(train_accs[-eval_interval:])
                print(f"{algorithm.upper()} - Iter {iteration+1}/{n_iterations}, "
                      f"Avg Accuracy: {avg_acc:.3f}")
    
        return train_accs
    
    # MAMLで学習
    print("\n=== Training MAML on Omniglot ===")
    maml_model = OmniglotCNN(n_way=5).to(device)
    maml_accs = train_meta_learning(
        maml_model,
        algorithm='maml',
        n_iterations=1000,
        n_way=5,
        k_shot=1,
        q_query=15,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        eval_interval=200
    )
    
    # Reptileで学習
    print("\n=== Training Reptile on Omniglot ===")
    reptile_model_omniglot = OmniglotCNN(n_way=5).to(device)
    reptile_accs = train_meta_learning(
        reptile_model_omniglot,
        algorithm='reptile',
        n_iterations=1000,
        n_way=5,
        k_shot=1,
        q_query=15,
        inner_lr=0.01,
        outer_lr=0.1,
        inner_steps=5,
        eval_interval=200
    )
    

### 収束性と精度評価
    
    
    # 結果の可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 精度曲線
    ax1 = axes[0]
    window = 50
    maml_smooth = np.convolve(maml_accs, np.ones(window)/window, mode='valid')
    reptile_smooth = np.convolve(reptile_accs, np.ones(window)/window, mode='valid')
    
    ax1.plot(maml_smooth, label='MAML', linewidth=2, alpha=0.8)
    ax1.plot(reptile_smooth, label='Reptile', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Query Accuracy')
    ax1.set_title('5-way 1-shot Learning Curve (Smoothed)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 最終性能の比較
    ax2 = axes[1]
    final_window = 100
    maml_final = np.mean(maml_accs[-final_window:])
    reptile_final = np.mean(reptile_accs[-final_window:])
    
    methods = ['MAML', 'Reptile']
    accuracies = [maml_final, reptile_final]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax2.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Final Accuracy')
    ax2.set_title('Final Performance Comparison', fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Final Results ===")
    print(f"MAML - Final Accuracy: {maml_final:.3f} ± {np.std(maml_accs[-final_window:]):.3f}")
    print(f"Reptile - Final Accuracy: {reptile_final:.3f} ± {np.std(reptile_accs[-final_window:]):.3f}")
    print(f"\n✓ 5-way 1-shot classification on Omniglot")
    print(f"✓ Random baseline: 20% (1/5)")
    print(f"✓ Both methods significantly outperform random")
    
    # 統計的比較
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(maml_accs[-final_window:],
                                        reptile_accs[-final_window:])
    print(f"\n統計的検定 (t-test):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.3f}")
    if p_value < 0.05:
        print(f"  → 有意差あり（p < 0.05）")
    else:
        print(f"  → 有意差なし（p >= 0.05）")
    

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **MAMLの原理**

     * 少数データからの高速適応を実現する初期パラメータを学習
     * 二段階最適化: Inner Loop（タスク適応）とOuter Loop（メタ学習）
     * 勾配の勾配（二次微分）により強力な適応能力を獲得
  2. **MAMLアルゴリズム**

     * Inner Loop: $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta)$
     * Outer Loop: $\theta \leftarrow \theta - \beta \nabla_\theta \sum \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i})$
     * FOMAML: 一次微分のみで効率化
  3. **PyTorch実装**

     * higherライブラリで二次微分を簡単に実装
     * エピソード学習ループの構築
     * 正弦波回帰タスクでの動作確認
  4. **Reptileアルゴリズム**

     * 一次微分のみでメタ学習を実現
     * 更新式: $\theta \leftarrow \theta + \epsilon (\theta' - \theta)$
     * 実装がシンプルで計算効率が高い
  5. **Omniglot実験**

     * 5-way 1-shot分類タスク
     * MAMLとReptileの性能比較
     * 両手法ともランダムベースラインを大きく上回る

### MAML vs Reptile まとめ

項目 | MAML | Reptile  
---|---|---  
**理論的基盤** | 二次微分による最適化 | 一次微分の方向へ移動  
**計算コスト** | 高い（二次微分） | 低い（一次微分のみ）  
**メモリ使用量** | 多い | 少ない  
**実装の複雑さ** | 複雑（higherが必要） | シンプル  
**性能** | 理論的に最適 | 実用的に十分  
**適用範囲** | 任意のモデル | 任意のモデル  
**推奨用途** | 最高性能が必要な場合 | 効率重視の場合  
  
### 実践的ガイドライン

状況 | 推奨手法 | 理由  
---|---|---  
研究・ベンチマーク | MAML | 最高性能を追求  
プロトタイピング | Reptile | 実装が簡単  
計算資源制約 | FOMAML/Reptile | 効率的  
大規模モデル | Reptile | メモリ効率  
少数ステップ適応 | MAML | より良い初期化  
  
### 次の章へ

第3章では、**Prototypical Networks** を学びます：

  * 埋め込み空間でのプロトタイプ学習
  * 距離ベースの分類
  * 実装とMAMLとの比較

* * *

## 演習問題

### 問題1（難易度：medium）

MAMLとReptileの更新式の違いを数式で説明し、なぜReptileの方が計算効率が良いのか述べてください。

解答例

**解答** ：

**MAML更新式** ：

$$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

ここで、$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta)$ なので、$\theta'_i$ は $\theta$ の関数です。

したがって、$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i})$ を計算するには**連鎖律** が必要：

$$ \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{test} \cdot \nabla_\theta \theta'_i $$

この $\nabla_\theta \theta'_i$ の計算が**二次微分** であり、計算コストが高い。

**Reptile更新式** ：

$$ \theta \leftarrow \theta + \epsilon (\theta'_i - \theta) $$

この式は**単純な重み付き平均** であり、微分計算は不要です。

**計算効率の違い** ：

項目 | MAML | Reptile  
---|---|---  
勾配計算 | $\nabla_\theta \nabla_{\theta'} \mathcal{L}$（二次） | $\nabla_\theta \mathcal{L}$（一次のみ）  
計算グラフ | Inner Loopの履歴を保持 | 不要  
メモリ使用量 | 高い（中間勾配を保存） | 低い  
計算時間 | 約2倍 | 基準  
  
**結論** ：Reptileは二次微分を計算せず、単純なパラメータ更新のみなので、約50-70%の計算コスト削減を実現します。

### 問題2（難易度：hard）

以下のコードは、MAMLのInner Loopを実装しようとしていますが、誤りがあります。問題点を指摘し、正しいコードを書いてください。
    
    
    # 誤ったコード
    def wrong_inner_loop(model, x_support, y_support, inner_lr=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=inner_lr)
    
        optimizer.zero_grad()
        predictions = model(x_support)
        loss = criterion(predictions, y_support)
        loss.backward()
        optimizer.step()
    
        return loss
    

解答例

**問題点** ：

  1. **計算グラフが切断される** : `optimizer.step()` を使うと、パラメータが更新されますが、Outer Loopで必要な二次微分のための計算グラフが保持されません。
  2. **higherライブラリ未使用** : MAMLでは、Inner Loopの更新を追跡し、Outer Loopで二次微分を計算する必要があります。

**正しい実装** ：
    
    
    import higher
    
    def correct_inner_loop(model, x_support, y_support, x_query, y_query,
                           inner_lr=0.01, inner_steps=1):
        """
        MAMLのInner Loop（正しい実装）
    
        Args:
            model: PyTorchモデル
            x_support: サポートセット入力
            y_support: サポートセット出力
            x_query: クエリセット入力
            y_query: クエリセット出力
            inner_lr: Inner Loop学習率
            inner_steps: 適応ステップ数
    
        Returns:
            query_loss: クエリセット損失（勾配が追跡される）
        """
        criterion = nn.MSELoss()
    
        # higherを使用してInner Loopを実装
        with higher.innerloop_ctx(
            model,
            optim.SGD(model.parameters(), lr=inner_lr),
            copy_initial_weights=False,
            track_higher_grads=True  # 二次微分を追跡
        ) as (fmodel, diffopt):
    
            # Inner Loop: タスク適応
            for _ in range(inner_steps):
                support_pred = fmodel(x_support)
                support_loss = criterion(support_pred, y_support)
                diffopt.step(support_loss)
    
            # クエリセットで評価（勾配が追跡される）
            query_pred = fmodel(x_query)
            query_loss = criterion(query_pred, y_query)
    
        return query_loss
    
    # 使用例
    model = SimpleMLP().to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # タスク生成
    x_sup, y_sup, amp, phase = generate_sinusoid_task(n_samples=5)
    x_qry, y_qry, _, _ = generate_sinusoid_task(
        amplitude=amp, phase=phase, n_samples=10
    )
    
    # MAML更新
    meta_optimizer.zero_grad()
    query_loss = correct_inner_loop(model, x_sup, y_sup, x_qry, y_qry)
    query_loss.backward()  # 二次微分が計算される
    meta_optimizer.step()
    
    print(f"✓ Query loss: {query_loss.item():.4f}")
    print("✓ 二次微分が正しく計算されました")
    

**重要なポイント** ：

  * `higher.innerloop_ctx`を使用して、Inner Loopの更新を追跡
  * `track_higher_grads=True`で二次微分を有効化
  * `fmodel`は元のmodelのコピーで、勾配が追跡される
  * Outer Loopで`query_loss.backward()`を呼ぶと、メタパラメータに対する勾配が計算される

### 問題3（難易度：hard）

5-way 5-shot Omniglot分類タスクで、MAMLとReptileの性能を比較する実験を設計してください。以下を含めること：

  * データ分割（訓練/検証/テスト）
  * ハイパーパラメータ設定
  * 評価指標
  * 期待される結果

解答例

**実験設計** ：

**1\. データ分割** ：

  * **訓練用文字** : 1,200文字（メタ訓練）
  * **検証用文字** : 200文字（ハイパーパラメータ調整）
  * **テスト用文字** : 223文字（最終評価）

**2\. ハイパーパラメータ** ：

パラメータ | MAML | Reptile  
---|---|---  
Inner LR (α) | 0.01 | 0.01  
Outer LR (β/ε) | 0.001 | 0.1  
Inner Steps | 5 | 5  
Batch Size | 4 tasks | 1 task  
Iterations | 60,000 | 60,000  
  
**3\. 評価指標** ：

  * **精度** : 正解率（Accuracy）
  * **収束速度** : 目標精度（例: 95%）到達までのイテレーション数
  * **適応速度** : テストタスクでの少数ステップ後の精度向上
  * **計算効率** : イテレーションあたりの実行時間

**実装コード** ：
    
    
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    def comprehensive_comparison(n_iterations=5000, n_way=5, k_shot=5):
        """
        MAMLとReptileの包括的比較実験
        """
        results = {
            'maml': {'train_acc': [], 'val_acc': [], 'time': []},
            'reptile': {'train_acc': [], 'val_acc': [], 'time': []}
        }
    
        # MAML学習
        print("=== Training MAML ===")
        maml_model = OmniglotCNN(n_way=n_way).to(device)
        start_time = time.time()
    
        maml_train_acc = train_meta_learning(
            maml_model, algorithm='maml',
            n_iterations=n_iterations, n_way=n_way, k_shot=k_shot,
            inner_lr=0.01, outer_lr=0.001, inner_steps=5
        )
    
        maml_time = time.time() - start_time
        results['maml']['train_acc'] = maml_train_acc
        results['maml']['time'] = maml_time
    
        # Reptile学習
        print("\n=== Training Reptile ===")
        reptile_model = OmniglotCNN(n_way=n_way).to(device)
        start_time = time.time()
    
        reptile_train_acc = train_meta_learning(
            reptile_model, algorithm='reptile',
            n_iterations=n_iterations, n_way=n_way, k_shot=k_shot,
            inner_lr=0.01, outer_lr=0.1, inner_steps=5
        )
    
        reptile_time = time.time() - start_time
        results['reptile']['train_acc'] = reptile_train_acc
        results['reptile']['time'] = reptile_time
    
        # テストセットでの評価
        print("\n=== Test Set Evaluation ===")
    
        def evaluate_test(model, n_test_tasks=100):
            task_gen = OmniglotTaskGenerator(n_way=n_way, k_shot=k_shot, q_query=15)
            accuracies = []
    
            for _ in range(n_test_tasks):
                support_x, support_y, query_x, query_y = task_gen.generate_task()
    
                # 適応
                optimizer = optim.SGD(model.parameters(), lr=0.01)
                for _ in range(5):
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(support_x), support_y)
                    loss.backward()
                    optimizer.step()
    
                # 評価
                with torch.no_grad():
                    pred = model(query_x)
                    acc = (pred.argmax(1) == query_y).float().mean()
                    accuracies.append(acc.item())
    
            return np.mean(accuracies), np.std(accuracies)
    
        maml_test_acc, maml_test_std = evaluate_test(maml_model)
        reptile_test_acc, reptile_test_std = evaluate_test(reptile_model)
    
        # 結果の可視化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # 学習曲線
        ax1 = axes[0, 0]
        window = 100
        maml_smooth = np.convolve(maml_train_acc, np.ones(window)/window, mode='valid')
        reptile_smooth = np.convolve(reptile_train_acc, np.ones(window)/window, mode='valid')
        ax1.plot(maml_smooth, label='MAML', linewidth=2)
        ax1.plot(reptile_smooth, label='Reptile', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title(f'{n_way}-way {k_shot}-shot Learning Curves', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # テスト精度
        ax2 = axes[0, 1]
        methods = ['MAML', 'Reptile']
        test_accs = [maml_test_acc, reptile_test_acc]
        test_stds = [maml_test_std, reptile_test_std]
        bars = ax2.bar(methods, test_accs, yerr=test_stds,
                       capsize=10, color=['#1f77b4', '#ff7f0e'],
                       alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Test Set Performance', fontsize=13)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
    
        for bar, acc in zip(bars, test_accs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
        # 計算時間
        ax3 = axes[1, 0]
        times = [maml_time, reptile_time]
        bars = ax3.bar(methods, times, color=['#1f77b4', '#ff7f0e'],
                       alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Computational Efficiency', fontsize=13)
        ax3.grid(True, alpha=0.3, axis='y')
    
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
        # サマリーテーブル
        ax4 = axes[1, 1]
        ax4.axis('off')
    
        summary_data = [
            ['Metric', 'MAML', 'Reptile'],
            ['Train Acc (final)', f'{np.mean(maml_train_acc[-100:]):.3f}',
             f'{np.mean(reptile_train_acc[-100:]):.3f}'],
            ['Test Acc', f'{maml_test_acc:.3f}±{maml_test_std:.3f}',
             f'{reptile_test_acc:.3f}±{reptile_test_std:.3f}'],
            ['Time (s)', f'{maml_time:.1f}', f'{reptile_time:.1f}'],
            ['Time per iter (ms)', f'{maml_time/n_iterations*1000:.2f}',
             f'{reptile_time/n_iterations*1000:.2f}']
        ]
    
        table = ax4.table(cellText=summary_data, cellLoc='center',
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#d0d0d0')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
        plt.tight_layout()
        plt.show()
    
        # 結果レポート
        print("\n" + "="*60)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("="*60)
        print(f"\nTask: {n_way}-way {k_shot}-shot classification")
        print(f"Iterations: {n_iterations}")
        print(f"\nMAML:")
        print(f"  Final Train Acc: {np.mean(maml_train_acc[-100:]):.3f}")
        print(f"  Test Acc: {maml_test_acc:.3f} ± {maml_test_std:.3f}")
        print(f"  Training Time: {maml_time:.1f}s ({maml_time/n_iterations*1000:.2f}ms/iter)")
        print(f"\nReptile:")
        print(f"  Final Train Acc: {np.mean(reptile_train_acc[-100:]):.3f}")
        print(f"  Test Acc: {reptile_test_acc:.3f} ± {reptile_test_std:.3f}")
        print(f"  Training Time: {reptile_time:.1f}s ({reptile_time/n_iterations*1000:.2f}ms/iter)")
        print(f"\nSpeedup: {maml_time/reptile_time:.2f}x")
        print("="*60)
    
        return results
    
    # 実験実行
    results = comprehensive_comparison(n_iterations=2000, n_way=5, k_shot=5)
    

**4\. 期待される結果** ：

メトリック | MAML | Reptile  
---|---|---  
テスト精度 | 95-98% | 94-97%  
収束速度 | やや速い | やや遅い  
計算時間 | 基準 | 50-70%削減  
メモリ使用量 | 高い | 低い  
  
**結論** ：

  * MAMLはわずかに高い精度を達成するが、計算コストが高い
  * Reptileは実用的に十分な精度を達成し、効率的
  * 5-shotの場合、差は1-shotより小さくなる傾向
  * 実務ではReptile、研究ではMAMLが推奨される

* * *

## 参考文献

  1. Finn, C., Abbeel, P., & Levine, S. (2017). _Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks_. ICML 2017.
  2. Nichol, A., Achiam, J., & Schulman, J. (2018). _On First-Order Meta-Learning Algorithms_. arXiv preprint arXiv:1803.02999.
  3. Antoniou, A., Edwards, H., & Storkey, A. (2018). _How to train your MAML_. ICLR 2019.
  4. Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015). _Human-level concept learning through probabilistic program induction_. Science, 350(6266), 1332-1338.
  5. Grefenstette, E., et al. (2019). _Higher: A pytorch library for meta-learning_. GitHub repository.
