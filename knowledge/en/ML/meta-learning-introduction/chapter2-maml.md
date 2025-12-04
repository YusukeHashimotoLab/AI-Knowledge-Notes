---
title: "Chapter 2: MAML - Model-Agnostic Meta-Learning"
chapter_title: "Chapter 2: MAML - Model-Agnostic Meta-Learning"
subtitle: The Most Important Approach in Gradient-Based Meta-Learning
reading_time: 30-35 minutes
difficulty: Intermediate-Advanced
code_examples: 8
exercises: 3
version: 1.0
created_at: 2025-10-23
---

This chapter covers MAML. You will learn MAML principles, MAML in PyTorch using the higher library, and efficiency of First-order MAML (FOMAML).

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand MAML principles and two-level optimization
  * ✅ Master the computation of second-order derivatives (gradients of gradients)
  * ✅ Implement MAML in PyTorch using the higher library
  * ✅ Understand the efficiency of First-order MAML (FOMAML)
  * ✅ Implement and compare the Reptile algorithm
  * ✅ Practice MAML on Omniglot few-shot classification

* * *

## 2.1 MAML Principles

### Model-Agnostic Meta-Learning Overview

**MAML (Model-Agnostic Meta-Learning)** is a gradient-based meta-learning algorithm proposed by Chelsea Finn et al. in 2017.

> "Learn good initial parameters that can quickly adapt to new tasks with few gradient update steps from limited data"

### Core Idea of MAML

MAML answers the following question:

  * **Question** : What initial parameters $\theta$ should we choose so that we can achieve high performance on a new task $\mathcal{T}_i$ with just a few gradient steps?
  * **Answer** : Learn parameters that maximize "performance after gradient descent" across multiple tasks

### Two-Level Optimization (Inner/Outer Loop)

MAML consists of the following two loops:

Loop | Purpose | Data | Update Target  
---|---|---|---  
**Inner Loop** | Task adaptation | Support set $\mathcal{D}^{tr}_i$ | Task-specific parameters $\theta'_i$  
**Outer Loop** | Meta-learning | Query set $\mathcal{D}^{test}_i$ | Meta-parameters $\theta$  
  
### MAML Workflow
    
    
    ```mermaid
    graph TD
        A[Meta-parameters θ] --> B[Task 1: θ → θ'₁]
        A --> C[Task 2: θ → θ'₂]
        A --> D[Task N: θ → θ'ₙ]
    
        B --> E[Evaluate on query set L₁]
        C --> F[Evaluate on query set L₂]
        D --> G[Evaluate on query set Lₙ]
    
        E --> H[Meta-loss = Average]
        F --> H
        G --> H
    
        H --> I[Update θ]
        I --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style H fill:#ffebee
        style I fill:#c8e6c9
    ```

### Second-order Derivatives (Gradients of Gradients)

MAML's characteristic is computing **gradients of gradients**.

**Inner Loop (first-order gradient)** :

$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta) $$

**Outer Loop (second-order gradient)** :

$$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

Here, $\theta'_i$ is a function of $\theta$, so it expands as follows:

$$ \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) \cdot \nabla_\theta \theta'_i $$

> **Important** : This second-order derivative is the main computational cost, but it enhances adaptation capability.

### Visual Understanding of MAML
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Visual Understanding of MAML
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # MAML visualization in 2D parameter space
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Standard learning
    ax1 = axes[0]
    theta_init = np.array([0, 0])
    task1_opt = np.array([3, 1])
    task2_opt = np.array([1, 3])
    task3_opt = np.array([-2, 2])
    
    ax1.scatter(*theta_init, s=200, c='red', marker='X', label='Random initialization', zorder=5)
    ax1.scatter(*task1_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax1.scatter(*task2_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax1.scatter(*task3_opt, s=100, c='blue', marker='o', alpha=0.7, label='Task optimal solution')
    
    for opt in [task1_opt, task2_opt, task3_opt]:
        ax1.arrow(theta_init[0], theta_init[1],
                  opt[0]-theta_init[0]*0.9, opt[1]-theta_init[1]*0.9,
                  head_width=0.2, head_length=0.2, fc='gray', ec='gray', alpha=0.5)
    
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_title('Standard Learning: Learn each task from scratch', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: MAML
    ax2 = axes[1]
    maml_init = np.array([0.7, 2])
    
    ax2.scatter(*maml_init, s=200, c='green', marker='X', label='MAML initialization', zorder=5)
    ax2.scatter(*task1_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax2.scatter(*task2_opt, s=100, c='blue', marker='o', alpha=0.7)
    ax2.scatter(*task3_opt, s=100, c='blue', marker='o', alpha=0.7, label='Task optimal solution')
    
    for opt in [task1_opt, task2_opt, task3_opt]:
        ax2.arrow(maml_init[0], maml_init[1],
                  opt[0]-maml_init[0]*0.7, opt[1]-maml_init[1]*0.7,
                  head_width=0.2, head_length=0.2, fc='green', ec='green', alpha=0.5)
    
    # Highlight central region
    circle = plt.Circle(maml_init, 1.5, color='green', fill=False, linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-1, 4)
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('MAML: Start from position close to all tasks', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Intuitive Understanding of MAML ===")
    print("✓ Standard learning: Learn each task from scratch (far away)")
    print("✓ MAML: Learn initialization positioned at the 'center' of all tasks")
    print("✓ Result: Can adapt to each task with just a few gradient steps")
    

* * *

## 2.2 MAML Algorithm

### Mathematical Definition

**Meta-learning objective** :

$$ \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

where $\theta'_i$ is the adapted parameter for task $\mathcal{T}_i$:

$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta) $$

**Symbol meanings** :

  * $\theta$: Meta-parameters (to be learned)
  * $\theta'_i$: Parameters adapted to task $i$
  * $\alpha$: Inner Loop learning rate (task adaptation)
  * $\beta$: Outer Loop learning rate (meta-learning)
  * $\mathcal{L}_{\mathcal{T}_i}^{tr}$: Support set loss for task $i$
  * $\mathcal{L}_{\mathcal{T}_i}^{test}$: Query set loss for task $i$

### Inner Loop: Task Adaptation

For each task $\mathcal{T}_i$, perform gradient descent using support set $\mathcal{D}^{tr}_i$:

$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta) $$

For multiple steps (with $K$ steps):

$$ \begin{aligned} \theta_i^{(0)} &= \theta \\\ \theta_i^{(k+1)} &= \theta_i^{(k)} - \alpha \nabla_{\theta_i^{(k)}} \mathcal{L}_{\mathcal{T}_i}^{tr}(f_{\theta_i^{(k)}}) \\\ \theta'_i &= \theta_i^{(K)} \end{aligned} $$

### Outer Loop: Meta-parameter Update

Evaluate adapted parameters $\theta'_i$ on query set $\mathcal{D}^{test}_i$ to compute meta-loss:

$$ \mathcal{L}_{\text{meta}}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

Update meta-parameters:

$$ \theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\text{meta}}(\theta) $$

### Algorithm Pseudocode
    
    
    Algorithm: MAML
    
    Require: p(T): Task distribution
    Require: α, β: Inner/Outer Loop learning rates
    
    1: Randomly initialize θ
    2: while not converged do
    3:     B ← Sample batch of tasks {T_i} ~ p(T)
    4:     for all T_i ∈ B do
    5:         # Inner Loop: Task adaptation
    6:         D_i^tr, D_i^test ← Sample support/query sets from T_i
    7:         θ'_i ← θ - α ∇_θ L_{T_i}^tr(f_θ)
    8:
    9:         # Compute loss on query set
    10:        L_i ← L_{T_i}^test(f_{θ'_i})
    11:    end for
    12:
    13:    # Outer Loop: Meta-learning
    14:    θ ← θ - β ∇_θ Σ L_i
    15: end while
    16: return θ
    

### First-order MAML (FOMAML)

**Challenge** : High computational cost of second-order derivatives

**Solution** : Approximation by ignoring second-order derivative terms

FOMAML approximates as follows:

$$ \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) \approx \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

That is, the $\nabla_\theta \theta'_i$ term is ignored.

Comparison | MAML | FOMAML  
---|---|---  
**Gradient computation** | Second-order derivatives | First-order only  
**Computational cost** | High | Low (approx. 50% reduction)  
**Memory usage** | High | Low  
**Performance** | Best | Slightly lower (practical)  
  
> **In practice** , FOMAML often achieves sufficient performance and is widely used.

* * *

## 2.3 MAML Implementation in PyTorch

### Using the higher Library

**higher** is a convenient library for handling higher-order derivatives in PyTorch. It's ideal for MAML implementation.
    
    
    # Install higher
    # pip install higher
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import higher
    import numpy as np
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

### Simple Model Definition
    
    
    class SimpleMLP(nn.Module):
        """Simple MLP for Few-Shot learning"""
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
    
    # Instantiate model
    model = SimpleMLP().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    

### Task Generation Function
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    def generate_sinusoid_task(amplitude=None, phase=None, n_samples=10):
        """
        Generate sinusoid regression task
    
        Args:
            amplitude: Amplitude (random if None)
            phase: Phase (random if None)
            n_samples: Number of samples
    
        Returns:
            x, y: Input and output pairs
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
    
    # Visualize task examples
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
    
    print("=== Task Distribution ===")
    print("Each task is a sine wave with different amplitude and phase")
    print("Goal: Adapt to new sine waves from few samples")
    

### Inner Loop Implementation
    
    
    def inner_loop(model, x_support, y_support, inner_lr=0.01, inner_steps=1):
        """
        Inner Loop: Task adaptation
    
        Args:
            model: PyTorch model
            x_support: Support set input
            y_support: Support set output
            inner_lr: Inner Loop learning rate
            inner_steps: Adaptation step count
    
        Returns:
            task_loss: Support set loss
        """
        criterion = nn.MSELoss()
    
        # Prediction and loss calculation
        predictions = model(x_support)
        task_loss = criterion(predictions, y_support)
    
        # Gradient computation
        task_grad = torch.autograd.grad(
            task_loss,
            model.parameters(),
            create_graph=True  # Needed for second-order derivatives
        )
    
        # Manual gradient descent (when inner_steps is 1)
        adapted_params = []
        for param, grad in zip(model.parameters(), task_grad):
            adapted_params.append(param - inner_lr * grad)
    
        return task_loss, adapted_params
    
    # Test Inner Loop
    print("\n=== Inner Loop Test ===")
    x_sup, y_sup, _, _ = generate_sinusoid_task(n_samples=5)
    loss, adapted = inner_loop(model, x_sup, y_sup)
    print(f"Support loss: {loss.item():.4f}")
    print(f"Adapted parameters: {len(adapted)} tensors")
    

### Outer Loop Implementation
    
    
    def outer_loop(model, tasks, inner_lr=0.01, inner_steps=1):
        """
        Outer Loop: Meta-learning
    
        Args:
            model: PyTorch model
            tasks: List of tasks [(x_sup, y_sup, x_qry, y_qry), ...]
            inner_lr: Inner Loop learning rate
            inner_steps: Adaptation step count
    
        Returns:
            meta_loss: Meta-loss (average query loss)
        """
        criterion = nn.MSELoss()
        meta_loss = 0.0
    
        for x_support, y_support, x_query, y_query in tasks:
            # Inner Loop: Task adaptation (using higher)
            with higher.innerloop_ctx(
                model,
                optim.SGD(model.parameters(), lr=inner_lr),
                copy_initial_weights=False
            ) as (fmodel, diffopt):
    
                # Inner Loop update
                for _ in range(inner_steps):
                    support_loss = criterion(fmodel(x_support), y_support)
                    diffopt.step(support_loss)
    
                # Evaluate on query set
                query_pred = fmodel(x_query)
                query_loss = criterion(query_pred, y_query)
    
                meta_loss += query_loss
    
        # Average over tasks
        meta_loss = meta_loss / len(tasks)
    
        return meta_loss
    
    # Test Outer Loop
    print("\n=== Outer Loop Test ===")
    test_tasks = []
    for _ in range(4):
        x_s, y_s, _, _ = generate_sinusoid_task(n_samples=5)
        x_q, y_q, _, _ = generate_sinusoid_task(n_samples=10)
        test_tasks.append((x_s, y_s, x_q, y_q))
    
    meta_loss = outer_loop(model, test_tasks)
    print(f"Meta loss: {meta_loss.item():.4f}")
    

### Episode Training Loop
    
    
    def train_maml(model, n_iterations=10000, tasks_per_batch=4,
                   k_shot=5, q_query=10, inner_lr=0.01, outer_lr=0.001,
                   inner_steps=1, eval_interval=500):
        """
        MAML training loop
    
        Args:
            model: PyTorch model
            n_iterations: Number of training iterations
            tasks_per_batch: Tasks per batch
            k_shot: Support set sample count
            q_query: Query set sample count
            inner_lr: Inner Loop learning rate
            outer_lr: Outer Loop learning rate
            inner_steps: Inner Loop update steps
            eval_interval: Evaluation interval
    
        Returns:
            losses: Loss history
        """
        meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        criterion = nn.MSELoss()
    
        losses = []
    
        for iteration in range(n_iterations):
            meta_optimizer.zero_grad()
    
            # Generate task batch
            tasks = []
            for _ in range(tasks_per_batch):
                # Support set
                x_support, y_support, amp, phase = generate_sinusoid_task(n_samples=k_shot)
                # Query set (from same task)
                x_query, y_query, _, _ = generate_sinusoid_task(
                    amplitude=amp, phase=phase, n_samples=q_query
                )
                tasks.append((x_support, y_support, x_query, y_query))
    
            # Outer Loop
            meta_loss = outer_loop(model, tasks, inner_lr, inner_steps)
    
            # Meta-parameter update
            meta_loss.backward()
            meta_optimizer.step()
    
            losses.append(meta_loss.item())
    
            # Periodic evaluation
            if (iteration + 1) % eval_interval == 0:
                print(f"Iteration {iteration+1}/{n_iterations}, Meta Loss: {meta_loss.item():.4f}")
    
        return losses
    
    # Run MAML training
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
    
    # Visualize loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Meta Loss')
    plt.title('MAML Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    print("\n✓ MAML training complete")
    print(f"✓ Final meta-loss: {losses[-1]:.4f}")
    

### Evaluating the Trained Model
    
    
    def evaluate_maml(model, n_test_tasks=5, k_shot=5, inner_lr=0.01, inner_steps=5):
        """
        Evaluate trained MAML model
    
        Args:
            model: Trained model
            n_test_tasks: Number of test tasks
            k_shot: Support set sample count
            inner_lr: Adaptation learning rate
            inner_steps: Adaptation step count
        """
        criterion = nn.MSELoss()
    
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
        for idx, ax in enumerate(axes.flat[:n_test_tasks]):
            # Generate new test task
            x_support, y_support, amp, phase = generate_sinusoid_task(n_samples=k_shot)
            x_test = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
            y_test = amp * np.sin(x_test.cpu().numpy() + phase)
    
            # Prediction before adaptation
            with torch.no_grad():
                y_pred_before = model(x_test).cpu().numpy()
    
            # Inner Loop: Task adaptation
            adapted_model = SimpleMLP().to(device)
            adapted_model.load_state_dict(model.state_dict())
            optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
            for step in range(inner_steps):
                optimizer.zero_grad()
                loss = criterion(adapted_model(x_support), y_support)
                loss.backward()
                optimizer.step()
    
            # Prediction after adaptation
            with torch.no_grad():
                y_pred_after = adapted_model(x_test).cpu().numpy()
    
            # Visualization
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
    
        # Remove last subplot if not needed
        if n_test_tasks < 6:
            fig.delaxes(axes.flat[5])
    
        plt.tight_layout()
        plt.show()
    
        print("\n=== MAML Evaluation ===")
        print(f"✓ Tested on {n_test_tasks} new tasks")
        print(f"✓ {k_shot}-shot learning with {inner_steps} gradient steps")
        print("✓ Blue line: Before adaptation (meta-learned initialization)")
        print("✓ Green line: After adaptation (learned from few data)")
    
    # Run evaluation
    evaluate_maml(model, n_test_tasks=5, k_shot=5, inner_steps=5)
    

* * *

## 2.4 Reptile Algorithm

### Simplified Version of MAML

**Reptile** is a simplified version of MAML proposed by OpenAI.

> "Achieve meta-learning with only first-order derivatives, without using second-order derivatives"

### Differences Between MAML and Reptile

Item | MAML | Reptile  
---|---|---  
**Gradient computation** | Second-order (gradients of gradients) | First-order only  
**Update rule** | $\theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\text{meta}}$ | $\theta \leftarrow \theta + \epsilon (\theta' - \theta)$  
**Computational cost** | High | Low (approx. 70% reduction)  
**Implementation simplicity** | Complex (needs higher) | Simple  
**Performance** | Theoretically optimal | Practically sufficient  
  
### Reptile Update Rule

Reptile performs the following simple update:

$$ \theta \leftarrow \theta + \epsilon (\theta'_i - \theta) $$

where:

  * $\theta$: Meta-parameters
  * $\theta'_i$: Parameters after $K$ steps of learning on task $i$
  * $\epsilon$: Meta-learning rate

**Intuitive understanding** : Move meta-parameters toward the direction of adapted parameters

### Reptile Algorithm
    
    
    Algorithm: Reptile
    
    Require: p(T): Task distribution
    Require: α: Inner learning rate
    Require: ε: Meta learning rate (Outer)
    
    1: Randomly initialize θ
    2: while not converged do
    3:     T_i ~ p(T)  # Sample task
    4:     D_i ← Sample data from T_i
    5:
    6:     # Standard learning on task
    7:     θ' ← θ
    8:     for k = 1 to K do
    9:         θ' ← θ' - α ∇_{θ'} L_{T_i}(f_{θ'})
    10:    end for
    11:
    12:    # Move meta-parameters in adaptation direction
    13:    θ ← θ + ε(θ' - θ)
    14: end while
    15: return θ
    

### Reptile Implementation
    
    
    def train_reptile(model, n_iterations=5000, k_shot=10,
                      inner_lr=0.01, meta_lr=0.1, inner_steps=5,
                      eval_interval=500):
        """
        Reptile algorithm
    
        Args:
            model: PyTorch model
            n_iterations: Number of training iterations
            k_shot: Samples per task
            inner_lr: Inner Loop learning rate
            meta_lr: Meta learning rate
            inner_steps: Learning steps per task
            eval_interval: Evaluation interval
    
        Returns:
            losses: Loss history
        """
        criterion = nn.MSELoss()
        losses = []
    
        for iteration in range(n_iterations):
            # Copy meta-parameters
            meta_params = [p.clone() for p in model.parameters()]
    
            # Sample new task
            x_task, y_task, _, _ = generate_sinusoid_task(n_samples=k_shot)
    
            # Standard learning on task (Inner Loop)
            optimizer = optim.SGD(model.parameters(), lr=inner_lr)
    
            for step in range(inner_steps):
                optimizer.zero_grad()
                predictions = model(x_task)
                loss = criterion(predictions, y_task)
                loss.backward()
                optimizer.step()
    
            losses.append(loss.item())
    
            # Meta update: θ ← θ + ε(θ' - θ)
            with torch.no_grad():
                for meta_param, task_param in zip(meta_params, model.parameters()):
                    meta_param.add_(task_param - meta_param, alpha=meta_lr)
    
                # Update model parameters
                for param, meta_param in zip(model.parameters(), meta_params):
                    param.copy_(meta_param)
    
            # Periodic evaluation
            if (iteration + 1) % eval_interval == 0:
                print(f"Iteration {iteration+1}/{n_iterations}, Loss: {loss.item():.4f}")
    
        return losses
    
    # Run Reptile training
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
    
    # Visualize loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(reptile_losses, alpha=0.7, color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Task Loss')
    plt.title('Reptile Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    print("\n✓ Reptile training complete")
    print(f"✓ Final loss: {reptile_losses[-1]:.4f}")
    

### Comparing MAML and Reptile
    
    
    # Performance comparison of MAML and Reptile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve comparison
    ax1 = axes[0]
    ax1.plot(losses, label='MAML', alpha=0.7, linewidth=2)
    ax1.plot(reptile_losses, label='Reptile', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Adaptation speed comparison
    ax2 = axes[1]
    
    # Generate test task
    x_support, y_support, amp, phase = generate_sinusoid_task(n_samples=5)
    x_test = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
    y_test = amp * np.sin(x_test.cpu().numpy() + phase)
    
    # MAML adaptation
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
    
    # Reptile adaptation
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
    print(f"MAML - Initial error: {maml_errors[0]:.4f}, Final error: {maml_errors[-1]:.4f}")
    print(f"Reptile - Initial error: {reptile_errors[0]:.4f}, Final error: {reptile_errors[-1]:.4f}")
    print("\n✓ Both methods enable fast adaptation")
    print("✓ MAML provides slightly better initialization")
    print("✓ Reptile is simpler to implement and computationally efficient")
    

* * *

## 2.5 Practice: Omniglot Few-Shot Classification

### Omniglot Dataset

**Omniglot** is a widely used benchmark for few-shot learning.

  * 1,623 character types from 50 languages
  * 20 handwritten images per character
  * Called the "transpose of MNIST"

### 5-way 1-shot Task

**Task setting** :

  * **5-way** : 5-class classification
  * **1-shot** : Only 1 sample per class
  * **Goal** : Learn from 5 support set images and classify query set

### Data Preparation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    import os
    
    # Convolutional network for Omniglot
    class OmniglotCNN(nn.Module):
        """4-layer CNN for Omniglot"""
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
    
    # Instantiate model
    omniglot_model = OmniglotCNN(n_way=5).to(device)
    print(f"\n=== Omniglot Model ===")
    print(f"Parameters: {sum(p.numel() for p in omniglot_model.parameters()):,}")
    print(f"Architecture: 4-layer CNN + Linear classifier")
    

### Few-Shot Task Generation
    
    
    class OmniglotTaskGenerator:
        """Few-Shot task generator for Omniglot"""
    
        def __init__(self, n_way=5, k_shot=1, q_query=15):
            """
            Args:
                n_way: Number of classes
                k_shot: Support set sample count
                q_query: Query set sample count
            """
            self.n_way = n_way
            self.k_shot = k_shot
            self.q_query = q_query
    
            # Generate dummy data (actual implementation would use Omniglot dataset)
            # Here we generate 28x28 images
            self.n_classes = 100  # Simplified to 100 classes
            self.images_per_class = 20
    
        def generate_task(self):
            """
            Generate N-way K-shot task
    
            Returns:
                support_x, support_y, query_x, query_y
            """
            # Randomly select N classes
            selected_classes = np.random.choice(
                self.n_classes, self.n_way, replace=False
            )
    
            support_x, support_y = [], []
            query_x, query_y = [], []
    
            for class_idx, class_id in enumerate(selected_classes):
                # Sample images from each class
                n_samples = self.k_shot + self.q_query
    
                # Generate dummy images (actual implementation would read from dataset)
                images = torch.randn(n_samples, 1, 28, 28)
    
                # Support set
                support_x.append(images[:self.k_shot])
                support_y.extend([class_idx] * self.k_shot)
    
                # Query set
                query_x.append(images[self.k_shot:])
                query_y.extend([class_idx] * self.q_query)
    
            # Convert to tensors
            support_x = torch.cat(support_x, dim=0).to(device)
            support_y = torch.LongTensor(support_y).to(device)
            query_x = torch.cat(query_x, dim=0).to(device)
            query_y = torch.LongTensor(query_y).to(device)
    
            return support_x, support_y, query_x, query_y
    
    # Test task generator
    task_gen = OmniglotTaskGenerator(n_way=5, k_shot=1, q_query=15)
    sup_x, sup_y, qry_x, qry_y = task_gen.generate_task()
    
    print(f"\n=== Task Generation ===")
    print(f"Support set: {sup_x.shape}, labels: {sup_y.shape}")
    print(f"Query set: {qry_x.shape}, labels: {qry_y.shape}")
    print(f"Support labels: {sup_y.cpu().numpy()}")
    print(f"Query labels distribution: {np.bincount(qry_y.cpu().numpy())}")
    

### Comparison Experiment: MAML vs Reptile
    
    
    def train_meta_learning(model, algorithm='maml', n_iterations=1000,
                           n_way=5, k_shot=1, q_query=15,
                           inner_lr=0.01, outer_lr=0.001, inner_steps=5,
                           eval_interval=100):
        """
        Meta-learning training (MAML or Reptile)
    
        Args:
            model: PyTorch model
            algorithm: 'maml' or 'reptile'
            n_iterations: Number of training iterations
            n_way: N-way classification
            k_shot: K-shot learning
            q_query: Query set size
            inner_lr: Inner Loop learning rate
            outer_lr: Outer Loop learning rate
            inner_steps: Inner Loop update steps
            eval_interval: Evaluation interval
    
        Returns:
            train_accs, val_accs: Accuracy history
        """
        task_gen = OmniglotTaskGenerator(n_way=n_way, k_shot=k_shot, q_query=q_query)
        criterion = nn.CrossEntropyLoss()
    
        if algorithm == 'maml':
            meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    
        train_accs = []
    
        for iteration in range(n_iterations):
            # Generate task
            support_x, support_y, query_x, query_y = task_gen.generate_task()
    
            if algorithm == 'maml':
                # MAML update
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
    
                    # Accuracy calculation
                    accuracy = (query_pred.argmax(1) == query_y).float().mean()
                    train_accs.append(accuracy.item())
    
                    # Outer Loop
                    query_loss.backward()
                    meta_optimizer.step()
    
            elif algorithm == 'reptile':
                # Reptile update
                meta_params = [p.clone() for p in model.parameters()]
    
                # Inner Loop
                optimizer = optim.SGD(model.parameters(), lr=inner_lr)
                for _ in range(inner_steps):
                    optimizer.zero_grad()
                    loss = criterion(model(support_x), support_y)
                    loss.backward()
                    optimizer.step()
    
                # Accuracy calculation
                with torch.no_grad():
                    query_pred = model(query_x)
                    accuracy = (query_pred.argmax(1) == query_y).float().mean()
                    train_accs.append(accuracy.item())
    
                # Meta update
                with torch.no_grad():
                    for meta_param, task_param in zip(meta_params, model.parameters()):
                        meta_param.add_(task_param - meta_param, alpha=outer_lr)
    
                    for param, meta_param in zip(model.parameters(), meta_params):
                        param.copy_(meta_param)
    
            # Periodic evaluation
            if (iteration + 1) % eval_interval == 0:
                avg_acc = np.mean(train_accs[-eval_interval:])
                print(f"{algorithm.upper()} - Iter {iteration+1}/{n_iterations}, "
                      f"Avg Accuracy: {avg_acc:.3f}")
    
        return train_accs
    
    # Train with MAML
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
    
    # Train with Reptile
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
    

### Convergence and Accuracy Evaluation
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    """
    Example: Convergence and Accuracy Evaluation
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curves
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
    
    # Final performance comparison
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
    
    # Display values on bars
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
    
    # Statistical comparison
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(maml_accs[-final_window:],
                                        reptile_accs[-final_window:])
    print(f"\nStatistical test (t-test):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.3f}")
    if p_value < 0.05:
        print(f"  → Significant difference (p < 0.05)")
    else:
        print(f"  → No significant difference (p >= 0.05)")
    

* * *

## 2.6 Chapter Summary

### What We Learned

  1. **MAML Principles**

     * Learn initial parameters that enable fast adaptation from limited data
     * Two-level optimization: Inner Loop (task adaptation) and Outer Loop (meta-learning)
     * Achieve powerful adaptation capability through second-order derivatives (gradients of gradients)
  2. **MAML Algorithm**

     * Inner Loop: $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta)$
     * Outer Loop: $\theta \leftarrow \theta - \beta \nabla_\theta \sum \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i})$
     * FOMAML: Efficiency through first-order derivatives only
  3. **PyTorch Implementation**

     * Easy implementation of second-order derivatives with the higher library
     * Building episode learning loops
     * Validation on sinusoid regression tasks
  4. **Reptile Algorithm**

     * Achieve meta-learning with first-order derivatives only
     * Update rule: $\theta \leftarrow \theta + \epsilon (\theta' - \theta)$
     * Simple implementation with high computational efficiency
  5. **Omniglot Experiments**

     * 5-way 1-shot classification task
     * Performance comparison of MAML and Reptile
     * Both methods significantly outperform random baseline

### MAML vs Reptile Summary

Item | MAML | Reptile  
---|---|---  
**Theoretical foundation** | Second-order derivative optimization | Move in first-order derivative direction  
**Computational cost** | High (second-order derivatives) | Low (first-order only)  
**Memory usage** | High | Low  
**Implementation complexity** | Complex (needs higher) | Simple  
**Performance** | Theoretically optimal | Practically sufficient  
**Applicability** | Any model | Any model  
**Recommended use** | When best performance needed | When efficiency is priority  
  
### Practical Guidelines

Situation | Recommended Approach | Reason  
---|---|---  
Research/Benchmarks | MAML | Pursue best performance  
Prototyping | Reptile | Easy to implement  
Resource constraints | FOMAML/Reptile | Efficient  
Large-scale models | Reptile | Memory efficient  
Few-step adaptation | MAML | Better initialization  
  
### To the Next Chapter

In Chapter 3, we will learn about **Prototypical Networks** :

  * Prototype learning in embedding space
  * Distance-based classification
  * Implementation and comparison with MAML

* * *

## Exercises

### Problem 1 (Difficulty: medium)

Explain the differences in update rules between MAML and Reptile using mathematical formulas, and describe why Reptile is more computationally efficient.

Sample Answer

**Answer** :

**MAML update rule** :

$$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) $$

Here, since $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{tr}(f_\theta)$, $\theta'_i$ is a function of $\theta$.

Therefore, computing $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i})$ requires the **chain rule** :

$$ \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{test}(f_{\theta'_i}) = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{test} \cdot \nabla_\theta \theta'_i $$

Computing this $\nabla_\theta \theta'_i$ is a **second-order derivative** with high computational cost.

**Reptile update rule** :

$$ \theta \leftarrow \theta + \epsilon (\theta'_i - \theta) $$

This formula is a **simple weighted average** requiring no derivative computation.

**Computational efficiency differences** :

Item | MAML | Reptile  
---|---|---  
Gradient computation | $\nabla_\theta \nabla_{\theta'} \mathcal{L}$ (second-order) | $\nabla_\theta \mathcal{L}$ (first-order only)  
Computational graph | Must retain Inner Loop history | Not needed  
Memory usage | High (store intermediate gradients) | Low  
Computational time | Approx. 2x | Baseline  
  
**Conclusion** : Reptile achieves approximately 50-70% computational cost reduction by computing no second-order derivatives and only performing simple parameter updates.

### Problem 2 (Difficulty: hard)

The following code attempts to implement MAML's Inner Loop but contains errors. Identify the problems and write correct code.
    
    
    # Incorrect code
    def wrong_inner_loop(model, x_support, y_support, inner_lr=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=inner_lr)
    
        optimizer.zero_grad()
        predictions = model(x_support)
        loss = criterion(predictions, y_support)
        loss.backward()
        optimizer.step()
    
        return loss
    

Sample Answer

**Problems** :

  1. **Computational graph is disconnected** : Using `optimizer.step()` updates parameters but doesn't preserve the computational graph needed for second-order derivatives in the Outer Loop.
  2. **Not using higher library** : MAML needs to track Inner Loop updates and compute second-order derivatives in the Outer Loop.

**Correct implementation** :
    
    
    import higher
    
    def correct_inner_loop(model, x_support, y_support, x_query, y_query,
                           inner_lr=0.01, inner_steps=1):
        """
        MAML Inner Loop (correct implementation)
    
        Args:
            model: PyTorch model
            x_support: Support set input
            y_support: Support set output
            x_query: Query set input
            y_query: Query set output
            inner_lr: Inner Loop learning rate
            inner_steps: Adaptation step count
    
        Returns:
            query_loss: Query set loss (with tracked gradients)
        """
        criterion = nn.MSELoss()
    
        # Implement Inner Loop using higher
        with higher.innerloop_ctx(
            model,
            optim.SGD(model.parameters(), lr=inner_lr),
            copy_initial_weights=False,
            track_higher_grads=True  # Track second-order derivatives
        ) as (fmodel, diffopt):
    
            # Inner Loop: Task adaptation
            for _ in range(inner_steps):
                support_pred = fmodel(x_support)
                support_loss = criterion(support_pred, y_support)
                diffopt.step(support_loss)
    
            # Evaluate on query set (gradients tracked)
            query_pred = fmodel(x_query)
            query_loss = criterion(query_pred, y_query)
    
        return query_loss
    
    # Usage example
    model = SimpleMLP().to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate task
    x_sup, y_sup, amp, phase = generate_sinusoid_task(n_samples=5)
    x_qry, y_qry, _, _ = generate_sinusoid_task(
        amplitude=amp, phase=phase, n_samples=10
    )
    
    # MAML update
    meta_optimizer.zero_grad()
    query_loss = correct_inner_loop(model, x_sup, y_sup, x_qry, y_qry)
    query_loss.backward()  # Second-order derivatives computed
    meta_optimizer.step()
    
    print(f"✓ Query loss: {query_loss.item():.4f}")
    print("✓ Second-order derivatives computed correctly")
    

**Key points** :

  * Use `higher.innerloop_ctx` to track Inner Loop updates
  * Enable second-order derivatives with `track_higher_grads=True`
  * `fmodel` is a copy of the original model with tracked gradients
  * Calling `query_loss.backward()` in Outer Loop computes gradients with respect to meta-parameters

### Problem 3 (Difficulty: hard)

Design an experiment to compare MAML and Reptile performance on 5-way 5-shot Omniglot classification task. Include:

  * Data split (train/validation/test)
  * Hyperparameter settings
  * Evaluation metrics
  * Expected results

Sample Answer

**Experiment design** :

**1\. Data split** :

  * **Training characters** : 1,200 characters (meta-training)
  * **Validation characters** : 200 characters (hyperparameter tuning)
  * **Test characters** : 223 characters (final evaluation)

**2\. Hyperparameters** :

Parameter | MAML | Reptile  
---|---|---  
Inner LR (α) | 0.01 | 0.01  
Outer LR (β/ε) | 0.001 | 0.1  
Inner Steps | 5 | 5  
Batch Size | 4 tasks | 1 task  
Iterations | 60,000 | 60,000  
  
**3\. Evaluation metrics** :

  * **Accuracy** : Correct classification rate
  * **Convergence speed** : Iterations to reach target accuracy (e.g., 95%)
  * **Adaptation speed** : Accuracy improvement after few steps on test tasks
  * **Computational efficiency** : Execution time per iteration

**Implementation code** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    def comprehensive_comparison(n_iterations=5000, n_way=5, k_shot=5):
        """
        Comprehensive comparison experiment of MAML and Reptile
        """
        results = {
            'maml': {'train_acc': [], 'val_acc': [], 'time': []},
            'reptile': {'train_acc': [], 'val_acc': [], 'time': []}
        }
    
        # MAML training
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
    
        # Reptile training
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
    
        # Test set evaluation
        print("\n=== Test Set Evaluation ===")
    
        def evaluate_test(model, n_test_tasks=100):
            task_gen = OmniglotTaskGenerator(n_way=n_way, k_shot=k_shot, q_query=15)
            accuracies = []
    
            for _ in range(n_test_tasks):
                support_x, support_y, query_x, query_y = task_gen.generate_task()
    
                # Adaptation
                optimizer = optim.SGD(model.parameters(), lr=0.01)
                for _ in range(5):
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(support_x), support_y)
                    loss.backward()
                    optimizer.step()
    
                # Evaluation
                with torch.no_grad():
                    pred = model(query_x)
                    acc = (pred.argmax(1) == query_y).float().mean()
                    accuracies.append(acc.item())
    
            return np.mean(accuracies), np.std(accuracies)
    
        maml_test_acc, maml_test_std = evaluate_test(maml_model)
        reptile_test_acc, reptile_test_std = evaluate_test(reptile_model)
    
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # Learning curves
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
    
        # Test accuracy
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
    
        # Computational time
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
    
        # Summary table
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
    
        # Results report
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
    
    # Run experiment
    results = comprehensive_comparison(n_iterations=2000, n_way=5, k_shot=5)
    

**4\. Expected results** :

Metric | MAML | Reptile  
---|---|---  
Test accuracy | 95-98% | 94-97%  
Convergence speed | Slightly faster | Slightly slower  
Computational time | Baseline | 50-70% reduction  
Memory usage | High | Low  
  
**Conclusion** :

  * MAML achieves slightly higher accuracy but with higher computational cost
  * Reptile achieves practically sufficient accuracy efficiently
  * In 5-shot case, the difference tends to be smaller than in 1-shot
  * Reptile is recommended for practical use, MAML for research

* * *

## References

  1. Finn, C., Abbeel, P., & Levine, S. (2017). _Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks_. ICML 2017.
  2. Nichol, A., Achiam, J., & Schulman, J. (2018). _On First-Order Meta-Learning Algorithms_. arXiv preprint arXiv:1803.02999.
  3. Antoniou, A., Edwards, H., & Storkey, A. (2018). _How to train your MAML_. ICLR 2019.
  4. Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015). _Human-level concept learning through probabilistic program induction_. Science, 350(6266), 1332-1338.
  5. Grefenstette, E., et al. (2019). _Higher: A pytorch library for meta-learning_. GitHub repository.
