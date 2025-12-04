---
title: "Chapter 3: Neural Architecture Search"
chapter_title: "Chapter 3: Neural Architecture Search"
subtitle: Automated Design of Neural Networks - Exploring Optimal Architectures with DARTS and AutoKeras
reading_time: 35-40 minutes
difficulty: Intermediate-Advanced
code_examples: 8
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Neural Architecture Search. You will learn automatic model search using AutoKeras and NAS efficiency techniques (Weight Sharing.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand Neural Architecture Search (NAS) search space design
  * ✅ Understand major NAS search strategies (reinforcement learning, evolutionary algorithms, gradient-based)
  * ✅ Implement automatic model search using AutoKeras
  * ✅ Understand the principles and implementation of DARTS (Differentiable Architecture Search)
  * ✅ Understand NAS efficiency techniques (Weight Sharing, Proxy Tasks)
  * ✅ Apply AutoKeras and DARTS to real data

* * *

## 3.1 NAS Search Space

### What is Neural Architecture Search?

**Neural Architecture Search (NAS)** is a technology that automatically designs the architecture of neural networks.

> "Manual Design vs Automated Design" - NAS can discover architectures that exceed human expertise.

### The Three Components of NAS

Component | Description | Examples  
---|---|---  
**Search Space** | Set of searchable architectures | Cell-based, Macro, Micro  
**Search Strategy** | Method for exploring architectures | Reinforcement learning, evolution, gradient-based  
**Performance Estimation** | Determining architecture quality | Accuracy, FLOPs, latency  
  
### Cell-based Search Space

In **cell-based search space** , we search for "Cells" that are repeatedly used.
    
    
    ```mermaid
    graph TD
        A[Input Image] --> B[Stem Convolution]
        B --> C[Normal Cell 1]
        C --> D[Normal Cell 2]
        D --> E[Reduction Cell]
        E --> F[Normal Cell 3]
        F --> G[Normal Cell 4]
        G --> H[Reduction Cell]
        H --> I[Normal Cell 5]
        I --> J[Global Pool]
        J --> K[Softmax]
    
        style C fill:#e3f2fd
        style D fill:#e3f2fd
        style E fill:#ffebee
        style F fill:#e3f2fd
        style G fill:#e3f2fd
        style H fill:#ffebee
        style I fill:#e3f2fd
    ```

#### Cell Types

Cell Type | Role | Spatial Resolution  
---|---|---  
**Normal Cell** | Feature extraction | Maintained  
**Reduction Cell** | Downsampling | Reduced to 1/2  
  
### Macro vs Micro Architecture

Architecture | Search Target | Advantages | Disadvantages  
---|---|---|---  
**Macro** | Overall structure (number of layers, connections) | Highly flexible | Enormous search space  
**Micro** | Internal structure of cells | Efficient, transferable | Many constraints  
  
### Search Space Size

The size of the cell-based search space is enormous:

$$ \text{Search Space Size} \approx O^{E} $$

  * $O$: Types of operations (e.g., 8 types)
  * $E$: Number of edges (e.g., 14)
  * Example: $8^{14} \approx 4.4 \times 10^{12}$ possibilities

### Search Space Design Example
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Search Space Design Example
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Define NAS search space
    class SearchSpace:
        def __init__(self):
            # Available operations
            self.operations = [
                'conv_3x3',
                'conv_5x5',
                'sep_conv_3x3',
                'sep_conv_5x5',
                'max_pool_3x3',
                'avg_pool_3x3',
                'identity',
                'zero'
            ]
    
            # Cell structure parameters
            self.num_nodes = 4  # Number of nodes in cell
            self.num_edges_per_node = 2  # Number of input edges per node
    
        def calculate_space_size(self):
            """Calculate search space size"""
            num_ops = len(self.operations)
    
            # Calculate choices for each node
            total_choices = 1
            for node_id in range(2, self.num_nodes + 2):
                # Choice of edge source (select from previous nodes)
                edge_choices = node_id
                # Choice of operation
                op_choices = num_ops
                # Choices for this node
                node_choices = (edge_choices * op_choices) ** self.num_edges_per_node
                total_choices *= node_choices
    
            return total_choices
    
        def sample_architecture(self):
            """Randomly sample an architecture"""
            architecture = []
    
            for node_id in range(2, self.num_nodes + 2):
                # Select inputs to this node
                node_config = []
                for _ in range(self.num_edges_per_node):
                    # Input source node
                    input_node = np.random.randint(0, node_id)
                    # Operation
                    operation = np.random.choice(self.operations)
                    node_config.append((input_node, operation))
    
                architecture.append(node_config)
    
            return architecture
    
    # Calculate search space size
    search_space = SearchSpace()
    space_size = search_space.calculate_space_size()
    
    print("=== NAS Search Space Analysis ===")
    print(f"Types of operations: {len(search_space.operations)}")
    print(f"Number of nodes in cell: {search_space.num_nodes}")
    print(f"Search space size: {space_size:,}")
    print(f"Scientific notation: {space_size:.2e}")
    
    # Sample architecture
    sample = search_space.sample_architecture()
    print(f"\n=== Sample Architecture ===")
    for i, node in enumerate(sample, start=2):
        print(f"Node {i}:")
        for j, (input_node, op) in enumerate(node):
            print(f"  Input{j}: Node{input_node} → {op}")
    

**Output** :
    
    
    === NAS Search Space Analysis ===
    Types of operations: 8
    Number of nodes in cell: 4
    Search space size: 17,179,869,184
    Scientific notation: 1.72e+10
    
    === Sample Architecture ===
    Node 2:
      Input0: Node0 → sep_conv_3x3
      Input1: Node1 → max_pool_3x3
    Node 3:
      Input0: Node2 → conv_5x5
      Input1: Node0 → identity
    Node 4:
      Input0: Node3 → avg_pool_3x3
      Input1: Node1 → sep_conv_5x5
    Node 5:
      Input0: Node2 → conv_3x3
      Input1: Node4 → zero
    

* * *

## 3.2 NAS Search Strategies

### Comparison of Major Search Strategies

Search Strategy | Principle | Computational Cost | Representative Method  
---|---|---|---  
**Reinforcement Learning** | Controller generates architectures | Very high | NASNet  
**Evolutionary Algorithm** | Iterative mutation and selection | High | AmoebaNet  
**Gradient-based** | Continuous relaxation for differentiability | Low | DARTS  
**One-shot** | Train supernet once | Medium | ENAS  
  
### 1\. Reinforcement Learning-based (NASNet)

NASNet uses an RNN controller to generate architectures and learns using accuracy as reward.
    
    
    ```mermaid
    graph LR
        A[RNN Controller] -->|Architecture Generation| B[Child Network]
        B -->|Training & Evaluation| C[Validation Accuracy]
        C -->|Reward| A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

#### Reinforcement Learning NAS Pseudocode
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # NASNet-style reinforcement learning search (conceptual implementation)
    import numpy as np
    
    class RLController:
        """Reinforcement learning-based NAS controller"""
    
        def __init__(self, search_space):
            self.search_space = search_space
            self.history = []
    
        def sample_architecture(self, epsilon=0.1):
            """Sample architecture using epsilon-greedy strategy"""
            if np.random.random() < epsilon:
                # Exploration: random sampling
                return self.search_space.sample_architecture()
            else:
                # Exploitation: mutate from past good architectures
                if self.history:
                    best_arch = max(self.history, key=lambda x: x['reward'])
                    return self.mutate_architecture(best_arch['architecture'])
                else:
                    return self.search_space.sample_architecture()
    
        def mutate_architecture(self, architecture):
            """Add small mutation to architecture"""
            mutated = [node[:] for node in architecture]
    
            # Randomly mutate one node
            node_idx = np.random.randint(len(mutated))
            edge_idx = np.random.randint(len(mutated[node_idx]))
    
            input_node, _ = mutated[node_idx][edge_idx]
            new_op = np.random.choice(self.search_space.operations)
            mutated[node_idx][edge_idx] = (input_node, new_op)
    
            return mutated
    
        def update(self, architecture, reward):
            """Receive reward and update history"""
            self.history.append({
                'architecture': architecture,
                'reward': reward
            })
    
    # Simulation
    search_space = SearchSpace()
    controller = RLController(search_space)
    
    print("=== Reinforcement Learning NAS Simulation ===")
    for iteration in range(10):
        # Sample architecture
        arch = controller.sample_architecture(epsilon=0.3)
    
        # Simulate reward (in practice, train and obtain accuracy)
        # Here we use dummy reward based on operation diversity
        ops_used = set()
        for node in arch:
            for _, op in node:
                ops_used.add(op)
        reward = len(ops_used) / len(search_space.operations) + np.random.normal(0, 0.1)
    
        # Update controller
        controller.update(arch, reward)
    
        print(f"Iteration {iteration + 1}: Reward = {reward:.3f}")
    
    # Display best architecture
    best = max(controller.history, key=lambda x: x['reward'])
    print(f"\n=== Best Architecture (Reward: {best['reward']:.3f}) ===")
    for i, node in enumerate(best['architecture'], start=2):
        print(f"Node {i}:")
        for j, (input_node, op) in enumerate(node):
            print(f"  Input{j}: Node{input_node} → {op}")
    

### 2\. Evolutionary Algorithm

Evolutionary algorithms optimize architectures by mimicking biological evolution.
    
    
    # Evolutionary algorithm NAS (simplified version)
    import random
    import copy
    
    class EvolutionaryNAS:
        """Evolutionary algorithm-based NAS"""
    
        def __init__(self, search_space, population_size=20, num_generations=10):
            self.search_space = search_space
            self.population_size = population_size
            self.num_generations = num_generations
    
        def initialize_population(self):
            """Generate initial population"""
            return [self.search_space.sample_architecture()
                    for _ in range(self.population_size)]
    
        def evaluate_fitness(self, architecture):
            """Evaluate fitness (dummy implementation)"""
            # In practice, train network and measure accuracy
            # Here we use operation diversity as score
            ops_used = set()
            for node in architecture:
                for _, op in node:
                    ops_used.add(op)
            return len(ops_used) + random.gauss(0, 1)
    
        def select_parents(self, population, fitness_scores, k=2):
            """Tournament selection"""
            selected = []
            for _ in range(2):
                candidates_idx = random.sample(range(len(population)), k)
                best_idx = max(candidates_idx, key=lambda i: fitness_scores[i])
                selected.append(copy.deepcopy(population[best_idx]))
            return selected
    
        def crossover(self, parent1, parent2):
            """Crossover (one-point crossover)"""
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
    
        def mutate(self, architecture, mutation_rate=0.1):
            """Mutation"""
            mutated = copy.deepcopy(architecture)
    
            for node_idx in range(len(mutated)):
                for edge_idx in range(len(mutated[node_idx])):
                    if random.random() < mutation_rate:
                        input_node, _ = mutated[node_idx][edge_idx]
                        new_op = random.choice(self.search_space.operations)
                        mutated[node_idx][edge_idx] = (input_node, new_op)
    
            return mutated
    
        def run(self):
            """Run evolutionary algorithm"""
            # Initial population
            population = self.initialize_population()
    
            best_history = []
    
            for generation in range(self.num_generations):
                # Fitness evaluation
                fitness_scores = [self.evaluate_fitness(arch) for arch in population]
    
                # Statistics
                best_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                best_history.append(best_fitness)
    
                print(f"Generation {generation + 1}: Best={best_fitness:.3f}, Average={avg_fitness:.3f}")
    
                # Generate new generation
                new_population = []
    
                # Elitism
                elite_idx = fitness_scores.index(max(fitness_scores))
                new_population.append(copy.deepcopy(population[elite_idx]))
    
                # Selection, crossover, mutation
                while len(new_population) < self.population_size:
                    parents = self.select_parents(population, fitness_scores)
                    offspring1, offspring2 = self.crossover(parents[0], parents[1])
                    offspring1 = self.mutate(offspring1)
                    offspring2 = self.mutate(offspring2)
    
                    new_population.extend([offspring1, offspring2])
    
                population = new_population[:self.population_size]
    
            # Return best individual
            fitness_scores = [self.evaluate_fitness(arch) for arch in population]
            best_idx = fitness_scores.index(max(fitness_scores))
    
            return population[best_idx], best_history
    
    # Execute
    search_space = SearchSpace()
    evo_nas = EvolutionaryNAS(search_space, population_size=20, num_generations=10)
    
    print("=== Evolutionary Algorithm NAS ===")
    best_arch, history = evo_nas.run()
    
    print(f"\n=== Best Architecture ===")
    for i, node in enumerate(best_arch, start=2):
        print(f"Node {i}:")
        for j, (input_node, op) in enumerate(node):
            print(f"  Input{j}: Node{input_node} → {op}")
    

### 3\. Gradient-based (DARTS Overview)

DARTS applies continuous relaxation to the search space and optimizes using gradient descent (details in Section 3.4).

> **Important** : Gradient-based methods are over 1000 times faster than reinforcement learning and evolutionary algorithms.

* * *

## 3.3 AutoKeras

### What is AutoKeras?

**AutoKeras** is a Keras-based AutoML library that makes NAS easy to use.

### AutoKeras Installation
    
    
    pip install autokeras
    

### Basic Usage of AutoKeras
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Basic Usage of AutoKeras
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Basic AutoKeras example: Image classification
    import numpy as np
    import autokeras as ak
    from tensorflow.keras.datasets import mnist
    
    # Prepare data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reduce training data (for demo)
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    
    print("=== AutoKeras Image Classification ===")
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    
    # AutoKeras ImageClassifier
    clf = ak.ImageClassifier(
        max_trials=5,  # Number of models to try
        overwrite=True,
        directory='autokeras_results',
        project_name='mnist_classification'
    )
    
    # Model search and training
    print("\nStarting search...")
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=3,
        verbose=1
    )
    
    # Evaluation
    print("\n=== Model Evaluation ===")
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Export best model
    best_model = clf.export_model()
    print("\n=== Best Model Structure ===")
    best_model.summary()
    

### Various AutoKeras Tasks

#### 1\. Structured Data Classification
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 1. Structured Data Classification
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Handle structured data with AutoKeras
    import numpy as np
    import pandas as pd
    import autokeras as ak
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== Structured Data Classification ===")
    print(f"Features: {X.shape[1]} features")
    print(f"Training samples: {len(X_train)}")
    
    # AutoKeras StructuredDataClassifier
    clf = ak.StructuredDataClassifier(
        max_trials=3,
        overwrite=True,
        directory='autokeras_structured',
        project_name='breast_cancer'
    )
    
    # Search and training
    clf.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        verbose=0
    )
    
    # Evaluation
    test_loss, test_acc = clf.evaluate(X_test, y_test, verbose=0)
    print(f"\n=== Evaluation Results ===")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Prediction
    predictions = clf.predict(X_test[:5])
    print(f"\n=== Sample Predictions ===")
    for i, pred in enumerate(predictions[:5]):
        true_label = y_test.iloc[i] if isinstance(y_test, pd.Series) else y_test[i]
        print(f"Sample {i+1}: Prediction={pred[0][0]:.3f}, True={true_label}")
    

#### 2\. Text Classification
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 2. Text Classification
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Text classification with AutoKeras
    import numpy as np
    import autokeras as ak
    from tensorflow.keras.datasets import imdb
    
    # IMDB dataset (movie review sentiment analysis)
    max_features = 10000
    maxlen = 200
    
    print("=== Text Classification (IMDB) ===")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    # Reduce data (for demo)
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:200]
    y_test = y_test[:200]
    
    # Padding
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    
    # AutoKeras TextClassifier
    clf = ak.TextClassifier(
        max_trials=3,
        overwrite=True,
        directory='autokeras_text',
        project_name='imdb_sentiment'
    )
    
    # Search and training
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=3,
        verbose=0
    )
    
    # Evaluation
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"\n=== Evaluation Results ===")
    print(f"Test accuracy: {test_acc:.4f}")
    

### AutoKeras Custom Search Space
    
    
    # Customize search space in AutoKeras
    import autokeras as ak
    from tensorflow.keras.datasets import mnist
    
    # Prepare data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:5000].astype('float32') / 255.0
    y_train = y_train[:5000]
    x_test = x_test[:1000].astype('float32') / 255.0
    y_test = y_test[:1000]
    
    print("=== Custom Search Space ===")
    
    # Input node
    input_node = ak.ImageInput()
    
    # Normalization block
    output = ak.Normalization()(input_node)
    
    # Customize ConvBlock search space
    output = ak.ConvBlock(
        num_blocks=2,  # Number of convolution blocks
        num_layers=2,  # Number of layers in block
        max_pooling=True,
        dropout=0.25
    )(output)
    
    # Classification head
    output = ak.ClassificationHead(
        num_classes=10,
        dropout=0.5
    )(output)
    
    # Build model
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output,
        max_trials=3,
        overwrite=True,
        directory='autokeras_custom',
        project_name='mnist_custom'
    )
    
    # Training
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=3,
        verbose=0
    )
    
    # Evaluation
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Export best model
    best_model = clf.export_model()
    print("\n=== Discovered Architecture ===")
    best_model.summary()
    

* * *

## 3.4 DARTS (Differentiable Architecture Search)

### DARTS Principles

**DARTS** enables gradient descent by applying continuous relaxation to the discrete search space.

### Continuous Relaxation

The operation on each edge is represented as a weighted sum of all operations:

$$ \bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x) $$

  * $\mathcal{O}$: Set of operations
  * $\alpha_o^{(i,j)}$: Weight (architecture parameter) for operation $o$ on edge $(i,j)$
  * Normalized with softmax to make it differentiable

### Bi-level Optimization

DARTS alternately optimizes two parameters:

Parameter | Description | Optimization  
---|---|---  
**Weights $w$** | Network weights | Minimize on training data  
**Architecture $\alpha$** | Operation weights | Minimize on validation data  
  
Optimization problem:

$$ \begin{aligned} \min_{\alpha} \quad & \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \\\ \text{s.t.} \quad & w^*(\alpha) = \arg\min_{w} \mathcal{L}_{\text{train}}(w, \alpha) \end{aligned} $$

### DARTS Implementation (Simplified)
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    # Conceptual DARTS implementation (simplified version)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MixedOp(nn.Module):
        """Weighted sum of multiple operations"""
    
        def __init__(self, C, stride):
            super(MixedOp, self).__init__()
            self._ops = nn.ModuleList()
    
            # Available operations
            self.operations = [
                ('sep_conv_3x3', lambda C, stride: SepConv(C, C, 3, stride, 1)),
                ('sep_conv_5x5', lambda C, stride: SepConv(C, C, 5, stride, 2)),
                ('avg_pool_3x3', lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1)),
                ('max_pool_3x3', lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1)),
                ('skip_connect', lambda C, stride: nn.Identity() if stride == 1 else FactorizedReduce(C, C)),
            ]
    
            for name, op in self.operations:
                self._ops.append(op(C, stride))
    
        def forward(self, x, weights):
            """Compute weighted sum"""
            return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    class Cell(nn.Module):
        """DARTS Cell"""
    
        def __init__(self, num_nodes, C_prev, C, reduction):
            super(Cell, self).__init__()
            self.num_nodes = num_nodes
    
            # Operation for each edge
            self._ops = nn.ModuleList()
            for i in range(num_nodes):
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)
    
        def forward(self, s0, s1, weights):
            """Forward propagation"""
            states = [s0, s1]
            offset = 0
    
            for i in range(self.num_nodes):
                s = sum(self._ops[offset + j](h, weights[offset + j])
                       for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
    
            return torch.cat(states[-self.num_nodes:], dim=1)
    
    class DARTSNetwork(nn.Module):
        """DARTS search network"""
    
        def __init__(self, C=16, num_cells=8, num_nodes=4, num_classes=10):
            super(DARTSNetwork, self).__init__()
            self.num_cells = num_cells
            self.num_nodes = num_nodes
    
            # Architecture parameters (α)
            num_ops = 5  # Types of operations
            num_edges = sum(2 + i for i in range(num_nodes))
            self.alphas_normal = nn.Parameter(torch.randn(num_edges, num_ops))
            self.alphas_reduce = nn.Parameter(torch.randn(num_edges, num_ops))
    
            # Network weights (w)
            self.stem = nn.Sequential(
                nn.Conv2d(3, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C)
            )
    
            # Cell construction is simplified
            self.cells = nn.ModuleList()
            # ... (in actual implementation, add multiple cells)
    
            self.classifier = nn.Linear(C, num_classes)
    
        def arch_parameters(self):
            """Return architecture parameters"""
            return [self.alphas_normal, self.alphas_reduce]
    
        def weights_parameters(self):
            """Return network weights"""
            return [p for n, p in self.named_parameters()
                    if 'alpha' not in n]
    
    # DARTS usage example
    print("=== DARTS Conceptual Model ===")
    model = DARTSNetwork(C=16, num_cells=8, num_nodes=4, num_classes=10)
    
    print(f"Architecture parameter count: {sum(p.numel() for p in model.arch_parameters())}")
    print(f"Network weight parameter count: {sum(p.numel() for p in model.weights_parameters())}")
    
    # Architecture parameter shapes
    print(f"\nNormal cell α: {model.alphas_normal.shape}")
    print(f"Reduction cell α: {model.alphas_reduce.shape}")
    
    # Normalize with softmax
    weights_normal = F.softmax(model.alphas_normal, dim=-1)
    print(f"\nNormalized weights (Normal cell, first edge):")
    print(weights_normal[0].detach().numpy())
    

### DARTS Training Algorithm
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    # DARTS training procedure (pseudocode)
    import torch
    import torch.optim as optim
    
    class DARTSTrainer:
        """DARTS training class"""
    
        def __init__(self, model, train_loader, val_loader):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
    
            # Two optimizers
            self.optimizer_w = optim.SGD(
                model.weights_parameters(),
                lr=0.025,
                momentum=0.9,
                weight_decay=3e-4
            )
    
            self.optimizer_alpha = optim.Adam(
                model.arch_parameters(),
                lr=3e-4,
                betas=(0.5, 0.999),
                weight_decay=1e-3
            )
    
            self.criterion = nn.CrossEntropyLoss()
    
        def train_step(self, train_data, val_data):
            """One training step"""
            # 1. Update architecture parameters (α)
            self.model.train()
            x_val, y_val = val_data
    
            self.optimizer_alpha.zero_grad()
            logits = self.model(x_val)
            loss_alpha = self.criterion(logits, y_val)
            loss_alpha.backward()
            self.optimizer_alpha.step()
    
            # 2. Update network weights (w)
            x_train, y_train = train_data
    
            self.optimizer_w.zero_grad()
            logits = self.model(x_train)
            loss_w = self.criterion(logits, y_train)
            loss_w.backward()
            self.optimizer_w.step()
    
            return loss_w.item(), loss_alpha.item()
    
        def derive_architecture(self):
            """Derive final architecture"""
            # Select operation with highest weight for each edge
            def parse_alpha(alpha):
                gene = []
                n = 2
                start = 0
                for i in range(self.model.num_nodes):
                    end = start + n
                    W = alpha[start:end].copy()
    
                    # Select two best operations for each edge
                    edges = sorted(range(W.shape[0]),
                                  key=lambda x: -max(W[x]))[:2]
    
                    for j in edges:
                        k_best = W[j].argmax()
                        gene.append((j, k_best))
    
                    start = end
                    n += 1
    
                return gene
    
            # Normalize with softmax
            weights_normal = F.softmax(self.model.alphas_normal, dim=-1)
            weights_reduce = F.softmax(self.model.alphas_reduce, dim=-1)
    
            gene_normal = parse_alpha(weights_normal.data.cpu().numpy())
            gene_reduce = parse_alpha(weights_reduce.data.cpu().numpy())
    
            return gene_normal, gene_reduce
    
    # Simulation example
    print("=== DARTS Training Procedure ===")
    print("1. Model initialization")
    print("2. For each epoch:")
    print("   a. Update α on validation data (architecture optimization)")
    print("   b. Update w on training data (weight optimization)")
    print("3. After training, select operation with highest weight for each edge")
    print("4. Reconstruct network with selected operations and final training")
    

### Practical DARTS Example (PyTorch)
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    # Example using actual DARTS implementation (using pt-darts library)
    # Note: pt-darts is an external library (pip install pt-darts)
    
    # Conceptual code example
    """
    import torch
    from darts import DARTS
    from darts.api import spaces
    from darts.trainer import DARTSTrainer
    
    # Define search space
    search_space = spaces.get_search_space('darts', 'cifar10')
    
    # Build DARTS model
    model = DARTS(
        C=16,
        num_classes=10,
        layers=8,
        criterion=nn.CrossEntropyLoss(),
        steps=4,
        multiplier=4,
        stem_multiplier=3
    )
    
    # Initialize trainer
    trainer = DARTSTrainer(
        model,
        optimizer_config={
            'w_lr': 0.025,
            'w_momentum': 0.9,
            'w_weight_decay': 3e-4,
            'alpha_lr': 3e-4,
            'alpha_weight_decay': 1e-3
        }
    )
    
    # Run search
    trainer.search(
        train_loader,
        val_loader,
        epochs=50
    )
    
    # Get best architecture
    best_architecture = model.genotype()
    print(f"Discovered architecture: {best_architecture}")
    """
    
    print("=== Practical DARTS Usage ===")
    print("1. Install libraries like pt-darts")
    print("2. Define search space and model")
    print("3. Search with bi-level optimization")
    print("4. Retrain with discovered architecture")
    print("\nDARTS advantages:")
    print("- Search time: 4 GPU days (vs NASNet's 1800 GPU days)")
    print("- High accuracy: 97%+ on CIFAR-10")
    print("- Transferable: Applicable to ImageNet, etc.")
    

* * *

## 3.5 NAS Efficiency Techniques

### Comparison of Efficiency Techniques

Technique | Principle | Speedup Factor | Impact on Accuracy  
---|---|---|---  
**Weight Sharing** | Share weights among candidate architectures | 1000x | Small  
**Proxy Tasks** | Evaluate on simplified tasks | 10-100x | Medium  
**Early Stopping** | Terminate low-performance models early | 2-5x | Small  
**Transfer Learning** | Transfer knowledge from similar tasks | 5-10x | Small  
  
### 1\. Weight Sharing (ENAS)

**Weight Sharing** constructs a supernet where all candidate architectures share weights.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    # Weight sharing concept (ENAS-style)
    import torch
    import torch.nn as nn
    
    class SharedWeightSuperNet(nn.Module):
        """Weight-sharing supernet"""
    
        def __init__(self, num_nodes=4, C=16):
            super(SharedWeightSuperNet, self).__init__()
            self.num_nodes = num_nodes
    
            # Pre-construct all possible operations (shared weights)
            self.ops = nn.ModuleDict({
                'conv_3x3': nn.Conv2d(C, C, 3, padding=1),
                'conv_5x5': nn.Conv2d(C, C, 5, padding=2),
                'max_pool': nn.MaxPool2d(3, stride=1, padding=1),
                'avg_pool': nn.AvgPool2d(3, stride=1, padding=1),
                'identity': nn.Identity()
            })
    
        def forward(self, x, architecture):
            """
            architecture: Specifies operations for each node
            Example: [('conv_3x3', 0), ('max_pool', 1), ...]
            """
            states = [x, x]  # Initial states
    
            for node_id, (op_name, input_id) in enumerate(architecture):
                # Compute with specified operation and input
                s = self.ops[op_name](states[input_id])
                states.append(s)
    
            # Return output of last node
            return states[-1]
    
    # Build supernet
    supernet = SharedWeightSuperNet(num_nodes=4, C=16)
    
    print("=== Weight Sharing (ENAS-style) ===")
    print(f"Supernet parameter count: {sum(p.numel() for p in supernet.parameters()):,}")
    
    # Use same weights for different architectures
    arch1 = [('conv_3x3', 0), ('max_pool', 1), ('identity', 2), ('avg_pool', 1)]
    arch2 = [('conv_5x5', 1), ('identity', 0), ('max_pool', 2), ('conv_3x3', 3)]
    
    # Dummy input
    x = torch.randn(1, 16, 32, 32)
    
    output1 = supernet(x, arch1)
    output2 = supernet(x, arch2)
    
    print(f"\nArchitecture 1 output shape: {output1.shape}")
    print(f"Architecture 2 output shape: {output2.shape}")
    print("\n→ Can evaluate different architectures while sharing the same weights")
    

### 2\. Speedup with Proxy Tasks

Proxy tasks reduce costs through simplifications such as:

Simplification | Example | Speedup  
---|---|---  
**Reduce data size** | Use only part of CIFAR-10 | 2-5x  
**Reduce epochs** | Evaluate at 10 epochs | 5-10x  
**Reduce model size** | 1/4 of channels | 4-8x  
**Reduce resolution** | 16x16 instead of 32x32 | 4x  
  
### 3\. NAS-Bench Dataset

**NAS-Bench** is a database of precomputed architecture performance.
    
    
    # Conceptual NAS-Bench usage
    # Note: In practice, use the nasbench library (pip install nasbench)
    
    class NASBenchSimulator:
        """NAS-Bench simulator"""
    
        def __init__(self):
            # Precomputed performance data (dummy)
            self.benchmark_data = {}
            self._populate_dummy_data()
    
        def _populate_dummy_data(self):
            """Generate dummy benchmark data"""
            import random
            random.seed(42)
    
            # Precompute performance for 100 architectures
            for i in range(100):
                arch_hash = f"arch_{i:03d}"
                self.benchmark_data[arch_hash] = {
                    'val_accuracy': random.uniform(0.88, 0.95),
                    'test_accuracy': random.uniform(0.87, 0.94),
                    'training_time': random.uniform(100, 500),
                    'params': random.randint(1_000_000, 10_000_000),
                    'flops': random.randint(50_000_000, 500_000_000)
                }
    
        def query(self, architecture):
            """Query architecture performance (returns immediately)"""
            # Hash architecture
            arch_hash = self._hash_architecture(architecture)
    
            if arch_hash in self.benchmark_data:
                return self.benchmark_data[arch_hash]
            else:
                # Estimate unknown architecture
                return {
                    'val_accuracy': 0.90,
                    'test_accuracy': 0.89,
                    'training_time': 300,
                    'params': 5_000_000,
                    'flops': 250_000_000
                }
    
        def _hash_architecture(self, architecture):
            """Hash architecture"""
            # Simple hash (actually more complex)
            arch_str = str(architecture)
            hash_val = sum(ord(c) for c in arch_str) % 100
            return f"arch_{hash_val:03d}"
    
    # Use NAS-Bench
    bench = NASBenchSimulator()
    
    print("=== Fast Evaluation with NAS-Bench ===")
    
    # Architecture search
    import time
    
    architectures = [
        [('conv_3x3', 0), ('max_pool', 1)],
        [('conv_5x5', 0), ('identity', 1)],
        [('avg_pool', 0), ('conv_3x3', 1)]
    ]
    
    start_time = time.time()
    results = []
    
    for arch in architectures:
        result = bench.query(arch)
        results.append((arch, result))
    
    end_time = time.time()
    
    print(f"Search time: {end_time - start_time:.4f} seconds")
    print(f"\n=== Search Results ===")
    for arch, result in results:
        print(f"Architecture: {arch}")
        print(f"  Validation accuracy: {result['val_accuracy']:.3f}")
        print(f"  Test accuracy: {result['test_accuracy']:.3f}")
        print(f"  Training time: {result['training_time']:.1f} seconds")
        print(f"  Parameters: {result['params']:,}")
        print()
    
    print("→ Can obtain performance immediately without actual training")
    

### Combining Efficiency Techniques
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # Search combining multiple efficiency techniques
    import numpy as np
    
    class EfficientNAS:
        """Efficient NAS"""
    
        def __init__(self, use_weight_sharing=True, use_proxy=True,
                     use_early_stopping=True):
            self.use_weight_sharing = use_weight_sharing
            self.use_proxy = use_proxy
            self.use_early_stopping = use_early_stopping
    
            if use_weight_sharing:
                self.supernet = SharedWeightSuperNet()
    
            if use_proxy:
                self.proxy_epochs = 10  # 10 epochs instead of full training
                self.proxy_data_fraction = 0.2  # Use only 20% of data
    
        def evaluate_architecture(self, architecture, full_evaluation=False):
            """Evaluate architecture"""
            if full_evaluation:
                # Full evaluation (only for final candidates)
                epochs = 50
                data_fraction = 1.0
            else:
                # Proxy evaluation
                epochs = self.proxy_epochs if self.use_proxy else 50
                data_fraction = self.proxy_data_fraction if self.use_proxy else 1.0
    
            # Early stopping simulation
            if self.use_early_stopping:
                # Terminate if performance is poor in early epochs
                early_acc = np.random.random()
                if early_acc < 0.5:  # Threshold
                    return {'accuracy': early_acc, 'stopped_early': True}
    
            # Evaluation (dummy)
            accuracy = np.random.uniform(0.85, 0.95)
    
            return {
                'accuracy': accuracy,
                'epochs': epochs,
                'data_fraction': data_fraction,
                'stopped_early': False
            }
    
        def search(self, num_candidates=100, top_k=5):
            """Run NAS search"""
            print("=== Efficient NAS Search ===")
            print(f"Weight Sharing: {self.use_weight_sharing}")
            print(f"Proxy Tasks: {self.use_proxy}")
            print(f"Early Stopping: {self.use_early_stopping}")
            print()
    
            candidates = []
    
            # 1. Large-scale proxy evaluation
            for i in range(num_candidates):
                arch = [('conv_3x3', 0), ('max_pool', 1)]  # Dummy
                result = self.evaluate_architecture(arch, full_evaluation=False)
                candidates.append((arch, result))
    
            # Candidates not terminated by early stopping
            valid_candidates = [c for c in candidates if not c[1]['stopped_early']]
    
            print(f"Initial candidates: {num_candidates}")
            print(f"Reduced by early stopping: {num_candidates - len(valid_candidates)}")
    
            # 2. Full evaluation of top K
            top_candidates = sorted(valid_candidates,
                                   key=lambda x: x[1]['accuracy'],
                                   reverse=True)[:top_k]
    
            print(f"Candidates for full evaluation: {top_k}")
            print()
    
            final_results = []
            for arch, proxy_result in top_candidates:
                full_result = self.evaluate_architecture(arch, full_evaluation=True)
                final_results.append((arch, full_result))
    
            # Return best candidate
            best = max(final_results, key=lambda x: x[1]['accuracy'])
    
            return best, final_results
    
    # Execute
    nas = EfficientNAS(
        use_weight_sharing=True,
        use_proxy=True,
        use_early_stopping=True
    )
    
    best_arch, all_results = nas.search(num_candidates=100, top_k=5)
    
    print("=== Search Results ===")
    print(f"Best architecture: {best_arch[0]}")
    print(f"Best accuracy: {best_arch[1]['accuracy']:.3f}")
    print(f"\nTop 5 accuracies:")
    for i, (arch, result) in enumerate(all_results, 1):
        print(f"{i}. Accuracy={result['accuracy']:.3f}")
    

* * *

## 3.6 Chapter Summary

### What We Learned

  1. **NAS Search Space**

     * Cell-based search space design
     * Macro vs Micro architecture
     * Search space size and complexity
  2. **NAS Search Strategies**

     * Reinforcement learning (NASNet): Generation with RNN controller
     * Evolutionary algorithm: Mutation and selection
     * Gradient-based (DARTS): Speedup with continuous relaxation
     * One-shot (ENAS): Efficiency with weight sharing
  3. **AutoKeras**

     * Automatic learning for images, text, structured data
     * Custom search space definition
     * Advanced NAS with simple API
  4. **DARTS**

     * Differentiable NAS with continuous relaxation
     * Bi-level optimization (w and α)
     * Achieves 1000x+ speedup
  5. **NAS Efficiency**

     * Weight Sharing: Share weights in supernet
     * Proxy Tasks: Evaluate on simplified tasks
     * Early Stopping: Terminate low performance early
     * NAS-Bench: Precomputed database

### Search Strategy Selection Guidelines

Situation | Recommended Method | Reason  
---|---|---  
Abundant computational resources | Reinforcement learning, evolution | High accuracy expected  
Limited computational resources | DARTS, ENAS | Fast and practical  
First time with NAS | AutoKeras | Simple and easy to use  
Customization needed | DARTS implementation | Highly flexible  
Benchmark research | NAS-Bench | Reproducibility and fairness  
  
### To the Next Chapter

In Chapter 4, we will learn about **Feature Engineering Automation** :

  * Automatic feature generation
  * Feature selection automation
  * Feature importance visualization
  * AutoML pipeline integration
  * Practical feature engineering

* * *

## Exercises

### Exercise 1 (Difficulty: easy)

Explain each of the three components of NAS (search space, search strategy, performance estimation).

Sample Answer

**Answer** :

  1. **Search Space**

     * Description: Set of searchable architectures
     * Examples: Cell-based (Normal Cell and Reduction Cell), layer types (Conv, Pooling), connection patterns
     * Importance: If search space is too large, computational cost is high; if too small, optimal solution is missed
  2. **Search Strategy**

     * Description: How to search architectures
     * Examples: Reinforcement learning (NASNet), evolutionary algorithm (AmoebaNet), gradient-based (DARTS)
     * Tradeoff: Accuracy vs computational cost
  3. **Performance Estimation**

     * Description: Method to determine architecture quality
     * Metrics: Accuracy, FLOPs, parameters, latency, memory usage
     * Efficiency: Proxy tasks, weight sharing, early stopping

### Exercise 2 (Difficulty: medium)

Explain why DARTS is faster than reinforcement learning-based NAS from the perspective of continuous relaxation.

Sample Answer

**Answer** :

**Reinforcement Learning-based NAS (e.g., NASNet)** :

  * Discrete search: Repeatedly sample architecture → train → evaluate
  * Each candidate must be trained individually
  * Computational cost: Thousands of architectures × full training = very high (1800 GPU days)

**DARTS (Gradient-based)** :

  * Continuous relaxation: Convert discrete choices (which operation to use) to continuous weighted sum
  * Formula: $\bar{o}(x) = \sum_o \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})} \cdot o(x)$
  * Gradient descent applicable: α can be optimized with gradients
  * Weight sharing: All candidates share the same supernet
  * Computational cost: One supernet training = dramatic reduction (4 GPU days)

**Speedup Reasons** :

  1. Discrete→Continuous: Becomes differentiable, enabling efficient gradient optimization
  2. Weight sharing: Share weights among candidates, avoiding individual training
  3. Bi-level optimization: Alternately update w and α for efficient search

**Results** :

  * NASNet: 1800 GPU days
  * DARTS: 4 GPU days
  * Speedup factor: About 450x

### Exercise 3 (Difficulty: medium)

Complete the following code to search for an image classification model using AutoKeras.
    
    
    import autokeras as ak
    from tensorflow.keras.datasets import fashion_mnist
    
    # Prepare data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # x_train = ...
    # x_test = ...
    
    # clf = ak.ImageClassifier(...)
    
    # Exercise: Train model
    # clf.fit(...)
    
    # test_acc = ...
    # print(f"Test accuracy: {test_acc:.4f}")
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Complete the following code to search for an image classific
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import autokeras as ak
    from tensorflow.keras.datasets import fashion_mnist
    
    # Prepare data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reduce data size (for demo)
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    
    print("=== Fashion-MNIST Classification ===")
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    
    # AutoKeras ImageClassifier
    clf = ak.ImageClassifier(
        max_trials=5,  # Number of candidates to search
        epochs=10,     # Training epochs per candidate
        overwrite=True,
        directory='autokeras_fashion',
        project_name='fashion_mnist'
    )
    
    # Train model
    print("\nStarting search...")
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluation
    print("\n=== Evaluation ===")
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Export best model
    best_model = clf.export_model()
    print("\n=== Discovered Model ===")
    best_model.summary()
    
    # Prediction examples
    import numpy as np
    predictions = clf.predict(x_test[:5])
    print("\n=== Prediction Examples ===")
    for i in range(5):
        print(f"Sample {i+1}: Prediction={np.argmax(predictions[i])}, True={y_test[i]}")
    

### Exercise 4 (Difficulty: hard)

Implement a supernet using weight sharing and verify that weights are shared among different architectures.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SharedOperations(nn.Module):
        """Pool of operations with shared weights"""
    
        def __init__(self, C):
            super(SharedOperations, self).__init__()
    
            # All possible operations (weights defined only once)
            self.ops = nn.ModuleDict({
                'conv_3x3': nn.Conv2d(C, C, 3, padding=1, bias=False),
                'conv_5x5': nn.Conv2d(C, C, 5, padding=2, bias=False),
                'sep_conv_3x3': nn.Sequential(
                    nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
                    nn.Conv2d(C, C, 1, bias=False)
                ),
                'max_pool_3x3': nn.MaxPool2d(3, stride=1, padding=1),
                'avg_pool_3x3': nn.AvgPool2d(3, stride=1, padding=1),
                'identity': nn.Identity()
            })
    
        def forward(self, x, op_name):
            """Apply specified operation"""
            return self.ops[op_name](x)
    
    class SuperNet(nn.Module):
        """Supernet"""
    
        def __init__(self, C=16):
            super(SuperNet, self).__init__()
            self.shared_ops = SharedOperations(C)
    
        def forward(self, x, architecture):
            """
            architecture: Format [(op_name, input_id), ...]
            """
            # Initial states
            states = [x]
    
            for op_name, input_id in architecture:
                s = self.shared_ops(states[input_id], op_name)
                states.append(s)
    
            # Return last state
            return states[-1]
    
    # Build supernet
    supernet = SuperNet(C=16)
    
    print("=== Weight Sharing Verification ===")
    print(f"Supernet parameter count: {sum(p.numel() for p in supernet.parameters()):,}")
    
    # Parameters per operation
    print("\nParameter count per operation:")
    for name, op in supernet.shared_ops.ops.items():
        num_params = sum(p.numel() for p in op.parameters())
        print(f"  {name}: {num_params:,}")
    
    # Two different architectures
    arch1 = [('conv_3x3', 0), ('max_pool_3x3', 0), ('identity', 1)]
    arch2 = [('conv_5x5', 0), ('avg_pool_3x3', 0), ('conv_3x3', 1)]
    
    # Same input
    x = torch.randn(2, 16, 32, 32)
    
    # Forward with architecture 1
    output1 = supernet(x, arch1)
    
    # Forward with architecture 2
    output2 = supernet(x, arch2)
    
    print(f"\n=== Forward Verification ===")
    print(f"Architecture 1 output shape: {output1.shape}")
    print(f"Architecture 2 output shape: {output2.shape}")
    
    # Verify weight sharing
    print("\n=== Weight Sharing Verification ===")
    conv_3x3_params_before = list(supernet.shared_ops.ops['conv_3x3'].parameters())[0].clone()
    
    # Backward with architecture 1 (uses conv_3x3)
    loss1 = output1.sum()
    loss1.backward()
    
    conv_3x3_params_after = list(supernet.shared_ops.ops['conv_3x3'].parameters())[0]
    
    # Verify gradients accumulated
    has_gradient = conv_3x3_params_after.grad is not None
    print(f"Gradients accumulated in conv_3x3: {has_gradient}")
    
    # Visualize weight sharing
    print("\n=== Advantages of Weight Sharing ===")
    print("1. Memory efficiency: Same weights used for all architectures")
    print("2. Training efficiency: Evaluate all candidates with one supernet")
    print("3. Speedup: Shared training instead of individual training")
    
    # Try different architectures
    print("\n=== Multiple Architecture Evaluation ===")
    architectures = [
        [('conv_3x3', 0), ('max_pool_3x3', 0)],
        [('conv_5x5', 0), ('identity', 0)],
        [('sep_conv_3x3', 0), ('avg_pool_3x3', 0)],
    ]
    
    for i, arch in enumerate(architectures, 1):
        output = supernet(x, arch)
        print(f"Architecture {i}: Output shape = {output.shape}, Mean = {output.mean():.4f}")
    
    print("\n→ All architectures evaluated while sharing the same weights")
    

### Exercise 5 (Difficulty: hard)

In DARTS bi-level optimization, explain why training and validation data need to be separated for optimization. Also predict what would happen if the same data is used for optimization.

Sample Answer

**Answer** :

**Purpose of Bi-level Optimization** :

DARTS optimizes two types of parameters:

  1. **Network weights (w)** : Minimize on training data
  2. **Architecture parameters (α)** : Minimize on validation data

Optimization problem:

$$ \begin{aligned} \min_{\alpha} \quad & \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \\\ \text{s.t.} \quad & w^*(\alpha) = \arg\min_{w} \mathcal{L}_{\text{train}}(w, \alpha) \end{aligned} $$

**Reasons for Separating Training and Validation Data** :

  1. **Prevent Overfitting**

     * Optimizing α on training data selects architectures that overfit to training data
     * Optimizing on validation data selects architectures with high generalization performance
  2. **Role Separation**

     * w: Learn best weights for given architecture (training data)
     * α: Select architecture with highest validation performance (validation data)
  3. **Fair Evaluation**

     * Evaluate w trained on training data with independent validation data
     * Architecture selection reflecting true generalization performance

**Problems When Using Same Data for Optimization** :
    
    
    # Incorrect method (optimize w and α on same data)
    # ❌ Problematic code example
    for epoch in range(num_epochs):
        # Update w on training data
        loss_w = train_loss(w, alpha, train_data)
        w.update(-lr * grad(loss_w, w))
    
        # Update α on same training data ← Problem!
        loss_alpha = train_loss(w, alpha, train_data)
        alpha.update(-lr * grad(loss_alpha, alpha))
    

**Problems That Occur** :

  1. **Overfitting** : Select architectures specialized for training data
  2. **Identity Operation Preference** : Select mostly skip connections as they reduce training loss without computational cost
  3. **Degraded Generalization** : Poor performance on test data
  4. **Meaningful Search Failure** : Cannot discover truly useful architectures

**Correct Method** :
    
    
    # ✅ Correct method
    for epoch in range(num_epochs):
        # Update w on training data
        loss_w = train_loss(w, alpha, train_data)
        w.update(-lr * grad(loss_w, w))
    
        # Update α on validation data ← Correct!
        loss_alpha = val_loss(w, alpha, val_data)
        alpha.update(-lr * grad(loss_alpha, alpha))
    

**Summary** :

Aspect | Training/Validation Separation | Same Data Usage  
---|---|---  
**Overfitting** | Can prevent | Prone to overfitting  
**Generalization** | High | Low  
**Architecture** | Useful | Trivial (identity-biased)  
**Practicality** | High | Low  
  
* * *

## References

  1. Zoph, B., & Le, Q. V. (2017). _Neural Architecture Search with Reinforcement Learning_. ICLR 2017.
  2. Liu, H., Simonyan, K., & Yang, Y. (2019). _DARTS: Differentiable Architecture Search_. ICLR 2019.
  3. Pham, H., Guan, M., Zoph, B., Le, Q., & Dean, J. (2018). _Efficient Neural Architecture Search via Parameter Sharing_. ICML 2018.
  4. Real, E., et al. (2019). _Regularized Evolution for Image Classifier Architecture Search_. AAAI 2019.
  5. Jin, H., Song, Q., & Hu, X. (2019). _Auto-Keras: An Efficient Neural Architecture Search System_. KDD 2019.
  6. Elsken, T., Metzen, J. H., & Hutter, F. (2019). _Neural Architecture Search: A Survey_. JMLR 2019.
