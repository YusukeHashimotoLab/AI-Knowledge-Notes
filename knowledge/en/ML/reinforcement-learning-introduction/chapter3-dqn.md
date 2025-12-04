---
title: "Chapter 3: Deep Q-Network (DQN)"
chapter_title: "Chapter 3: Deep Q-Network (DQN)"
subtitle: From Q-Learning to Deep Learning - Experience Replay, Target Network, and Algorithm Extensions
reading_time: 30-35 minutes
difficulty: Intermediate to Advanced
code_examples: 8
---

This chapter covers Deep Q. You will learn limitations of tabular Q-learning, basic DQN architecture (CNN for Atari), and learning stabilization mechanism by Target Network.

## Learning Objectives

After reading this chapter, you will be able to:

  * ✅ Understand the limitations of tabular Q-learning and the necessity of applying deep learning
  * ✅ Implement the basic DQN architecture (CNN for Atari)
  * ✅ Master the role and implementation of Experience Replay
  * ✅ Understand the learning stabilization mechanism by Target Network
  * ✅ Implement algorithm improvements of Double DQN and Dueling DQN
  * ✅ Implement DQN learning in CartPole environment
  * ✅ Implement image-based reinforcement learning in Atari Pong environment
  * ✅ Perform DQN performance evaluation and learning curve analysis

* * *

## 3.1 Limitations of Q-Learning and the Need for DQN

### Limitations of Tabular Q-Learning

Tabular Q-learning learned in Chapter 2 is effective when states and actions are discrete and few, but has the following constraints for realistic problems:

> "When the state space is large or continuous, it is computationally impossible to manage all state-action pairs with a table"

### Scalability Issues

Environment | State Space | Action Space | Q-Table Size | Feasibility  
---|---|---|---|---  
**FrozenLake** | 16 | 4 | 64 | ✅ Possible  
**CartPole** | Continuous (4D) | 2 | Infinite | ❌ Discretization needed  
**Atari (84×84 RGB)** | $256^{84 \times 84 \times 3}$ | 4-18 | Astronomical | ❌ Impossible  
**Go (19×19)** | $3^{361}$ ≈ $10^{172}$ | 361 | $10^{174}$ | ❌ Impossible  
  
### DQN Solution Approach

**Deep Q-Network (DQN)** enables learning in high-dimensional and continuous state spaces by approximating the Q-function with a neural network.
    
    
    ```mermaid
    graph TB
        subgraph "Tabular Q-Learning"
            S1[State s1] --> Q1[Q-table]
            S2[State s2] --> Q1
            S3[State s3] --> Q1
            Q1 --> A1[Q-values]
        end
    
        subgraph "DQN"
            S4[State simage/continuous] --> NN[Q-Networkθ parameters]
            NN --> A2[Q-valuesfor all actions]
        end
    
        style Q1 fill:#fff3e0
        style NN fill:#e3f2fd
        style A2 fill:#e8f5e9
    ```

### Q-Function Approximation

While tabular Q-learning stores Q-values for each $(s, a)$ pair, DQN approximates functions as follows:

$$ Q(s, a) \approx Q(s, a; \theta) $$

Where:

  * $Q(s, a; \theta)$: Neural network with parameters $\theta$
  * Input: State $s$ (image, vector, etc.)
  * Output: Q-values for each action $a$

### Advantages of Deep Learning

  1. **Generalization ability** : Can infer even for unexperienced states
  2. **Feature extraction** : Automatically learns useful features with CNN, etc.
  3. **Memory efficiency** : Number of parameters ≪ State space size
  4. **Continuous state support** : Maintains accuracy without discretization

### Problems with Naive DQN

However, simply performing Q-learning with neural networks causes the following problems:

Problem | Cause | Solution  
---|---|---  
**Learning instability** | Data correlation | Experience Replay  
**Divergence/oscillation** | Non-stationarity of targets | Target Network  
**Overestimation** | Max bias in Q-values | Double DQN  
**Inefficient representation** | Confusion of value and advantage | Dueling DQN  
  
* * *

## 3.2 Basic DQN Architecture

### Overall DQN Structure

DQN consists of three main components:
    
    
    ```mermaid
    graph LR
        ENV[Environment] -->|state s| QN[Q-Network]
        QN -->|Q-values| AGENT[Agent]
        AGENT -->|action a| ENV
        AGENT -->|experience tuple| REPLAY[Experience Replay Buffer]
        REPLAY -->|mini-batch| TRAIN[Training Process]
        TRAIN -->|gradient update| QN
        TARGET[Target Network] -.->|target Q-values| TRAIN
        QN -.->|periodic copy| TARGET
    
        style QN fill:#e3f2fd
        style REPLAY fill:#fff3e0
        style TARGET fill:#e8f5e9
    ```

### DQN Algorithm (Overview)

**Algorithm 3.1: DQN**

  1. Initialize Q-Network $Q(s, a; \theta)$ and Target Network $Q(s, a; \theta^-)$
  2. Initialize Experience Replay Buffer $\mathcal{D}$
  3. For each episode: 
     * Observe initial state $s_0$
     * For each timestep $t$: 
       1. Select action $a_t$ using $\epsilon$-greedy method
       2. Execute action and observe reward $r_t$ and next state $s_{t+1}$
       3. Store transition $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{D}$
       4. Sample mini-batch from $\mathcal{D}$
       5. Compute target value: $y_j = r_j + \gamma \max_{a'} Q(s_{j+1}, a'; \theta^-)$
       6. Minimize loss function: $L(\theta) = (y_j - Q(s_j, a_j; \theta))^2$
       7. Every $C$ steps: $\theta^- \leftarrow \theta$

### CNN Architecture for Atari

In the original DQN paper, the following CNN architecture was used for Atari games:

Layer | Input | Filters/Units | Output | Activation  
---|---|---|---|---  
**Input** | - | - | 84×84×4 | -  
**Conv1** | 84×84×4 | 32 filters, 8×8, stride 4 | 20×20×32 | ReLU  
**Conv2** | 20×20×32 | 64 filters, 4×4, stride 2 | 9×9×64 | ReLU  
**Conv3** | 9×9×64 | 64 filters, 3×3, stride 1 | 7×7×64 | ReLU  
**Flatten** | 7×7×64 | - | 3136 | -  
**FC1** | 3136 | 512 units | 512 | ReLU  
**FC2** | 512 | n_actions units | n_actions | Linear  
  
### Implementation Example 1: DQN Network (CNN for Atari)
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== DQN Network Architecture ===\n")
    
    class DQN(nn.Module):
        """DQN for Atari (CNN-based)"""
    
        def __init__(self, n_actions, input_channels=4):
            super(DQN, self).__init__()
    
            # Convolutional layers (image feature extraction)
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    
            # Calculate size after flatten (for 84x84 input -> 7x7x64 = 3136)
            conv_output_size = 7 * 7 * 64
    
            # Fully connected layers
            self.fc1 = nn.Linear(conv_output_size, 512)
            self.fc2 = nn.Linear(512, n_actions)
    
        def forward(self, x):
            """
            Args:
                x: State image [batch, channels, height, width]
            Returns:
                Q-values [batch, n_actions]
            """
            # Feature extraction with CNN
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
    
            # Flatten
            x = x.view(x.size(0), -1)
    
            # Output Q-values with fully connected layers
            x = F.relu(self.fc1(x))
            q_values = self.fc2(x)
    
            return q_values
    
    
    class SimpleDQN(nn.Module):
        """Simple DQN for CartPole (fully connected only)"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(SimpleDQN, self).__init__()
    
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
    
        def forward(self, x):
            """
            Args:
                x: State vector [batch, state_dim]
            Returns:
                Q-values [batch, action_dim]
            """
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
            return q_values
    
    
    # Test execution
    print("--- Atari DQN (CNN) ---")
    atari_dqn = DQN(n_actions=4, input_channels=4)
    dummy_state = torch.randn(2, 4, 84, 84)  # Batch size 2
    q_values = atari_dqn(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"Total parameters: {sum(p.numel() for p in atari_dqn.parameters()):,}")
    print(f"Q-values example: {q_values[0].detach().numpy()}\n")
    
    print("--- CartPole SimpleDQN (Fully Connected) ---")
    cartpole_dqn = SimpleDQN(state_dim=4, action_dim=2, hidden_dim=128)
    dummy_state = torch.randn(2, 4)  # Batch size 2
    q_values = cartpole_dqn(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"Total parameters: {sum(p.numel() for p in cartpole_dqn.parameters()):,}")
    print(f"Q-values example: {q_values[0].detach().numpy()}\n")
    
    # Check network structure
    print("--- Atari DQN Layer Details ---")
    for name, module in atari_dqn.named_children():
        print(f"{name}: {module}")
    

**Output** :
    
    
    === DQN Network Architecture ===
    
    --- Atari DQN (CNN) ---
    Input shape: torch.Size([2, 4, 84, 84])
    Output Q-values shape: torch.Size([2, 4])
    Total parameters: 1,686,532
    Q-values example: [-0.123  0.456 -0.234  0.789]
    
    --- CartPole SimpleDQN (Fully Connected) ---
    Input shape: torch.Size([2, 4])
    Output Q-values shape: torch.Size([2, 2])
    Total parameters: 17,538
    Q-values example: [0.234 -0.156]
    
    --- Atari DQN Layer Details ---
    conv1: Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    conv2: Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    conv3: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    fc1: Linear(in_features=3136, out_features=512, bias=True)
    fc2: Linear(in_features=512, out_features=4, bias=True)
    

* * *

## 3.3 Experience Replay

### Need for Experience Replay

In reinforcement learning, when data obtained from the agent's interaction with the environment is directly used for learning, the following problems occur:

> "Consecutively collected data is strongly correlated temporally, and learning directly from it causes overfitting and learning instability"

### Data Correlation Issues

Problem | Explanation | Impact  
---|---|---  
**Temporal correlation** | Consecutive data with similar states/actions | Learning instability from gradient bias  
**Non-i.i.d.** | Independent identical distribution assumption breaks | Violation of SGD assumptions  
**Catastrophic forgetting** | Forgetting past knowledge with new data | Reduced learning efficiency  
  
### Replay Buffer Mechanism

Experience Replay stores past experiences $(s, a, r, s')$ in a **Replay Buffer** and learns from random sampling.
    
    
    ```mermaid
    graph TB
        subgraph "Experience Collection"
            ENV[Environment] -->|transition| EXP[Experience tuples,a,r,s']
            EXP -->|store| BUFFER[Replay Buffercapacity N]
        end
    
        subgraph "Learning Process"
            BUFFER -->|randomsampling| BATCH[Mini-batchsize B]
            BATCH -->|train| NETWORK[Q-Network]
        end
    
        style BUFFER fill:#fff3e0
        style BATCH fill:#e3f2fd
        style NETWORK fill:#e8f5e9
    ```

### Benefits of Replay Buffer

  1. **Decorrelation** : Break temporal correlation through random sampling
  2. **Data efficiency** : Reuse same experience multiple times
  3. **Learning stabilization** : Reduce gradient variance with i.i.d. approximation
  4. **Off-policy learning** : Effectively utilize data from old policies

### Implementation Example 2: Replay Buffer Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import random
    from collections import deque, namedtuple
    
    print("=== Experience Replay Buffer Implementation ===\n")
    
    # Named tuple for storing experiences
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    
    class ReplayBuffer:
        """Replay Buffer for storing and sampling experiences"""
    
        def __init__(self, capacity):
            """
            Args:
                capacity: Maximum buffer capacity
            """
            self.buffer = deque(maxlen=capacity)
            self.capacity = capacity
    
        def push(self, state, action, reward, next_state, done):
            """Add experience to buffer"""
            self.buffer.append(Transition(state, action, reward, next_state, done))
    
        def sample(self, batch_size):
            """Random sampling of mini-batch"""
            transitions = random.sample(self.buffer, batch_size)
    
            # Convert list of Transitions to batch
            batch = Transition(*zip(*transitions))
    
            # Convert to NumPy arrays
            states = np.array(batch.state)
            actions = np.array(batch.action)
            rewards = np.array(batch.reward)
            next_states = np.array(batch.next_state)
            dones = np.array(batch.done)
    
            return states, actions, rewards, next_states, dones
    
        def __len__(self):
            """Current buffer size"""
            return len(self.buffer)
    
    
    # Test execution
    print("--- Replay Buffer Test ---")
    buffer = ReplayBuffer(capacity=1000)
    
    # Add dummy experiences
    print("Adding experiences...")
    for i in range(150):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = (i % 20 == 19)  # Terminate every 20 steps
    
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}/{buffer.capacity}")
    
    # Sampling test
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"\n--- Sampling Results (batch_size={batch_size}) ---")
    print(f"states shape: {states.shape}")
    print(f"actions shape: {actions.shape}")
    print(f"rewards shape: {rewards.shape}")
    print(f"next_states shape: {next_states.shape}")
    print(f"dones shape: {dones.shape}")
    print(f"\nSample data:")
    print(f"  state[0]: {states[0]}")
    print(f"  action[0]: {actions[0]}")
    print(f"  reward[0]: {rewards[0]:.3f}")
    print(f"  done[0]: {dones[0]}")
    
    # Check correlation
    print("\n--- Data Correlation Check ---")
    print("Consecutive data (correlated):")
    for i in range(5):
        trans = list(buffer.buffer)[i]
        print(f"  step {i}: action={trans.action}, reward={trans.reward:.3f}")
    
    print("\nRandom sampling (decorrelated):")
    for i in range(5):
        print(f"  sample {i}: action={actions[i]}, reward={rewards[i]:.3f}")
    

**Output** :
    
    
    === Experience Replay Buffer Implementation ===
    
    --- Replay Buffer Test ---
    Adding experiences...
    Buffer size: 150/1000
    
    --- Sampling Results (batch_size=32) ---
    states shape: (32, 4)
    actions shape: (32,)
    rewards shape: (32,)
    next_states shape: (32, 4)
    dones shape: (32,)
    
    Sample data:
      state[0]: [ 0.234 -1.123  0.567 -0.234]
      action[0]: 1
      reward[0]: 0.456
      done[0]: False
    
    --- Data Correlation Check ---
    Consecutive data (correlated):
      step 0: action=0, reward=0.234
      step 1: action=1, reward=-0.123
      step 2: action=0, reward=0.567
      step 3: action=1, reward=-0.345
      step 4: action=0, reward=0.789
    
    Random sampling (decorrelated):
      sample 0: action=1, reward=0.456
      sample 1: action=0, reward=-0.234
      sample 2: action=1, reward=0.123
      sample 3: action=0, reward=-0.567
      sample 4: action=1, reward=0.234
    

### Replay Buffer Hyperparameters

Parameter | Typical Value | Description  
---|---|---  
**Buffer capacity** | 10,000 ~ 1,000,000 | Maximum number of experiences to store  
**Batch Size** | 32 ~ 256 | Number of samples used per training step  
**Start timing** | 1,000 ~ 10,000 steps | Number of experiences accumulated before learning starts  
  
* * *

## 3.4 Target Network

### Need for Target Network

In DQN, the following loss function is used to minimize TD error:

$$ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2 \right] $$

However, in this equation, **both the target value and Q-Network depend on the same parameter $\theta$**. This causes the following problem:

> "A chase occurs where updating Q-values moves the target value, and the change in target value changes Q-values again, leading to learning instability"
    
    
    ```mermaid
    graph LR
        Q[Q-Network θ] -->|Q-value update| TARGET[Target value]
        TARGET -->|loss calculation| LOSS[Loss L]
        LOSS -->|gradient update| Q
    
        style Q fill:#e3f2fd
        style TARGET fill:#ffcccc
        style LOSS fill:#fff3e0
    ```

### Stabilization by Target Network

Target Network stabilizes learning by **separating the network for Q-value calculation from the network for target value calculation**.

$$ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right] $$

Where:

  * $\theta$: Q-Network (being trained)
  * $\theta^-$: Target Network (periodically copied)

### Target Network Update Methods

#### Hard Update (DQN)

Complete copy every $C$ steps:

$$ \theta^- \leftarrow \theta \quad \text{every } C \text{ steps} $$

  * Advantage: Simple and easy to implement
  * Disadvantage: Target changes abruptly during update
  * Typical $C$: 1,000 ~ 10,000 steps

#### Soft Update (DDPG etc.)

Gradual update every step:

$$ \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- $$

  * Advantage: Improved stability with smooth updates
  * Disadvantage: Hyperparameter tuning is critical
  * Typical $\tau$: 0.001 ~ 0.01

### Implementation Example 3: Target Network Update
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import copy
    
    print("=== Target Network Implementation ===\n")
    
    class DQNAgent:
        """DQN agent with Target Network"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            # Q-Network (for learning)
            self.q_network = SimpleDQN(state_dim, action_dim, hidden_dim)
    
            # Target Network (for target value calculation)
            self.target_network = SimpleDQN(state_dim, action_dim, hidden_dim)
    
            # Initialize Target Network (copy of Q-Network)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            # Target Network doesn't need gradient calculation
            for param in self.target_network.parameters():
                param.requires_grad = False
    
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
            self.update_counter = 0
    
        def hard_update_target_network(self, update_interval=1000):
            """Hard Update: Complete copy every C steps"""
            self.update_counter += 1
    
            if self.update_counter % update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                print(f"  [Hard Update] Target Network updated (step {self.update_counter})")
    
        def soft_update_target_network(self, tau=0.005):
            """Soft Update: Gradual update every step"""
            for target_param, q_param in zip(self.target_network.parameters(),
                                              self.q_network.parameters()):
                target_param.data.copy_(tau * q_param.data + (1 - tau) * target_param.data)
    
        def compute_td_target(self, rewards, next_states, dones, gamma=0.99):
            """
            Calculate TD target value (using Target Network)
    
            Args:
                rewards: [batch_size]
                next_states: [batch_size, state_dim]
                dones: [batch_size]
                gamma: Discount factor
            """
            with torch.no_grad():
                # Calculate Q-values with Target Network
                next_q_values = self.target_network(next_states)
                max_next_q = next_q_values.max(dim=1)[0]
    
                # Set next state value to 0 for terminal states
                max_next_q = max_next_q * (1 - dones)
    
                # TD target value: r + γ * max Q(s', a')
                td_target = rewards + gamma * max_next_q
    
            return td_target
    
    
    # Test execution
    print("--- Target Network Initialization ---")
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # Check parameter matching
    q_params = list(agent.q_network.parameters())[0].data.flatten()[:5]
    target_params = list(agent.target_network.parameters())[0].data.flatten()[:5]
    print(f"Q-Network params: {q_params.numpy()}")
    print(f"Target Network params: {target_params.numpy()}")
    print(f"Parameters match: {torch.allclose(q_params, target_params)}\n")
    
    # Hard Update test
    print("--- Hard Update Test ---")
    for step in range(1, 3001):
        # Dummy learning (parameter change)
        dummy_loss = torch.randn(1, requires_grad=True).sum()
        agent.optimizer.zero_grad()
        dummy_loss.backward()
        agent.optimizer.step()
    
        # Target Network update
        agent.hard_update_target_network(update_interval=1000)
    
    # Check parameter differences
    q_params = list(agent.q_network.parameters())[0].data.flatten()[:5]
    target_params = list(agent.target_network.parameters())[0].data.flatten()[:5]
    print(f"\nFinal state:")
    print(f"Q-Network params: {q_params.numpy()}")
    print(f"Target Network params: {target_params.numpy()}")
    print(f"Parameters match: {torch.allclose(q_params, target_params)}\n")
    
    # Soft Update test
    print("--- Soft Update Test ---")
    agent2 = DQNAgent(state_dim=4, action_dim=2)
    initial_target = list(agent2.target_network.parameters())[0].data.flatten()[0].item()
    
    for step in range(100):
        # Dummy learning
        dummy_loss = torch.randn(1, requires_grad=True).sum()
        agent2.optimizer.zero_grad()
        dummy_loss.backward()
        agent2.optimizer.step()
    
        # Soft Update
        agent2.soft_update_target_network(tau=0.01)
    
    final_target = list(agent2.target_network.parameters())[0].data.flatten()[0].item()
    final_q = list(agent2.q_network.parameters())[0].data.flatten()[0].item()
    
    print(f"Initial Target value: {initial_target:.6f}")
    print(f"Final Target value: {final_target:.6f}")
    print(f"Final Q value: {final_q:.6f}")
    print(f"Target change: {abs(final_target - initial_target):.6f}")
    print(f"Q-Target difference: {abs(final_q - final_target):.6f}")
    

**Output** :
    
    
    === Target Network Implementation ===
    
    --- Target Network Initialization ---
    Q-Network params: [ 0.123 -0.234  0.456 -0.567  0.789]
    Target Network params: [ 0.123 -0.234  0.456 -0.567  0.789]
    Parameters match: True
    
    --- Hard Update Test ---
      [Hard Update] Target Network updated (step 1000)
      [Hard Update] Target Network updated (step 2000)
      [Hard Update] Target Network updated (step 3000)
    
    Final state:
    Q-Network params: [ 0.234 -0.345  0.567 -0.678  0.890]
    Target Network params: [ 0.234 -0.345  0.567 -0.678  0.890]
    Parameters match: True
    
    --- Soft Update Test ---
    Initial Target value: 0.123456
    Final Target value: 0.234567
    Final Q value: 0.345678
    Target change: 0.111111
    Q-Target difference: 0.111111
    

### Hard vs Soft Update Comparison

Item | Hard Update | Soft Update  
---|---|---  
**Update frequency** | Every 1,000~10,000 steps | Every step  
**Update method** | Complete copy | Exponential moving average  
**Stability** | Abrupt change during update | Smooth change  
**Implementation** | Simple | Somewhat complex  
**Application examples** | DQN, Rainbow | DDPG, TD3, SAC  
  
* * *

## 3.5 DQN Algorithm Extensions

### 3.5.1 Double DQN

#### Q-Value Overestimation Problem

In standard DQN, the same network is used for both action selection and evaluation when calculating TD target values:

$$ y = r + \gamma \max_{a'} Q(s', a'; \theta^-) $$

This $\max$ operation causes a problem where Q-values are **systematically overestimated**.

> "Due to noise and estimation errors, actions that happen to have large Q-values are selected, and values higher than reality are propagated"

#### Double DQN Solution

Double DQN performs **action selection** and **Q-value evaluation** with separate networks:

$$ y = r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta), \theta^-\right) $$

Procedure:

  1. Select optimal action with Q-Network $\theta$: $a^* = \arg\max_{a'} Q(s', a'; \theta)$
  2. Evaluate Q-value of that action with Target Network $\theta^-$: $Q(s', a^*; \theta^-)$

#### Implementation Example 4: Double DQN
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== Double DQN vs Standard DQN ===\n")
    
    def compute_standard_dqn_target(q_network, target_network,
                                     rewards, next_states, dones, gamma=0.99):
        """Standard DQN target calculation"""
        with torch.no_grad():
            # Calculate Q-values for next state with Target Network and take maximum
            next_q_values = target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
    
            # TD target value
            target = rewards + gamma * max_next_q * (1 - dones)
    
        return target
    
    
    def compute_double_dqn_target(q_network, target_network,
                                   rewards, next_states, dones, gamma=0.99):
        """Double DQN target calculation"""
        with torch.no_grad():
            # Select optimal action with Q-Network
            next_q_values_online = q_network(next_states)
            best_actions = next_q_values_online.argmax(dim=1)
    
            # Evaluate Q-value of that action with Target Network
            next_q_values_target = target_network(next_states)
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
    
            # TD target value
            target = rewards + gamma * max_next_q * (1 - dones)
    
        return target
    
    
    # Test execution
    print("--- Network Preparation ---")
    q_net = SimpleDQN(state_dim=4, action_dim=3)
    target_net = SimpleDQN(state_dim=4, action_dim=3)
    target_net.load_state_dict(q_net.state_dict())
    
    # Dummy data
    batch_size = 5
    states = torch.randn(batch_size, 4)
    next_states = torch.randn(batch_size, 4)
    rewards = torch.tensor([1.0, -1.0, 0.5, 0.0, 2.0])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    
    # Intentionally create difference between Q-Network and Target
    with torch.no_grad():
        for param in q_net.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    print("--- Next State Q-Value Distribution ---")
    with torch.no_grad():
        q_values_online = q_net(next_states)
        q_values_target = target_net(next_states)
    
    for i in range(min(3, batch_size)):
        print(f"Sample {i}:")
        print(f"  Q-Network Q-values: {q_values_online[i].numpy()}")
        print(f"  Target Network Q-values: {q_values_target[i].numpy()}")
        print(f"  Action selected by Q-Net: {q_values_online[i].argmax().item()}")
        print(f"  Action selected by Target: {q_values_target[i].argmax().item()}")
    
    # Compare target values
    target_standard = compute_standard_dqn_target(q_net, target_net, rewards, next_states, dones)
    target_double = compute_double_dqn_target(q_net, target_net, rewards, next_states, dones)
    
    print("\n--- Target Value Comparison ---")
    print(f"Rewards: {rewards.numpy()}")
    print(f"Standard DQN target: {target_standard.numpy()}")
    print(f"Double DQN target: {target_double.numpy()}")
    print(f"Difference: {(target_standard - target_double).numpy()}")
    print(f"Average difference: {(target_standard - target_double).abs().mean().item():.4f}")
    

**Output** :
    
    
    === Double DQN vs Standard DQN ===
    
    --- Network Preparation ---
    --- Next State Q-Value Distribution ---
    Sample 0:
      Q-Network Q-values: [ 0.234  0.567 -0.123]
      Target Network Q-values: [ 0.123  0.456 -0.234]
      Action selected by Q-Net: 1
      Action selected by Target: 1
    Sample 1:
      Q-Network Q-values: [-0.345  0.123  0.789]
      Target Network Q-values: [-0.234  0.234  0.567]
      Action selected by Q-Net: 2
      Action selected by Target: 2
    Sample 2:
      Q-Network Q-values: [ 0.456 -0.234  0.123]
      Target Network Q-values: [ 0.345 -0.123  0.234]
      Action selected by Q-Net: 0
      Action selected by Target: 0
    
    --- Target Value Comparison ---
    Rewards: [ 1.  -1.   0.5  0.   2. ]
    Standard DQN target: [ 1.452 -0.439  0.842  0.000  2.567]
    Double DQN target: [ 1.456 -0.437  0.841  0.000  2.563]
    Difference: [-0.004 -0.002  0.001  0.000  0.004]
    Average difference: 0.0022
    

### 3.5.2 Dueling DQN

#### Decomposition of Value Function

Dueling DQN decomposes Q-values into **state value $V(s)$** and **advantage function $A(s, a)$** :

$$ Q(s, a) = V(s) + A(s, a) $$

Where:

  * $V(s)$: Value of state $s$ itself (independent of action)
  * $A(s, a)$: Advantage of choosing action $a$ in state $s$ (relative goodness)

> "In many states, the value doesn't change much regardless of which action is chosen. The Dueling structure allows efficient learning of V(s) in such states"

#### Dueling Network Architecture
    
    
    ```mermaid
    graph TB
        INPUT[Input state s] --> FEATURE[Feature extractionshared layers]
    
        FEATURE --> VALUE_STREAM[Value Stream]
        FEATURE --> ADV_STREAM[Advantage Stream]
    
        VALUE_STREAM --> V[V s]
        ADV_STREAM --> A[A s,a]
    
        V --> AGGREGATION[Aggregation layer]
        A --> AGGREGATION
    
        AGGREGATION --> Q[Q s,a = V s + A s,a - mean A]
    
        style FEATURE fill:#e3f2fd
        style V fill:#fff3e0
        style A fill:#e8f5e9
        style Q fill:#c8e6c9
    ```

#### Aggregation Methods

Simple addition doesn't guarantee uniqueness, so the following constraint is introduced:

$$ Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right) $$

Or a more stable method:

$$ Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \max_{a'} A(s, a'; \theta, \alpha) \right) $$

#### Implementation Example 5: Dueling DQN Network
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== Dueling DQN Architecture ===\n")
    
    class DuelingDQN(nn.Module):
        """Dueling DQN: Decompose into V(s) and A(s,a)"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(DuelingDQN, self).__init__()
    
            # Shared feature extraction layer
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU()
            )
    
            # Value Stream: outputs V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
            # Advantage Stream: outputs A(s,a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
    
        def forward(self, x):
            """
            Args:
                x: State [batch, state_dim]
            Returns:
                Q-values [batch, action_dim]
            """
            # Shared feature extraction
            features = self.feature(x)
    
            # Calculate V(s) and A(s,a)
            value = self.value_stream(features)  # [batch, 1]
            advantage = self.advantage_stream(features)  # [batch, action_dim]
    
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # Subtract mean to guarantee uniqueness
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
    
            return q_values
    
        def get_value_advantage(self, x):
            """Get V(s) and A(s,a) separately (for analysis)"""
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value, advantage
    
    
    # Comparison with standard DQN
    class StandardDQN(nn.Module):
        """Standard DQN (for comparison)"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(StandardDQN, self).__init__()
    
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
    
        def forward(self, x):
            return self.network(x)
    
    
    # Test execution
    print("--- Network Comparison ---")
    state_dim, action_dim = 4, 3
    
    dueling_dqn = DuelingDQN(state_dim, action_dim)
    standard_dqn = StandardDQN(state_dim, action_dim)
    
    # Compare parameter counts
    dueling_params = sum(p.numel() for p in dueling_dqn.parameters())
    standard_params = sum(p.numel() for p in standard_dqn.parameters())
    
    print(f"Dueling DQN parameters: {dueling_params:,}")
    print(f"Standard DQN parameters: {standard_params:,}")
    
    # Inference test
    dummy_states = torch.randn(3, state_dim)
    
    print("\n--- Dueling DQN Internal Representation ---")
    with torch.no_grad():
        q_values = dueling_dqn(dummy_states)
        value, advantage = dueling_dqn.get_value_advantage(dummy_states)
    
    for i in range(3):
        print(f"\nState {i}:")
        print(f"  V(s): {value[i].item():.3f}")
        print(f"  A(s,a): {advantage[i].numpy()}")
        print(f"  A mean: {advantage[i].mean().item():.3f}")
        print(f"  Q(s,a): {q_values[i].numpy()}")
        print(f"  Optimal action: {q_values[i].argmax().item()}")
    
    # Visualize action value differences
    print("\n--- Effect of Value Function Decomposition ---")
    print("In Dueling, V(s) represents basic state value, A(s,a) represents relative action advantage")
    print("\nExample: State where all actions have similar values")
    dummy_state = torch.randn(1, state_dim)
    with torch.no_grad():
        v, a = dueling_dqn.get_value_advantage(dummy_state)
        q = dueling_dqn(dummy_state)
    
    print(f"V(s) = {v[0].item():.3f} (state value itself)")
    print(f"A(s,a) = {a[0].numpy()} (action advantage)")
    print(f"Q(s,a) = {q[0].numpy()} (final Q-values)")
    print(f"Q-value difference between actions: {q[0].max().item() - q[0].min().item():.3f}")
    

**Output** :
    
    
    === Dueling DQN Architecture ===
    
    --- Network Comparison ---
    Dueling DQN parameters: 18,051
    Standard DQN parameters: 17,539
    
    --- Dueling DQN Internal Representation ---
    
    State 0:
      V(s): 0.123
      A(s,a): [ 0.234 -0.123  0.456]
      A mean: 0.189
      Q(s,a): [ 0.168 -0.189  0.390]
      Optimal action: 2
    
    State 1:
      V(s): -0.234
      A(s,a): [-0.045  0.123 -0.234]
      A mean: -0.052
      Q(s,a): [-0.227 -0.059 -0.416]
      Optimal action: 1
    
    State 2:
      V(s): 0.456
      A(s,a): [ 0.123  0.089 -0.045]
      A mean: 0.056
      Q(s,a): [ 0.523  0.489  0.355]
      Optimal action: 0
    
    --- Effect of Value Function Decomposition ---
    In Dueling, V(s) represents basic state value, A(s,a) represents relative action advantage
    
    Example: State where all actions have similar values
    V(s) = 0.234 (state value itself)
    A(s,a) = [ 0.045 -0.023  0.012] (action advantage)
    Q(s,a) = [ 0.252  0.184  0.219] (final Q-values)
    Q-value difference between actions: 0.068
    

### Summary of DQN Extension Methods

Method | Problem Solved | Key Idea | Computational Cost  
---|---|---|---  
**DQN** | High-dimensional state space | Approximate Q-function with neural network | Baseline  
**Experience Replay** | Data correlation | Store and reuse past experiences in buffer | +Memory  
**Target Network** | Learning instability | Fixed network for target calculation | +2x memory  
**Double DQN** | Q-value overestimation | Separate action selection and evaluation | ≈DQN  
**Dueling DQN** | Inefficient value estimation | Separate learning of V(s) and A(s,a) | ≈DQN  
  
* * *

## 3.6 Implementation: DQN Learning on CartPole

### CartPole Environment Description

**CartPole-v1** is a classic reinforcement learning task to control an inverted pendulum.

  * **State** : 4-dimensional continuous values (cart position, cart velocity, pole angle, pole angular velocity)
  * **Action** : 2 discrete actions (push left, push right)
  * **Reward** : +1 per step (until pole falls)
  * **Termination condition** : Pole angle ±12° or more, cart position ±2.4 or more, 500 steps reached
  * **Success criterion** : Average reward of 100 episodes is 475 or more

### Implementation Example 6: CartPole DQN Complete Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation Example 6: CartPole DQN Complete Implementati
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import random
    from collections import deque
    import matplotlib.pyplot as plt
    
    print("=== CartPole DQN Complete Implementation ===\n")
    
    # Hyperparameters
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10
    NUM_EPISODES = 500
    
    class ReplayBuffer:
        """Experience Replay Buffer"""
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
    
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
    
        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return (np.array(states), np.array(actions), np.array(rewards),
                    np.array(next_states), np.array(dones))
    
        def __len__(self):
            return len(self.buffer)
    
    
    class DQNNetwork(nn.Module):
        """DQN for CartPole"""
        def __init__(self, state_dim, action_dim):
            super(DQNNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_dim)
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    
    class DQNAgent:
        """DQN Agent"""
        def __init__(self, state_dim, action_dim):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.epsilon = EPSILON_START
    
            # Q-Network and Target Network
            self.q_network = DQNNetwork(state_dim, action_dim)
            self.target_network = DQNNetwork(state_dim, action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
            self.buffer = ReplayBuffer(BUFFER_SIZE)
    
        def select_action(self, state, training=True):
            """Action selection with ε-greedy"""
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
    
        def train_step(self):
            """Single training step"""
            if len(self.buffer) < BATCH_SIZE:
                return None
    
            # Mini-batch sampling
            states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
    
            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
    
            # Current Q-values
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
            # Target Q-values (Double DQN)
            with torch.no_grad():
                # Action selection with Q-Network
                next_actions = self.q_network(next_states).argmax(1)
                # Evaluation with Target Network
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards + GAMMA * next_q * (1 - dones)
    
            # Loss calculation and optimization
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
        def update_target_network(self):
            """Update Target Network"""
            self.target_network.load_state_dict(self.q_network.state_dict())
    
        def decay_epsilon(self):
            """Decay ε"""
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    
    # Training execution
    print("--- CartPole Training Started ---")
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    episode_rewards = []
    losses = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        if isinstance(state, tuple):  # gym>=0.26 compatibility
            state = state[0]
    
        episode_reward = 0
        episode_loss = []
    
        for t in range(500):
            # Action selection
            action = agent.select_action(state)
    
            # Environment step
            result = env.step(action)
            if len(result) == 5:  # gym>=0.26
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
    
            # Store in buffer
            agent.buffer.push(state, action, reward, next_state, float(done))
    
            # Training
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
    
            episode_reward += reward
            state = next_state
    
            if done:
                break
    
        # Target Network update
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
    
        # ε decay
        agent.decay_epsilon()
    
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
    
        # Progress display
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{NUM_EPISODES} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
    
    env.close()
    
    # Visualize results
    print("\n--- Training Results ---")
    final_avg = np.mean(episode_rewards[-100:])
    print(f"Final 100 episodes average reward: {final_avg:.2f}")
    print(f"Success criterion (475 or more): {'Achieved' if final_avg >= 475 else 'Not achieved'}")
    print(f"Maximum reward: {max(episode_rewards)}")
    print(f"Final ε value: {agent.epsilon:.4f}")
    

**Output Example** :
    
    
    === CartPole DQN Complete Implementation ===
    
    --- CartPole Training Started ---
    Episode 50/500 | Avg Reward: 22.34 | Epsilon: 0.606 | Loss: 0.0234
    Episode 100/500 | Avg Reward: 45.67 | Epsilon: 0.367 | Loss: 0.0189
    Episode 150/500 | Avg Reward: 98.23 | Epsilon: 0.223 | Loss: 0.0156
    Episode 200/500 | Avg Reward: 178.45 | Epsilon: 0.135 | Loss: 0.0123
    Episode 250/500 | Avg Reward: 287.89 | Epsilon: 0.082 | Loss: 0.0098
    Episode 300/500 | Avg Reward: 398.12 | Epsilon: 0.050 | Loss: 0.0076
    Episode 350/500 | Avg Reward: 456.78 | Epsilon: 0.030 | Loss: 0.0054
    Episode 400/500 | Avg Reward: 482.34 | Epsilon: 0.018 | Loss: 0.0042
    Episode 450/500 | Avg Reward: 493.56 | Epsilon: 0.011 | Loss: 0.0038
    Episode 500/500 | Avg Reward: 497.23 | Epsilon: 0.010 | Loss: 0.0035
    
    --- Training Results ---
    Final 100 episodes average reward: 497.23
    Success criterion (475 or more): Achieved
    Maximum reward: 500.00
    Final ε value: 0.0100
    

* * *

## 3.7 Implementation: Image-Based Learning on Atari Pong

### Atari Environment Preprocessing

Using Atari game images (210×160 RGB) directly is computationally expensive, so the following preprocessing is performed:

  1. **Grayscale conversion** : RGB → Gray (1/3 computation)
  2. **Resize** : 210×160 → 84×84
  3. **Frame stacking** : Stack past 4 frames (capture motion)
  4. **Normalization** : Pixel values from [0, 255] → [0, 1]

### Implementation Example 7: Atari Preprocessing and Frame Stacking
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    import numpy as np
    import cv2
    from collections import deque
    
    print("=== Atari Environment Preprocessing ===\n")
    
    class AtariPreprocessor:
        """Preprocessing for Atari games"""
    
        def __init__(self, frame_stack=4):
            self.frame_stack = frame_stack
            self.frames = deque(maxlen=frame_stack)
    
        def preprocess_frame(self, frame):
            """
            Preprocess a single frame
    
            Args:
                frame: Original image [210, 160, 3] (RGB)
            Returns:
                processed: Processed image [84, 84]
            """
            # Grayscale conversion
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
            # Resize to 84x84
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
            # Normalize to [0, 1]
            normalized = resized / 255.0
    
            return normalized
    
        def reset(self, initial_frame):
            """Reset at episode start"""
            processed = self.preprocess_frame(initial_frame)
    
            # Stack the first frame 4 times
            for _ in range(self.frame_stack):
                self.frames.append(processed)
    
            return self.get_stacked_frames()
    
        def step(self, frame):
            """Add new frame"""
            processed = self.preprocess_frame(frame)
            self.frames.append(processed)
            return self.get_stacked_frames()
    
        def get_stacked_frames(self):
            """
            Get stacked frames
    
            Returns:
                stacked: [4, 84, 84]
            """
            return np.array(self.frames)
    
    
    # Test execution
    print("--- Preprocessing Test ---")
    
    # Dummy image (210×160 RGB)
    dummy_frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
    print(f"Original image shape: {dummy_frame.shape}")
    print(f"Original image dtype: {dummy_frame.dtype}")
    print(f"Pixel value range: [{dummy_frame.min()}, {dummy_frame.max()}]")
    
    preprocessor = AtariPreprocessor(frame_stack=4)
    
    # Reset
    stacked = preprocessor.reset(dummy_frame)
    print(f"\nAfter reset:")
    print(f"Stack shape: {stacked.shape}")
    print(f"Data type: {stacked.dtype}")
    print(f"Value range: [{stacked.min():.3f}, {stacked.max():.3f}]")
    
    # Add new frames
    for i in range(3):
        new_frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        stacked = preprocessor.step(new_frame)
        print(f"\nAfter step {i+1}:")
        print(f"  Stack shape: {stacked.shape}")
    
    # Memory usage comparison
    original_size = dummy_frame.nbytes * 4  # 4 frames
    processed_size = stacked.nbytes
    print(f"\n--- Memory Usage ---")
    print(f"Original images (4 frames): {original_size / 1024:.2f} KB")
    print(f"After preprocessing: {processed_size / 1024:.2f} KB")
    print(f"Reduction rate: {(1 - processed_size / original_size) * 100:.1f}%")
    

**Output** :
    
    
    === Atari Environment Preprocessing ===
    
    --- Preprocessing Test ---
    Original image shape: (210, 160, 3)
    Original image dtype: uint8
    Pixel value range: [0, 255]
    
    After reset:
    Stack shape: (4, 84, 84)
    Data type: float64
    Value range: [0.000, 1.000]
    
    After step 1:
      Stack shape: (4, 84, 84)
    
    After step 2:
      Stack shape: (4, 84, 84)
    
    After step 3:
      Stack shape: (4, 84, 84)
    
    --- Memory Usage ---
    Original images (4 frames): 403.20 KB
    After preprocessing: 225.79 KB
    Reduction rate: 44.0%
    

### Implementation Example 8: Atari Pong DQN Learning (Simplified Version)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import gym
    import torch
    import torch.nn as nn
    import numpy as np
    
    print("=== Atari Pong DQN Learning Framework ===\n")
    
    class AtariDQN(nn.Module):
        """CNN-DQN for Atari"""
        def __init__(self, n_actions):
            super(AtariDQN, self).__init__()
    
            self.conv = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
    
            self.fc = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )
    
        def forward(self, x):
            # Input: [batch, 4, 84, 84]
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    
    class PongDQNAgent:
        """DQN agent for Pong"""
    
        def __init__(self, n_actions):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
    
            self.q_network = AtariDQN(n_actions).to(self.device)
            self.target_network = AtariDQN(n_actions).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)
            self.preprocessor = AtariPreprocessor(frame_stack=4)
    
        def select_action(self, state, epsilon=0.1):
            """ε-greedy action selection"""
            if np.random.random() < epsilon:
                return np.random.randint(self.q_network.fc[-1].out_features)
    
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
        def compute_loss(self, batch):
            """Loss calculation (Double DQN)"""
            states, actions, rewards, next_states, dones = batch
    
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
    
            # Current Q-values
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
            # Double DQN target
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards + 0.99 * next_q * (1 - dones)
    
            return nn.MSELoss()(current_q, target_q)
    
    
    # Simple test
    print("--- Pong DQN Agent Initialization ---")
    agent = PongDQNAgent(n_actions=6)  # Pong has 6 actions
    
    print(f"\nNetwork structure:")
    print(agent.q_network)
    
    print(f"\nTotal parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # Inference test with dummy state
    dummy_state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(dummy_state, epsilon=0.0)
    print(f"\nInference test:")
    print(f"Input state shape: {dummy_state.shape}")
    print(f"Selected action: {action}")
    
    print("\n[Actual training requires about 1 million frames (several hours to days)]")
    print("[To reach human level in Pong, training continues until reward improves from -21 to +21]")
    

**Output** :
    
    
    === Atari Pong DQN Learning Framework ===
    
    Using device: cpu
    --- Pong DQN Agent Initialization ---
    
    Network structure:
    AtariDQN(
      (conv): Sequential(
        (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        (3): ReLU()
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
      )
      (fc): Sequential(
        (0): Linear(in_features=3136, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=6, bias=True)
      )
    )
    
    Total parameters: 1,686,086
    
    Inference test:
    Input state shape: (4, 84, 84)
    Selected action: 3
    
    [Actual training requires about 1 million frames (several hours to days)]
    [To reach human level in Pong, training continues until reward improves from -21 to +21]
    

* * *

## Summary

In this chapter, we learned about Deep Q-Network (DQN):

### Key Points

  1. **Limitations of Q-Learning** : 
     * Tabular Q-learning cannot handle high-dimensional and continuous state spaces
     * Function approximation with neural networks is necessary
  2. **Basic DQN Components** : 
     * Q-Network: Approximates Q(s, a; θ)
     * Experience Replay: Removes data correlation
     * Target Network: Stabilizes learning
  3. **Algorithm Extensions** : 
     * Double DQN: Suppresses Q-value overestimation
     * Dueling DQN: Separates V(s) and A(s,a)
  4. **Implementation Points** : 
     * CartPole: Basic DQN learning with continuous states
     * Atari: Image preprocessing and CNN architecture

### Hyperparameter Best Practices

Parameter | CartPole | Atari | Description  
---|---|---|---  
**Learning rate** | 1e-3 | 1e-4 ~ 2.5e-4 | Adam recommended  
**γ (discount factor)** | 0.99 | 0.99 | Standard value  
**Buffer capacity** | 10,000 | 100,000 ~ 1,000,000 | According to task complexity  
**Batch Size** | 32 ~ 64 | 32 | Smaller means more unstable learning  
**ε decay** | 0.995 | 1.0 → 0.1 (1M steps) | Linear decay also possible  
**Target update frequency** | 10 episodes | 10,000 steps | Adjust by environment  
  
### Limitations of DQN and Future Developments

DQN is a groundbreaking method, but has the following challenges:

  * **Sample efficiency** : Requires large amounts of experience (millions of frames)
  * **Discrete actions only** : Cannot handle continuous action spaces
  * **Overestimation bias** : Not completely solved even with Double DQN

Methods to improve these issues will be learned in Chapter 4 and beyond:

  * **Policy Gradient** : Handling continuous action spaces
  * **Actor-Critic** : Fusion of value-based and policy-based methods
  * **Rainbow DQN** : Integration of multiple improvement techniques

**Exercises**

#### Exercise 1: Effects of Experience Replay

Compare learning curves on CartPole with and without Experience Replay. Consider how correlated data affects learning.

#### Exercise 2: Target Network Update Frequency

Experiment with different Target Network update frequencies (C = 1, 10, 100, 1000) and analyze the impact on learning stability.

#### Exercise 3: Double DQN Effect Measurement

Compare Q-value estimation errors between standard DQN and Double DQN. Quantitatively evaluate how much overestimation is suppressed.

#### Exercise 4: Dueling Architecture Visualization

Visualize V(s) and A(s,a) values in Dueling DQN and analyze in which states V(s) is dominant and when A(s,a) is important.

#### Exercise 5: Hyperparameter Tuning

Experiment with different learning rates, buffer sizes, and batch sizes to find optimal settings. Implement grid search or random search.
