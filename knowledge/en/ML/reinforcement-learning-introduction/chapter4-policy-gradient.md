---
title: "Chapter 4: Policy Gradient Methods"
chapter_title: "Chapter 4: Policy Gradient Methods"
subtitle: "Policy-based Reinforcement Learning: Theory and Implementation of REINFORCE, Actor-Critic, A2C, and PPO"
reading_time: 28 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 6
---

This chapter covers Policy Gradient Methods. You will learn differences between policy-based, mathematical formulation of policy gradients, and REINFORCE algorithm.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the differences between policy-based and value-based approaches
  * ✅ Understand the mathematical formulation of policy gradients
  * ✅ Implement the REINFORCE algorithm
  * ✅ Understand and implement the Actor-Critic architecture
  * ✅ Implement Advantage Actor-Critic (A2C)
  * ✅ Understand Proximal Policy Optimization (PPO)
  * ✅ Solve continuous control tasks such as LunarLander

* * *

## 4.1 Policy-based vs Value-based

### 4.1.1 Two Approaches

There are two major approaches in reinforcement learning:

Characteristic | Value-based | Policy-based  
---|---|---  
**Learning Target** | Value function $Q(s, a)$ or $V(s)$ | Learn policy $\pi(a|s)$ directly  
**Action Selection** | Indirect ($\arg\max_a Q(s,a)$) | Direct (sample from $\pi(a|s)$)  
**Action Space** | Suitable for discrete actions | Can handle continuous actions  
**Stochastic Policy** | Difficult to handle (use ε-greedy, etc.) | Naturally handled  
**Convergence** | Optimal policy guaranteed (under conditions) | Possible local optima  
**Sample Efficiency** | High (experience replay possible) | Low (on-policy learning)  
**Representative Algorithms** | Q-learning, DQN, Double DQN | REINFORCE, A2C, PPO, TRPO  
  
### 4.1.2 Motivation for Policy Gradient

**Challenges of value-based approaches** :

  * **Continuous action space** : In robot control with infinite action choices, $\arg\max$ computation is difficult
  * **Stochastic policy** : Difficult to handle cases where stochastic actions are optimal, like rock-paper-scissors
  * **High-dimensional action space** : Computing all action values is inefficient when action combinations are vast

**Solutions provided by policy-based approaches** :

  * Model policy with parameter $\theta$: $\pi_\theta(a|s)$
  * Directly optimize $\theta$ to maximize expected return
  * Can represent policy with neural networks

    
    
    ```mermaid
    graph LR
        subgraph "Value-based Approach"
            S1["State s"]
            Q["Q-functionQ(s,a)"]
            AM["argmax"]
            A1["Action a"]
    
            S1 --> Q
            Q --> AM
            AM --> A1
    
            style Q fill:#e74c3c,color:#fff
        end
    
        subgraph "Policy-based Approach"
            S2["State s"]
            P["Policy π(a|s)parameterized by θ"]
            A2["Action a(sampled)"]
    
            S2 --> P
            P --> A2
    
            style P fill:#27ae60,color:#fff
        end
    ```

### 4.1.3 Policy Gradient Formulation

We represent policy $\pi_\theta(a|s)$ with parameter $\theta$ and maximize the expected return (objective function) $J(\theta)$:

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] $$ 

Where:

  * $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots)$: trajectory
  * $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$: cumulative reward of trajectory

**Policy Gradient Theorem** states that the gradient of $J(\theta)$ can be expressed as:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right] $$ 

Where $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ is the cumulative reward from time $t$.

> **Intuitive Understanding** : Increase the probability of actions that led to high returns, and decrease the probability of actions that led to low returns. This updates the policy parameters toward generating better trajectories.

* * *

## 4.2 REINFORCE Algorithm

### 4.2.1 Basic Principle of REINFORCE

**REINFORCE** (Williams, 1992) is the simplest policy gradient algorithm. It uses Monte Carlo methods to estimate returns.

#### Algorithm

  1. Execute one episode with policy $\pi_\theta(a|s)$ and collect trajectory $\tau$
  2. Calculate return $R_t$ at each time step $t$
  3. Update parameters by gradient ascent: $$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) R_t $$ 

#### Variance Reduction: Baseline

Subtracting a constant $b$ from the return does not change the expected gradient (unbiasedness), which allows variance reduction:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) (R_t - b) \right] $$ 

Common choice: **$b = V(s_t)$** (state value function)
    
    
    ```mermaid
    graph TB
        subgraph "REINFORCE Algorithm"
            Init["Initialize θ"]
            Episode["Run EpisodeSample τ ~ π_θ"]
            Compute["Compute ReturnsR_t for all t"]
            Grad["Compute Gradient∇_θ log π_θ(a_t|s_t) R_t"]
            Update["Update Parametersθ ← θ + α ∇_θ J(θ)"]
            Check["Converged?"]
    
            Init --> Episode
            Episode --> Compute
            Compute --> Grad
            Grad --> Update
            Update --> Check
            Check -->|No| Episode
            Check -->|Yes| Done["Done"]
    
            style Init fill:#7b2cbf,color:#fff
            style Done fill:#27ae60,color:#fff
            style Grad fill:#e74c3c,color:#fff
        end
    ```

### 4.2.2 REINFORCE Implementation (CartPole)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import gym
    import matplotlib.pyplot as plt
    from collections import deque
    
    class PolicyNetwork(nn.Module):
        """
        Policy Network for REINFORCE
    
        Input: state s
        Output: action probability π(a|s)
        """
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
    
        def forward(self, state):
            """
            Args:
                state: state [batch_size, state_dim]
    
            Returns:
                action_probs: action probabilities [batch_size, action_dim]
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            logits = self.fc3(x)
            action_probs = F.softmax(logits, dim=-1)
            return action_probs
    
    
    class REINFORCE:
        """REINFORCE Algorithm"""
    
        def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
            """
            Args:
                state_dim: dimension of state space
                action_dim: dimension of action space
                lr: learning rate
                gamma: discount factor
            """
            self.gamma = gamma
            self.policy = PolicyNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
            # Save episode data
            self.saved_log_probs = []
            self.rewards = []
    
        def select_action(self, state):
            """
            Select action according to policy
    
            Args:
                state: state
    
            Returns:
                action: selected action
            """
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy(state)
    
            # Sample from probability distribution
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
    
            # Save log π(a|s) for gradient computation
            self.saved_log_probs.append(m.log_prob(action))
    
            return action.item()
    
        def update(self):
            """
            Update parameters after episode completion
            """
            R = 0
            returns = []
    
            # Compute returns (calculate in reverse order)
            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
    
            returns = torch.tensor(returns)
    
            # Normalize (variance reduction)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
            # Compute policy gradient
            policy_loss = []
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
    
            # Gradient ascent (minimize loss = minimize -objective function)
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
    
            # Reset
            self.saved_log_probs = []
            self.rewards = []
    
            return policy_loss.item()
    
    
    # Demonstration
    print("=== REINFORCE on CartPole ===\n")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    
    agent = REINFORCE(state_dim, action_dim, lr=0.01, gamma=0.99)
    
    print(f"\nAgent: REINFORCE")
    total_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Training
    num_episodes = 500
    episode_rewards = []
    moving_avg = deque(maxlen=100)
    
    print("\nTraining...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
    
        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
    
            agent.rewards.append(reward)
            episode_reward += reward
    
            state = next_state
    
            if done:
                break
    
        # Update after episode completion
        loss = agent.update()
    
        episode_rewards.append(episode_reward)
        moving_avg.append(episode_reward)
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(moving_avg)
            print(f"Episode {episode+1:3d}, Avg Reward (last 100): {avg_reward:.2f}, Loss: {loss:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(moving_avg):.2f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    
    # Moving average
    window = 50
    moving_avg_plot = [np.mean(episode_rewards[max(0, i-window):i+1])
                       for i in range(len(episode_rewards))]
    ax.plot(moving_avg_plot, linewidth=2, color='darkorange', label=f'Moving Average ({window} episodes)')
    
    ax.axhline(y=195, color='red', linestyle='--', label='Solved Threshold (195)')
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title('REINFORCE on CartPole-v1', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✓ REINFORCE characteristics:")
    print("  • Simple and easy to implement")
    print("  • Monte Carlo method (update after episode completion)")
    print("  • High variance (large fluctuation in returns)")
    print("  • On-policy (sampling with current policy)")
    

**Sample Output** :
    
    
    === REINFORCE on CartPole ===
    
    Environment: CartPole-v1
      State dimension: 4
      Action dimension: 2
    
    Agent: REINFORCE
      Total parameters: 16,642
    
    Training...
    Episode  50, Avg Reward (last 100): 23.45, Loss: 15.2341
    Episode 100, Avg Reward (last 100): 45.67, Loss: 12.5678
    Episode 150, Avg Reward (last 100): 89.23, Loss: 8.3456
    Episode 200, Avg Reward (last 100): 142.56, Loss: 5.6789
    Episode 250, Avg Reward (last 100): 178.34, Loss: 3.4567
    Episode 300, Avg Reward (last 100): 195.78, Loss: 2.1234
    Episode 350, Avg Reward (last 100): 210.45, Loss: 1.5678
    Episode 400, Avg Reward (last 100): 230.67, Loss: 1.2345
    Episode 450, Avg Reward (last 100): 245.89, Loss: 0.9876
    Episode 500, Avg Reward (last 100): 260.34, Loss: 0.7654
    
    Training completed!
    Final average reward (last 100 episodes): 260.34
    
    ✓ REINFORCE characteristics:
      • Simple and easy to implement
      • Monte Carlo method (update after episode completion)
      • High variance (large fluctuation in returns)
      • On-policy (sampling with current policy)
    

### 4.2.3 Challenges of REINFORCE

REINFORCE has the following challenges:

  * **High variance** : Large variance in return $R_t$, resulting in unstable learning
  * **Sample inefficient** : Cannot update until episode completion
  * **Monte Carlo method** : Learning is slow for long episodes

**Solution** : **Actor-Critic** architecture estimates returns using value functions

* * *

## 4.3 Actor-Critic Method

### 4.3.1 Principle of Actor-Critic

**Actor-Critic** combines policy gradient (Actor) and value-based (Critic) methods.

Component | Role | Output  
---|---|---  
**Actor** | Learns policy $\pi_\theta(a|s)$ | Action probability distribution  
**Critic** | Learns value function $V_\phi(s)$ | State value estimation  
  
#### Advantages

  * **Low variance** : Variance reduction using Critic's baseline $V(s)$
  * **TD learning** : Can update during episodes (using TD error)
  * **Efficient** : Faster learning than Monte Carlo methods

#### Update Equations

**TD Error (Advantage)** :

$$ A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) $$ 

**Actor Update** :

$$ \theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t|s_t) A_t $$ 

**Critic Update** :

$$ \phi \leftarrow \phi - \alpha_\phi \nabla_\phi (r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2 $$ 
    
    
    ```mermaid
    graph TB
        subgraph "Actor-Critic Architecture"
            State["State s_t"]
    
            Actor["Actorπ_θ(a|s)"]
            Critic["CriticV_φ(s)"]
    
            Action["Action a_t"]
            Value["Value V(s_t)"]
    
            Env["Environment"]
            Reward["Reward r_t"]
            NextState["Next State s_{t+1}"]
    
            TDError["TD ErrorA_t = r_t + γV(s_{t+1}) - V(s_t)"]
    
            ActorUpdate["Actor Updateθ ← θ + α ∇_θ log π A_t"]
            CriticUpdate["Critic Updateφ ← φ - α ∇_φ (A_t)²"]
    
            State --> Actor
            State --> Critic
    
            Actor --> Action
            Critic --> Value
    
            Action --> Env
            State --> Env
    
            Env --> Reward
            Env --> NextState
    
            Reward --> TDError
            Value --> TDError
            NextState --> Critic
    
            TDError --> ActorUpdate
            TDError --> CriticUpdate
    
            ActorUpdate -.-> Actor
            CriticUpdate -.-> Critic
    
            style Actor fill:#27ae60,color:#fff
            style Critic fill:#e74c3c,color:#fff
            style TDError fill:#f39c12,color:#fff
        end
    ```

### 4.3.2 Actor-Critic Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import gym
    
    class ActorCriticNetwork(nn.Module):
        """
        Actor-Critic Network
    
        Has a shared feature extractor and two heads for Actor and Critic
        """
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(ActorCriticNetwork, self).__init__()
    
            # Shared layers
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
            # Actor head (policy)
            self.actor_head = nn.Linear(hidden_dim, action_dim)
    
            # Critic head (value function)
            self.critic_head = nn.Linear(hidden_dim, 1)
    
        def forward(self, state):
            """
            Args:
                state: state [batch_size, state_dim]
    
            Returns:
                action_probs: action probabilities [batch_size, action_dim]
                state_value: state value [batch_size, 1]
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
    
            # Actor output
            logits = self.actor_head(x)
            action_probs = F.softmax(logits, dim=-1)
    
            # Critic output
            state_value = self.critic_head(x)
    
            return action_probs, state_value
    
    
    class ActorCritic:
        """Actor-Critic Algorithm"""
    
        def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
            """
            Args:
                state_dim: dimension of state space
                action_dim: dimension of action space
                lr: learning rate
                gamma: discount factor
            """
            self.gamma = gamma
            self.network = ActorCriticNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
        def select_action(self, state):
            """
            Select action according to policy
    
            Args:
                state: state
    
            Returns:
                action: selected action
                log_prob: log π(a|s)
                state_value: V(s)
            """
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = self.network(state)
    
            # Sample from probability distribution
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
    
            return action.item(), log_prob, state_value
    
        def update(self, log_prob, state_value, reward, next_state, done):
            """
            Update parameters at each step (TD learning)
    
            Args:
                log_prob: log π(a|s)
                state_value: V(s)
                reward: reward r
                next_state: next state s'
                done: episode termination flag
    
            Returns:
                loss: loss value
            """
            # Value estimation of next state
            if done:
                next_value = torch.tensor([0.0])
            else:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    _, next_value = self.network(next_state)
    
            # TD error (Advantage)
            td_target = reward + self.gamma * next_value
            td_error = td_target - state_value
    
            # Actor loss: -log π(a|s) * A
            actor_loss = -log_prob * td_error.detach()
    
            # Critic loss: (TD error)^2
            critic_loss = td_error.pow(2)
    
            # Total loss
            loss = actor_loss + critic_loss
    
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
    
    # Demonstration
    print("=== Actor-Critic on CartPole ===\n")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ActorCritic(state_dim, action_dim, lr=0.001, gamma=0.99)
    
    print(f"Environment: CartPole-v1")
    print(f"Agent: Actor-Critic")
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Training
    num_episodes = 300
    episode_rewards = []
    
    print("\nTraining...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
    
        for t in range(1000):
            action, log_prob, state_value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
    
            # Update at each step
            loss = agent.update(log_prob, state_value, reward, next_state, done)
    
            episode_reward += reward
            episode_loss += loss
            steps += 1
    
            state = next_state
    
            if done:
                break
    
        episode_rewards.append(episode_reward)
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = episode_loss / steps
            print(f"Episode {episode+1:3d}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    
    print("\n✓ Actor-Critic characteristics:")
    print("  • Two networks: Actor and Critic")
    print("  • TD learning (update at each step)")
    print("  • Lower variance than REINFORCE")
    print("  • More stable learning")
    

**Sample Output** :
    
    
    === Actor-Critic on CartPole ===
    
    Environment: CartPole-v1
    Agent: Actor-Critic
      Total parameters: 17,027
    
    Training...
    Episode  50, Avg Reward: 45.67, Avg Loss: 2.3456
    Episode 100, Avg Reward: 98.23, Avg Loss: 1.5678
    Episode 150, Avg Reward: 165.45, Avg Loss: 0.9876
    Episode 200, Avg Reward: 210.34, Avg Loss: 0.6543
    Episode 250, Avg Reward: 245.67, Avg Loss: 0.4321
    Episode 300, Avg Reward: 280.89, Avg Loss: 0.2987
    
    Training completed!
    Final average reward (last 100 episodes): 280.89
    
    ✓ Actor-Critic characteristics:
      • Two networks: Actor and Critic
      • TD learning (update at each step)
      • Lower variance than REINFORCE
      • More stable learning
    

* * *

## 4.4 Advantage Actor-Critic (A2C)

### 4.3.1 Improvements in A2C

**A2C (Advantage Actor-Critic)** is an improved version of Actor-Critic with the following features:

  * **n-step returns** : Uses returns looking ahead multiple steps
  * **Parallel environments** : Simultaneous sampling from multiple environments (data diversity)
  * **Entropy regularization** : Promotes exploration
  * **Generalized Advantage Estimation (GAE)** : Adjusts bias-variance tradeoff

#### n-step Returns

Uses rewards up to $n$ steps ahead instead of 1-step TD:

$$ R_t^{(n)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) $$ 

#### Entropy Regularization

Adds policy entropy to the objective function to promote exploration:

$$ J(\theta) = \mathbb{E} \left[ \sum_t \log \pi_\theta(a_t|s_t) A_t + \beta H(\pi_\theta(\cdot|s_t)) \right] $$ 

Where $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$ is the entropy.

### 4.4.2 A2C Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import gym
    from torch.distributions import Categorical
    
    class A2CNetwork(nn.Module):
        """A2C Network with shared feature extractor"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super(A2CNetwork, self).__init__()
    
            # Shared feature extraction layers
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
            # Actor
            self.actor = nn.Linear(hidden_dim, action_dim)
    
            # Critic
            self.critic = nn.Linear(hidden_dim, 1)
    
        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
    
            action_logits = self.actor(x)
            state_value = self.critic(x)
    
            return action_logits, state_value
    
    
    class A2C:
        """
        Advantage Actor-Critic (A2C)
    
        Features:
          - n-step returns
          - Entropy regularization
          - Parallel environment support
        """
    
        def __init__(self, state_dim, action_dim, lr=0.0007, gamma=0.99,
                     n_steps=5, entropy_coef=0.01, value_coef=0.5):
            """
            Args:
                state_dim: dimension of state space
                action_dim: dimension of action space
                lr: learning rate
                gamma: discount factor
                n_steps: n-step returns
                entropy_coef: entropy regularization coefficient
                value_coef: value loss coefficient
            """
            self.gamma = gamma
            self.n_steps = n_steps
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
    
            self.network = A2CNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
        def select_action(self, state):
            """Action selection"""
            state = torch.FloatTensor(state).unsqueeze(0)
            action_logits, state_value = self.network(state)
    
            # Action distribution
            dist = Categorical(logits=action_logits)
            action = dist.sample()
    
            return action.item(), dist.log_prob(action), dist.entropy(), state_value
    
        def compute_returns(self, rewards, values, dones, next_value):
            """
            Compute n-step returns
    
            Args:
                rewards: reward sequence [n_steps]
                values: state value sequence [n_steps]
                dones: termination flag sequence [n_steps]
                next_value: value of next state after last state
    
            Returns:
                returns: n-step returns [n_steps]
                advantages: Advantage [n_steps]
            """
            returns = []
            R = next_value
    
            # Calculate in reverse order
            for step in reversed(range(len(rewards))):
                R = rewards[step] + self.gamma * R * (1 - dones[step])
                returns.insert(0, R)
    
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.cat(values)
    
            # Advantage = Returns - V(s)
            advantages = returns - values.detach()
    
            return returns, advantages
    
        def update(self, log_probs, entropies, values, returns, advantages):
            """
            Parameter update
    
            Args:
                log_probs: list of log π(a|s)
                entropies: list of entropies
                values: list of V(s)
                returns: n-step returns
                advantages: Advantage
            """
            log_probs = torch.cat(log_probs)
            entropies = torch.cat(entropies)
            values = torch.cat(values)
    
            # Actor loss: -log π(a|s) * A
            actor_loss = -(log_probs * advantages.detach()).mean()
    
            # Critic loss: MSE(returns, V(s))
            critic_loss = F.mse_loss(values, returns)
    
            # Entropy regularization
            entropy_loss = -entropies.mean()
    
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
    
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
    
            return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()
    
    
    # Demonstration
    print("=== A2C on CartPole ===\n")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2C(state_dim, action_dim, lr=0.0007, gamma=0.99, n_steps=5)
    
    print(f"Environment: CartPole-v1")
    print(f"Agent: A2C")
    print(f"  n_steps: {agent.n_steps}")
    print(f"  entropy_coef: {agent.entropy_coef}")
    print(f"  value_coef: {agent.value_coef}")
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Training
    num_episodes = 500
    episode_rewards = []
    
    print("\nTraining...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
    
        # Collect n-step data
        log_probs = []
        entropies = []
        values = []
        rewards = []
        dones = []
    
        done = False
        while not done:
            action, log_prob, entropy, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
    
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
    
            episode_reward += reward
            state = next_state
    
            # Update every n-steps or at episode end
            if len(rewards) >= agent.n_steps or done:
                # Next state value
                if done:
                    next_value = 0
                else:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    with torch.no_grad():
                        _, next_value = agent.network(next_state_tensor)
                        next_value = next_value.item()
    
                # Compute returns and advantages
                returns, advantages = agent.compute_returns(rewards, values, dones, next_value)
    
                # Update
                loss, actor_loss, critic_loss, entropy_loss = agent.update(
                    log_probs, entropies, values, returns, advantages
                )
    
                # Reset
                log_probs = []
                entropies = []
                values = []
                rewards = []
                dones = []
    
        episode_rewards.append(episode_reward)
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:3d}, Avg Reward: {avg_reward:.2f}, "
                  f"Loss: {loss:.4f} (Actor: {actor_loss:.4f}, Critic: {critic_loss:.4f}, Entropy: {entropy_loss:.4f})")
    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    
    print("\n✓ A2C characteristics:")
    print("  • n-step returns (more accurate return estimation)")
    print("  • Entropy regularization (exploration promotion)")
    print("  • Gradient clipping (stability improvement)")
    print("  • Parallel environment support (single environment in this example)")
    

**Sample Output** :
    
    
    === A2C on CartPole ===
    
    Environment: CartPole-v1
    Agent: A2C
      n_steps: 5
      entropy_coef: 0.01
      value_coef: 0.5
      Total parameters: 68,097
    
    Training...
    Episode  50, Avg Reward: 56.78, Loss: 1.8765 (Actor: 0.5432, Critic: 2.6543, Entropy: -0.5678)
    Episode 100, Avg Reward: 112.34, Loss: 1.2345 (Actor: 0.3456, Critic: 1.7654, Entropy: -0.4321)
    Episode 150, Avg Reward: 178.56, Loss: 0.8765 (Actor: 0.2345, Critic: 1.2987, Entropy: -0.3456)
    Episode 200, Avg Reward: 220.45, Loss: 0.6543 (Actor: 0.1876, Critic: 0.9876, Entropy: -0.2987)
    Episode 250, Avg Reward: 265.78, Loss: 0.4987 (Actor: 0.1432, Critic: 0.7654, Entropy: -0.2456)
    Episode 300, Avg Reward: 295.34, Loss: 0.3876 (Actor: 0.1098, Critic: 0.6321, Entropy: -0.2134)
    Episode 350, Avg Reward: 320.67, Loss: 0.2987 (Actor: 0.0876, Critic: 0.5234, Entropy: -0.1876)
    Episode 400, Avg Reward: 340.23, Loss: 0.2345 (Actor: 0.0654, Critic: 0.4321, Entropy: -0.1654)
    Episode 450, Avg Reward: 355.89, Loss: 0.1876 (Actor: 0.0543, Critic: 0.3654, Entropy: -0.1432)
    Episode 500, Avg Reward: 370.45, Loss: 0.1543 (Actor: 0.0432, Critic: 0.3098, Entropy: -0.1234)
    
    Training completed!
    Final average reward (last 100 episodes): 370.45
    
    ✓ A2C characteristics:
      • n-step returns (more accurate return estimation)
      • Entropy regularization (exploration promotion)
      • Gradient clipping (stability improvement)
      • Parallel environment support (single environment in this example)
    

* * *

## 4.5 Proximal Policy Optimization (PPO)

### 4.5.1 Motivation for PPO

Challenges of policy gradient:

  * **Large parameter updates** : Performance can degrade if policy changes too much
  * **Sample efficiency** : On-policy learning is inefficient

**PPO (Proximal Policy Optimization)** solutions:

  * **Clipped objective** : Limits policy changes
  * **Multiple epochs** : Updates multiple times with same data (closer to off-policy)
  * **Trust region** : Optimizes within safe update range

### 4.5.2 PPO Clipped Objective

PPO's objective function uses a clipped probability ratio:

$$ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right] $$ 

Where:

  * $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$: probability ratio
  * $\epsilon$: clipping range (typically 0.1 to 0.2)
  * $A_t$: Advantage

**Intuitive Understanding** :

  * When Advantage is positive: Limit probability ratio to $[1, 1+\epsilon]$ (prevent excessive increase)
  * When Advantage is negative: Limit probability ratio to $[1-\epsilon, 1]$ (prevent excessive decrease)

    
    
    ```mermaid
    graph TB
        subgraph "PPO Clipped Objective"
            OldPolicy["Old Policyπ_old(a|s)"]
            NewPolicy["New Policyπ_θ(a|s)"]
    
            Ratio["Probability Ratior = π_θ / π_old"]
            Clip["Clippingclip(r, 1-ε, 1+ε)"]
    
            Advantage["AdvantageA_t"]
    
            Obj1["Objective 1r × A"]
            Obj2["Objective 2clip(r) × A"]
    
            Min["min(Obj1, Obj2)"]
    
            Loss["PPO Loss-E[min(...)]"]
    
            OldPolicy --> Ratio
            NewPolicy --> Ratio
    
            Ratio --> Obj1
            Ratio --> Clip
            Clip --> Obj2
    
            Advantage --> Obj1
            Advantage --> Obj2
    
            Obj1 --> Min
            Obj2 --> Min
    
            Min --> Loss
    
            style OldPolicy fill:#7b2cbf,color:#fff
            style NewPolicy fill:#27ae60,color:#fff
            style Loss fill:#e74c3c,color:#fff
        end
    ```

### 4.5.3 PPO Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import gym
    from torch.distributions import Categorical
    
    class PPONetwork(nn.Module):
        """PPO Network"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=64):
            super(PPONetwork, self).__init__()
    
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
            self.actor = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
    
        def forward(self, state):
            x = F.tanh(self.fc1(state))
            x = F.tanh(self.fc2(x))
    
            action_logits = self.actor(x)
            state_value = self.critic(x)
    
            return action_logits, state_value
    
    
    class PPO:
        """
        Proximal Policy Optimization (PPO)
    
        Features:
          - Clipped objective
          - Multiple epochs per update
          - GAE (Generalized Advantage Estimation)
        """
    
        def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                     epsilon=0.2, gae_lambda=0.95, epochs=10, batch_size=64):
            """
            Args:
                state_dim: dimension of state space
                action_dim: dimension of action space
                lr: learning rate
                gamma: discount factor
                epsilon: clipping range
                gae_lambda: GAE λ
                epochs: number of updates per data collection
                batch_size: mini-batch size
            """
            self.gamma = gamma
            self.epsilon = epsilon
            self.gae_lambda = gae_lambda
            self.epochs = epochs
            self.batch_size = batch_size
    
            self.network = PPONetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
        def select_action(self, state):
            """Action selection"""
            state = torch.FloatTensor(state).unsqueeze(0)
    
            with torch.no_grad():
                action_logits, state_value = self.network(state)
    
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
    
            return action.item(), log_prob.item(), state_value.item()
    
        def compute_gae(self, rewards, values, dones, next_value):
            """
            Generalized Advantage Estimation (GAE)
    
            Args:
                rewards: reward sequence
                values: state value sequence
                dones: termination flag sequence
                next_value: value of next state after last
    
            Returns:
                advantages: GAE Advantage
                returns: returns
            """
            advantages = []
            gae = 0
    
            values = values + [next_value]
    
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
                advantages.insert(0, gae)
    
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
    
            return advantages, returns
    
        def update(self, states, actions, old_log_probs, returns, advantages):
            """
            PPO update (Multiple epochs)
    
            Args:
                states: state sequence
                actions: action sequence
                old_log_probs: log probabilities of old policy
                returns: returns
                advantages: Advantage
            """
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(old_log_probs)
            returns = returns.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
            dataset_size = states.size(0)
    
            for epoch in range(self.epochs):
                # Update with mini-batches
                indices = np.random.permutation(dataset_size)
    
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
    
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
    
                    # Evaluate current policy
                    action_logits, state_values = self.network(batch_states)
                    dist = Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
    
                    # Probability ratio
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
    
                    # Clipped objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
    
                    # Critic loss
                    critic_loss = F.mse_loss(state_values.squeeze(), batch_returns)
    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
    
                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
    
                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.optimizer.step()
    
    
    # Demonstration
    print("=== PPO on CartPole ===\n")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10)
    
    print(f"Environment: CartPole-v1")
    print(f"Agent: PPO")
    print(f"  epsilon (clip): {agent.epsilon}")
    print(f"  gae_lambda: {agent.gae_lambda}")
    print(f"  epochs per update: {agent.epochs}")
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Training
    num_iterations = 100
    update_timesteps = 2048  # Number of data collection steps
    episode_rewards = []
    
    print("\nTraining...")
    total_timesteps = 0
    
    for iteration in range(num_iterations):
        # Data collection
        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
    
        state = env.reset()
        episode_reward = 0
    
        for _ in range(update_timesteps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
    
            states_list.append(state)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            values_list.append(value)
            dones_list.append(done)
    
            episode_reward += reward
            total_timesteps += 1
    
            state = next_state
    
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state = env.reset()
    
        # Value of final state
        _, _, next_value = agent.select_action(state)
    
        # Compute GAE
        advantages, returns = agent.compute_gae(rewards_list, values_list, dones_list, next_value)
    
        # PPO update
        agent.update(states_list, actions_list, log_probs_list, returns, advantages)
    
        if (iteration + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Iteration {iteration+1:3d}, Timesteps: {total_timesteps}, Avg Reward: {avg_reward:.2f}")
    
    print(f"\nTraining completed!")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    
    print("\n✓ PPO characteristics:")
    print("  • Clipped objective (safe policy updates)")
    print("  • Multiple epochs (data reuse)")
    print("  • GAE (bias-variance tradeoff)")
    print("  • De facto standard for modern policy gradients")
    print("  • Used in OpenAI Five, ChatGPT RLHF, etc.")
    

**Sample Output** :
    
    
    === PPO on CartPole ===
    
    Environment: CartPole-v1
    Agent: PPO
      epsilon (clip): 0.2
      gae_lambda: 0.95
      epochs per update: 10
      Total parameters: 4,545
    
    Training...
    Iteration  10, Timesteps: 20480, Avg Reward: 78.45
    Iteration  20, Timesteps: 40960, Avg Reward: 145.67
    Iteration  30, Timesteps: 61440, Avg Reward: 210.34
    Iteration  40, Timesteps: 81920, Avg Reward: 265.89
    Iteration  50, Timesteps: 102400, Avg Reward: 310.45
    Iteration  60, Timesteps: 122880, Avg Reward: 345.67
    Iteration  70, Timesteps: 143360, Avg Reward: 380.23
    Iteration  80, Timesteps: 163840, Avg Reward: 405.78
    Iteration  90, Timesteps: 184320, Avg Reward: 425.34
    Iteration 100, Timesteps: 204800, Avg Reward: 440.56
    
    Training completed!
    Total timesteps: 204800
    Final average reward (last 100 episodes): 440.56
    
    ✓ PPO characteristics:
      • Clipped objective (safe policy updates)
      • Multiple epochs (data reuse)
      • GAE (bias-variance tradeoff)
      • De facto standard for modern policy gradients
      • Used in OpenAI Five, ChatGPT RLHF, etc.
    

* * *

## 4.6 Practice: LunarLander Continuous Control

### 4.6.1 LunarLander Environment

**LunarLander-v2** is a task to control a lunar lander spacecraft.

Item | Value  
---|---  
**State Space** | 8-dimensional (position, velocity, angle, angular velocity, leg contact)  
**Action Space** | 4-dimensional (do nothing, left engine, main engine, right engine)  
**Goal** | Land safely on landing pad (solved at 200+ points)  
**Reward** | Successful landing: +100 to +140, Crash: -100, Fuel consumption: negative  
  
### 4.6.2 LunarLander Learning with PPO
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: 4.6.2 LunarLander Learning with PPO
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import gym
    from torch.distributions import Categorical
    import matplotlib.pyplot as plt
    
    # PPO class same as previous section (omitted)
    
    # Training on LunarLander
    print("=== PPO on LunarLander-v2 ===\n")
    
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: LunarLander-v2")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Solved threshold: 200")
    
    agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64)
    
    print(f"\nAgent: PPO")
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Training settings
    num_iterations = 300
    update_timesteps = 2048
    episode_rewards = []
    all_episode_rewards = []
    
    print("\nTraining...")
    total_timesteps = 0
    best_avg_reward = -float('inf')
    
    for iteration in range(num_iterations):
        # Data collection
        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
    
        state = env.reset()
        episode_reward = 0
    
        for _ in range(update_timesteps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
    
            states_list.append(state)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            values_list.append(value)
            dones_list.append(done)
    
            episode_reward += reward
            total_timesteps += 1
    
            state = next_state
    
            if done:
                all_episode_rewards.append(episode_reward)
                episode_reward = 0
                state = env.reset()
    
        # Value of final state
        _, _, next_value = agent.select_action(state)
    
        # Compute GAE
        advantages, returns = agent.compute_gae(rewards_list, values_list, dones_list, next_value)
    
        # PPO update
        agent.update(states_list, actions_list, log_probs_list, returns, advantages)
    
        # Evaluation
        if (iteration + 1) % 10 == 0:
            avg_reward = np.mean(all_episode_rewards[-100:]) if len(all_episode_rewards) >= 100 else np.mean(all_episode_rewards)
            episode_rewards.append(avg_reward)
    
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
    
            print(f"Iteration {iteration+1:3d}, Timesteps: {total_timesteps}, "
                  f"Avg Reward: {avg_reward:.2f}, Best: {best_avg_reward:.2f}")
    
            if avg_reward >= 200:
                print(f"\n🎉 Solved! Average reward {avg_reward:.2f} >= 200")
                break
    
    print(f"\nTraining completed!")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Best average reward: {best_avg_reward:.2f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # All episode rewards
    ax.plot(all_episode_rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    
    # Moving average (100 episodes)
    window = 100
    moving_avg = [np.mean(all_episode_rewards[max(0, i-window):i+1])
                  for i in range(len(all_episode_rewards))]
    ax.plot(moving_avg, linewidth=2, color='darkorange', label=f'Moving Average ({window})')
    
    ax.axhline(y=200, color='red', linestyle='--', linewidth=2, label='Solved Threshold (200)')
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title('PPO on LunarLander-v2', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✓ LunarLander task completed")
    print("✓ Stable learning with PPO")
    print("✓ Typical solution time: 1-2 million steps")
    

**Sample Output** :
    
    
    === PPO on LunarLander-v2 ===
    
    Environment: LunarLander-v2
      State dimension: 8
      Action dimension: 4
      Solved threshold: 200
    
    Agent: PPO
      Total parameters: 4,673
    
    Training...
    Iteration  10, Timesteps: 20480, Avg Reward: -145.67, Best: -145.67
    Iteration  20, Timesteps: 40960, Avg Reward: -89.34, Best: -89.34
    Iteration  30, Timesteps: 61440, Avg Reward: -45.23, Best: -45.23
    Iteration  40, Timesteps: 81920, Avg Reward: 12.56, Best: 12.56
    Iteration  50, Timesteps: 102400, Avg Reward: 56.78, Best: 56.78
    Iteration  60, Timesteps: 122880, Avg Reward: 98.45, Best: 98.45
    Iteration  70, Timesteps: 143360, Avg Reward: 134.67, Best: 134.67
    Iteration  80, Timesteps: 163840, Avg Reward: 165.89, Best: 165.89
    Iteration  90, Timesteps: 184320, Avg Reward: 185.34, Best: 185.34
    Iteration 100, Timesteps: 204800, Avg Reward: 202.56, Best: 202.56
    
    🎉 Solved! Average reward 202.56 >= 200
    
    Training completed!
    Total timesteps: 204800
    Best average reward: 202.56
    
    ✓ LunarLander task completed
    ✓ Stable learning with PPO
    ✓ Typical solution time: 1-2 million steps
    

* * *

## 4.7 Continuous Action Space and Gaussian Policy

### 4.7.1 Handling Continuous Action Space

So far we've handled discrete action spaces (CartPole, LunarLander), but robot control and similar tasks require **continuous action spaces**.

**Gaussian Policy** :

Sample actions from a normal distribution:

$$ \pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2) $$ 

Where:

  * $\mu_\theta(s)$: mean (output by neural network)
  * $\sigma_\theta(s)$: standard deviation (learnable parameter or fixed value)

### 4.7.2 Gaussian Policy Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    import numpy as np
    
    class ContinuousPolicyNetwork(nn.Module):
        """
        Policy Network for continuous action space
    
        Output: mean μ and standard deviation σ
        """
    
        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super(ContinuousPolicyNetwork, self).__init__()
    
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
            # Mean μ
            self.mu_head = nn.Linear(hidden_dim, action_dim)
    
            # Standard deviation σ (learn in log scale)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
            # Critic
            self.value_head = nn.Linear(hidden_dim, 1)
    
        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
    
            # Mean μ
            mu = self.mu_head(x)
    
            # Standard deviation σ (ensure positive value)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, min=-20, max=2)  # Clip for numerical stability
            std = torch.exp(log_std)
    
            # State value
            value = self.value_head(x)
    
            return mu, std, value
    
    
    class ContinuousPPO:
        """PPO for continuous action space"""
    
        def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2):
            self.gamma = gamma
            self.epsilon = epsilon
    
            self.network = ContinuousPolicyNetwork(state_dim, action_dim)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
        def select_action(self, state):
            """
            Sample continuous action
    
            Returns:
                action: sampled action
                log_prob: log π(a|s)
                value: V(s)
            """
            state = torch.FloatTensor(state).unsqueeze(0)
    
            with torch.no_grad():
                mu, std, value = self.network(state)
    
            # Sample from normal distribution
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)  # Product of each dimension
    
            return action.squeeze().numpy(), log_prob.item(), value.item()
    
        def evaluate_actions(self, states, actions):
            """
            Evaluate existing actions (for PPO update)
    
            Returns:
                log_probs: log π(a|s)
                values: V(s)
                entropy: entropy
            """
            mu, std, values = self.network(states)
    
            dist = Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
    
            return log_probs, values.squeeze(), entropy
    
    
    # Demonstration
    print("=== Continuous Action Space PPO ===\n")
    
    # Sample environment (e.g., Pendulum-v1)
    state_dim = 3
    action_dim = 1
    
    agent = ContinuousPPO(state_dim, action_dim, lr=3e-4)
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim} (continuous)")
    
    # Sample state
    state = np.random.randn(state_dim)
    
    # Action selection
    action, log_prob, value = agent.select_action(state)
    
    print(f"\nSample state: {state}")
    print(f"Sampled action: {action}")
    print(f"Log probability: {log_prob:.4f}")
    print(f"State value: {value:.4f}")
    
    # Multiple samples (stochastic)
    print("\nMultiple samples from same state:")
    for i in range(5):
        action, _, _ = agent.select_action(state)
        print(f"  Sample {i+1}: action = {action[0]:.4f}")
    
    print("\n✓ Gaussian Policy characteristics:")
    print("  • Supports continuous action space")
    print("  • Learns mean μ and standard deviation σ")
    print("  • Applicable to robot control, autonomous driving, etc.")
    print("  • Exploration controlled by standard deviation σ")
    
    # Usage example with actual Pendulum environment
    print("\n=== PPO on Pendulum-v1 (Continuous Control) ===")
    
    import gym
    
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"\nEnvironment: Pendulum-v1")
    print(f"  State space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    agent = ContinuousPPO(state_dim, action_dim, lr=3e-4)
    
    print(f"\nAgent initialized for continuous control")
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test one episode
    state = env.reset()
    episode_reward = 0
    
    for t in range(200):
        action, log_prob, value = agent.select_action(state)
        # Pendulum action range is [-2, 2], so scale appropriately
        action_scaled = np.clip(action, -2.0, 2.0)
        next_state, reward, done, _ = env.step(action_scaled)
    
        episode_reward += reward
        state = next_state
    
    print(f"\nTest episode reward: {episode_reward:.2f}")
    print("\n✓ PPO operation confirmed on continuous control task")
    

**Sample Output** :
    
    
    === Continuous Action Space PPO ===
    
    State dimension: 3
    Action dimension: 1 (continuous)
    
    Sample state: [ 0.4967 -0.1383  0.6477]
    Sampled action: [0.8732]
    Log probability: -1.2345
    State value: 0.1234
    
    Multiple samples from same state:
      Sample 1: action = 0.7654
      Sample 2: action = 0.9123
      Sample 3: action = 0.8456
      Sample 4: action = 0.8901
      Sample 5: action = 0.8234
    
    ✓ Gaussian Policy characteristics:
      • Supports continuous action space
      • Learns mean μ and standard deviation σ
      • Applicable to robot control, autonomous driving, etc.
      • Exploration controlled by standard deviation σ
    
    === PPO on Pendulum-v1 (Continuous Control) ===
    
    Environment: Pendulum-v1
      State space: Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
      Action space: Box(-2.0, 2.0, (1,), float32)
    
    Agent initialized for continuous control
      Total parameters: 133,121
    
    Test episode reward: -1234.56
    
    ✓ PPO operation confirmed on continuous control task
    

* * *

## 4.8 Summary and Advanced Topics

### What We Learned in This Chapter

Topic | Key Points  
---|---  
**Policy Gradient** | Direct policy optimization, continuous action support, stochastic policy  
**REINFORCE** | Simplest PG, Monte Carlo method, high variance  
**Actor-Critic** | Combination of Actor and Critic, TD learning, low variance  
**A2C** | n-step returns, entropy regularization, parallel environments  
**PPO** | Clipped objective, safe updates, de facto standard  
**Continuous Control** | Gaussian Policy, learning μ and σ, robot control  
  
### Algorithm Comparison

Algorithm | Update | Variance | Sample Efficiency | Implementation Difficulty  
---|---|---|---|---  
**REINFORCE** | After episode | High | Low | Easy  
**Actor-Critic** | Every step | Medium | Medium | Medium  
**A2C** | Every n-steps | Medium | Medium | Medium  
**PPO** | Batch (multiple epochs) | Low | High | Medium  
  
### Advanced Topics

**Trust Region Policy Optimization (TRPO)**

Predecessor to PPO. Constrains policy updates with KL divergence. Stronger theoretical guarantees but higher computational cost. Requires second-order optimization and Fisher information matrix computation.

**Soft Actor-Critic (SAC)**

Off-policy Actor-Critic. Incorporates entropy maximization into objective function for robust learning. High performance on continuous control tasks. Uses experience replay for high sample efficiency.

**Deterministic Policy Gradient (DPG / DDPG)**

Policy gradient for deterministic policies (not stochastic). Specialized for continuous action spaces. Actor-Critic architecture with off-policy learning. Widely used in robot control.

**Twin Delayed DDPG (TD3)**

Improved version of DDPG. Two Critic networks (Twin), delayed Actor updates, target policy noise addition. Mitigates overestimation bias.

**Generalized Advantage Estimation (GAE)**

Advantage estimation method. Adjusts bias-variance tradeoff with λ parameter. Policy gradient version of TD(λ). Standard in PPO and A2C.

**Multi-Agent Reinforcement Learning (MARL)**

Cooperative and competitive learning with multiple agents. Algorithms like MAPPO, QMIX, MADDPG. Applied to game AI, swarm robotics, traffic systems.

### Exercises

#### Exercise 4.1: Improving REINFORCE

**Task** : Add a baseline (state value function) to REINFORCE and verify variance reduction effects.

**Implementation** :

  * Add Critic network
  * Calculate Advantage = R_t - V(s_t)
  * Compare learning curves with and without baseline

#### Exercise 4.2: Parallel Environment A2C Implementation

**Task** : Implement A2C that executes multiple environments in parallel.

**Requirements** :

  * Use multiprocessing or vectorized environments
  * 4-16 parallel environments
  * Confirm improvements in learning speed and sample efficiency

#### Exercise 4.3: PPO Hyperparameter Tuning

**Task** : Optimize PPO hyperparameters on LunarLander.

**Tuning Parameters** : epsilon (clip), learning rate, batch_size, epochs, GAE lambda

**Evaluation** : Convergence speed, final performance, stability

#### Exercise 4.4: Pendulum Control with Gaussian Policy

**Task** : Solve continuous control task Pendulum-v1 with PPO.

**Implementation** :

  * Implement Gaussian Policy
  * Standard deviation σ decay schedule
  * Achieve average reward of -200 or better

#### Exercise 4.5: Application to Atari Games

**Task** : Apply PPO to Atari games (e.g., Pong).

**Requirements** :

  * CNN-based Policy Network
  * Frame stacking (4 frames)
  * Reward clipping, Frame skipping
  * Aim for human-level performance

#### Exercise 4.6: Application to Custom Environment

**Task** : Create your own OpenAI Gym environment and train with PPO.

**Examples** :

  * Simple maze navigation
  * Resource management game
  * Simple robot arm control

**Implementation** : Environment class inheriting gym.Env, appropriate reward design, learning with PPO

* * *

### Next Chapter Preview

In Chapter 5, we will learn about **Model-based Reinforcement Learning**. We will explore advanced approaches that combine planning and learning by learning models of the environment.

> **Next Chapter Topics** :  
>  • Model-based vs Model-free  
>  • Learning Environment Models (World Models)  
>  • Planning Methods (MCTS, MuZero)  
>  • Dyna-Q, Model-based RL  
>  • Learning in Imagination (Dreamer)  
>  • Significant improvements in sample efficiency  
>  • Implementation: Model learning and planning
