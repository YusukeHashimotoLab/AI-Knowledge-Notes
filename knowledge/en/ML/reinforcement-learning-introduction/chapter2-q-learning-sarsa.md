---
title: "Chapter 2: Q-Learning and SARSA"
chapter_title: "Chapter 2: Q-Learning and SARSA"
subtitle: Value Function Estimation and Policy Optimization using Temporal Difference Learning
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
code_examples: 8
exercises: 5
---

This chapter covers Q. You will learn Explaining the mechanism, Understanding the characteristics, and Balancing exploration.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understanding the basic principles of Temporal Difference (TD) learning and its differences from Monte Carlo methods
  * ✅ Explaining the mechanism and update equations of the Q-Learning algorithm (off-policy)
  * ✅ Understanding the characteristics and application scenarios of the SARSA algorithm (on-policy)
  * ✅ Balancing exploration and exploitation using ε-greedy policies
  * ✅ Explaining the effects of hyperparameters: learning rate and discount factor
  * ✅ Implementing solutions for OpenAI Gym's Taxi-v3 and Cliff Walking environments

* * *

## 2.1 Fundamentals of Temporal Difference (TD) Learning

### Challenges of Monte Carlo Methods

The Monte Carlo method we learned in Chapter 1 had the constraint of **needing to wait until the end of an episode** :

$$ V(s_t) \leftarrow V(s_t) + \alpha [G_t - V(s_t)] $$ 

where $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$ is the actual return.

### Basic Idea of Temporal Difference Learning

**Temporal Difference (TD) learning** **updates the value function at each step** without waiting for the episode to end:

$$ V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)] $$ 

where:

  * $r_{t+1} + \gamma V(s_{t+1})$: **TD target**
  * $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$: **TD error**
  * $\alpha$: learning rate

    
    
    ```mermaid
    graph LR
        S1["State s_t"] --> A["Action a_t"]
        A --> S2["State s_t+1"]
        S2 --> R["Reward r_t+1"]
        R --> Update["Update V(s_t)"]
        S2 --> Update
    
        style S1 fill:#b3e5fc
        style S2 fill:#c5e1a5
        style R fill:#fff9c4
        style Update fill:#ffab91
    ```

### Implementation of TD(0)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import gym
    
    def td_0_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.99):
        """
        State value function estimation using TD(0)
    
        Args:
            env: Environment
            policy: Policy (state -> action probability distribution)
            num_episodes: Number of episodes
            alpha: Learning rate
            gamma: Discount factor
    
        Returns:
            V: State value function
        """
        # Initialize state value function
        V = np.zeros(env.observation_space.n)
    
        for episode in range(num_episodes):
            state, _ = env.reset()
    
            while True:
                # Select action according to policy
                action = np.random.choice(env.action_space.n, p=policy[state])
    
                # Interact with environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    
                # TD(0) update
                td_target = reward + gamma * V[next_state]
                td_error = td_target - V[state]
                V[state] = V[state] + alpha * td_error
    
                if done:
                    break
    
                state = next_state
    
        return V
    
    
    # Example usage: FrozenLake environment
    print("=== Value Function Estimation using TD(0) ===")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Random policy
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    
    # Execute TD(0)
    V = td_0_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.99)
    
    print(f"State value function:\n{V.reshape(4, 4)}")
    env.close()
    

### Comparison of Monte Carlo Methods and TD Learning

Aspect | Monte Carlo Method | TD Learning  
---|---|---  
**Update Timing** | After episode ends | After each step  
**Return Calculation** | Actual return $G_t$ | Estimated return $r + \gamma V(s')$  
**Bias** | None (unbiased) | Present (depends on initial values)  
**Variance** | High | Low  
**Continuing Tasks** | Not applicable | Applicable  
**Convergence Speed** | Slow | Fast  
  
> "TD learning achieves efficient learning through bootstrapping (updating using its own estimates)"

* * *

## 2.2 Q-Learning

### Action-Value Function Q(s, a)

Instead of the state value function $V(s)$, we learn the **action-value function** $Q(s, a)$:

$$ Q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a] $$ 

This represents "the expected return after taking action $a$ in state $s$".

### Q-Learning Update Equation

**Q-learning** applies TD learning to the action-value function:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right] $$ 

Key points:

  * $\max_{a'} Q(s_{t+1}, a')$: Uses the value of the **best action** in the next state
  * **Off-policy type** : The actual action differs from the action used for updating
  * Can directly learn the optimal policy

    
    
    ```mermaid
    graph TB
        Start["State s, Action a"] --> Execute["Execute in environment"]
        Execute --> Observe["Observe s', r"]
        Observe --> MaxQ["max_a' Q(s', a')"]
        MaxQ --> Target["TD target = r + γ max Q(s', a')"]
        Target --> Update["Update Q(s,a)"]
        Update --> Next["Next step"]
    
        style Start fill:#b3e5fc
        style MaxQ fill:#fff59d
        style Update fill:#ffab91
    ```

### Implementation of Q-Learning Algorithm
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import gym
    
    class QLearningAgent:
        """Q-Learning Agent"""
    
        def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
            """
            Args:
                n_states: Number of states
                n_actions: Number of actions
                alpha: Learning rate
                gamma: Discount factor
                epsilon: ε for ε-greedy
            """
            self.Q = np.zeros((n_states, n_actions))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.n_actions = n_actions
    
        def select_action(self, state):
            """Select action using ε-greedy policy"""
            if np.random.rand() < self.epsilon:
                # Random action (exploration)
                return np.random.randint(self.n_actions)
            else:
                # Best action (exploitation)
                return np.argmax(self.Q[state])
    
        def update(self, state, action, reward, next_state, done):
            """Update Q-value"""
            if done:
                # Terminal state
                td_target = reward
            else:
                # Q-learning update equation
                td_target = reward + self.gamma * np.max(self.Q[next_state])
    
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
    
    
    def train_q_learning(env, agent, num_episodes=1000):
        """Train Q-learning"""
        episode_rewards = []
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
    
            while True:
                # Select action
                action = agent.select_action(state)
    
                # Execute in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    
                # Update Q-value
                agent.update(state, action, reward, next_state, done)
    
                total_reward += reward
    
                if done:
                    break
    
                state = next_state
    
            episode_rewards.append(total_reward)
    
            # Display progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
        return episode_rewards
    
    
    # Example usage: FrozenLake
    print("\n=== Training Q-Learning ===")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    rewards = train_q_learning(env, agent, num_episodes=1000)
    
    print(f"\nLearned Q-table (partial):")
    print(agent.Q[:16].reshape(4, 4, -1)[:, :, 0])  # Q-values for action 0
    env.close()
    

### Visualizing the Q-Table
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def visualize_q_table(Q, env_shape=(4, 4)):
        """Visualize Q-table"""
        n_states = Q.shape[0]
        n_actions = Q.shape[1]
    
        fig, axes = plt.subplots(1, n_actions, figsize=(16, 4))
    
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    
        for action in range(n_actions):
            Q_action = Q[:, action].reshape(env_shape)
    
            sns.heatmap(Q_action, annot=True, fmt='.2f', cmap='YlOrRd',
                       ax=axes[action], cbar=True, square=True)
            axes[action].set_title(f'Q-value: {action_names[action]}')
            axes[action].set_xlabel('Column')
            axes[action].set_ylabel('Row')
    
        plt.tight_layout()
        plt.savefig('q_table_visualization.png', dpi=150, bbox_inches='tight')
        print("Q-table saved: q_table_visualization.png")
        plt.close()
    
    
    # Visualize Q-table
    visualize_q_table(agent.Q)
    

### Visualizing Learning Curves
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_learning_curve(rewards, window=100):
        """Plot learning curve"""
        # Calculate moving average
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
        plt.figure(figsize=(12, 5))
    
        # Episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.3, label='Episode Reward')
        plt.plot(range(window-1, len(rewards)), smoothed_rewards,
                 linewidth=2, label=f'{window}-Episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Q-Learning Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # Cumulative rewards
        plt.subplot(1, 2, 2)
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, linewidth=2, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward')
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('q_learning_curve.png', dpi=150, bbox_inches='tight')
        print("Learning curve saved: q_learning_curve.png")
        plt.close()
    
    
    plot_learning_curve(rewards)
    

* * *

## 2.3 SARSA (State-Action-Reward-State-Action)

### Basic Principle of SARSA

**SARSA** is the **on-policy version** of Q-learning. It updates using the action actually taken:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right] $$ 

Key difference:

  * Q-learning: Uses $\max_{a'} Q(s_{t+1}, a')$ (best action)
  * SARSA: Uses $Q(s_{t+1}, a_{t+1})$ (action actually taken)

    
    
    ```mermaid
    graph LR
        S1["S_t"] --> A1["A_t"]
        A1 --> R["R_t+1"]
        R --> S2["S_t+1"]
        S2 --> A2["A_t+1"]
        A2 --> Update["Update Q(S_t, A_t)"]
    
        style S1 fill:#b3e5fc
        style A1 fill:#c5e1a5
        style R fill:#fff9c4
        style S2 fill:#b3e5fc
        style A2 fill:#c5e1a5
        style Update fill:#ffab91
    ```

### Comparison of Q-Learning and SARSA

Aspect | Q-Learning | SARSA  
---|---|---  
**Learning Type** | Off-policy | On-policy  
**Update Equation** | $r + \gamma \max_a Q(s', a)$ | $r + \gamma Q(s', a')$  
**Exploration Impact** | Does not affect learning | Affects learning  
**Convergence Target** | Optimal policy | Value of current policy  
**Safety** | Does not consider risk | Considers risk  
**Application Scenario** | Simulation environments | Real environment learning  
  
### Implementation of SARSA
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import gym
    
    class SARSAAgent:
        """SARSA Agent"""
    
        def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
            """
            Args:
                n_states: Number of states
                n_actions: Number of actions
                alpha: Learning rate
                gamma: Discount factor
                epsilon: ε for ε-greedy
            """
            self.Q = np.zeros((n_states, n_actions))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.n_actions = n_actions
    
        def select_action(self, state):
            """Select action using ε-greedy policy"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_actions)
            else:
                return np.argmax(self.Q[state])
    
        def update(self, state, action, reward, next_state, next_action, done):
            """Update Q-value (SARSA)"""
            if done:
                td_target = reward
            else:
                # SARSA update equation (uses action actually taken next)
                td_target = reward + self.gamma * self.Q[next_state, next_action]
    
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
    
    
    def train_sarsa(env, agent, num_episodes=1000):
        """Train SARSA"""
        episode_rewards = []
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            action = agent.select_action(state)  # Initial action selection
            total_reward = 0
    
            while True:
                # Execute in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    
                if not done:
                    # Select next action (characteristic of SARSA)
                    next_action = agent.select_action(next_state)
                else:
                    next_action = None
    
                # Update Q-value
                agent.update(state, action, reward, next_state, next_action, done)
    
                total_reward += reward
    
                if done:
                    break
    
                state = next_state
                action = next_action  # Transition to next action
    
            episode_rewards.append(total_reward)
    
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
        return episode_rewards
    
    
    # Example usage
    print("\n=== Training SARSA ===")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    sarsa_agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    sarsa_rewards = train_sarsa(env, sarsa_agent, num_episodes=1000)
    
    print(f"\nLearned Q-table (SARSA):")
    print(sarsa_agent.Q[:16].reshape(4, 4, -1)[:, :, 0])
    env.close()
    

* * *

## 2.4 ε-Greedy Exploration Strategy

### Exploration vs Exploitation Trade-off

In reinforcement learning, balancing **exploration** and **exploitation** is crucial:

  * **Exploration** : Try new states and actions to understand the environment
  * **Exploitation** : Select the best action based on current knowledge

### ε-Greedy Policy

The simplest exploration strategy:

$$ a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\\ \arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon \end{cases} $$ 

### Epsilon Decay
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class EpsilonGreedy:
        """ε-greedy policy (with decay functionality)"""
    
        def __init__(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
            """
            Args:
                epsilon_start: Initial ε
                epsilon_end: Minimum ε
                epsilon_decay: Decay rate
            """
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
    
        def select_action(self, Q, state, n_actions):
            """Select action"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(n_actions)
            else:
                return np.argmax(Q[state])
    
        def decay(self):
            """Decay ε"""
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    
    # Visualize epsilon decay patterns
    print("\n=== Visualizing ε Decay Patterns ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Different decay rates
    decay_rates = [0.99, 0.995, 0.999]
    
    for i, decay_rate in enumerate(decay_rates):
        epsilon_greedy = EpsilonGreedy(epsilon_start=1.0, epsilon_end=0.01,
                                       epsilon_decay=decay_rate)
        epsilons = [epsilon_greedy.epsilon]
    
        for _ in range(1000):
            epsilon_greedy.decay()
            epsilons.append(epsilon_greedy.epsilon)
    
        axes[i].plot(epsilons, linewidth=2)
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('ε')
        axes[i].set_title(f'Decay Rate = {decay_rate}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('epsilon_decay.png', dpi=150, bbox_inches='tight')
    print("ε decay patterns saved: epsilon_decay.png")
    plt.close()
    

### Other Exploration Strategies

#### Softmax (Boltzmann) Exploration

Probabilistic selection based on action values:

$$ P(a | s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)} $$ 

$\tau$ is the temperature parameter (higher means more random)

#### Upper Confidence Bound (UCB)

Exploration considering uncertainty:

$$ a = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a)}} \right] $$ 

$N(s,a)$ is the number of times action $a$ was selected, $c$ is the exploration coefficient

* * *

## 2.5 Impact of Hyperparameters

### Learning Rate α

The learning rate $\alpha$ controls the strength of updates:

  * **Large α (e.g., 0.5)** : Fast learning but unstable
  * **Small α (e.g., 0.01)** : Stable but slow convergence
  * **Recommended value** : 0.1 - 0.3

### Discount Factor γ

The discount factor $\gamma$ determines the importance of future rewards:

  * **γ = 0** : Only considers immediate rewards (myopic)
  * **γ → 1** : Considers distant future (far-sighted)
  * **Recommended value** : 0.95 - 0.99

### Implementation of Hyperparameter Search
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    import gym
    
    def hyperparameter_search(env_name, param_name, param_values, num_episodes=500):
        """Investigate the impact of hyperparameters"""
        results = {}
    
        for value in param_values:
            print(f"\nTraining with {param_name} = {value}...")
    
            env = gym.make(env_name)
    
            if param_name == 'alpha':
                agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                                      alpha=value, gamma=0.99, epsilon=0.1)
            elif param_name == 'gamma':
                agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                                      alpha=0.1, gamma=value, epsilon=0.1)
            elif param_name == 'epsilon':
                agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                                      alpha=0.1, gamma=0.99, epsilon=value)
    
            rewards = train_q_learning(env, agent, num_episodes=num_episodes)
            results[value] = rewards
            env.close()
    
        return results
    
    
    # Investigate impact of learning rate
    print("=== Investigating Impact of Learning Rate α ===")
    
    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
    alpha_results = hyperparameter_search('FrozenLake-v1', 'alpha', alpha_values)
    
    # Visualization
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    for alpha, rewards in alpha_results.items():
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(smoothed, label=f'α = {alpha}', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Impact of Learning Rate α')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Investigate impact of discount factor
    gamma_values = [0.5, 0.9, 0.95, 0.99, 0.999]
    gamma_results = hyperparameter_search('FrozenLake-v1', 'gamma', gamma_values)
    
    plt.subplot(1, 2, 2)
    for gamma, rewards in gamma_results.items():
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(smoothed, label=f'γ = {gamma}', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Impact of Discount Factor γ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_impact.png', dpi=150, bbox_inches='tight')
    print("\nHyperparameter impact saved: hyperparameter_impact.png")
    plt.close()
    

* * *

## 2.6 Practice: Taxi-v3 Environment

### Overview of Taxi-v3 Environment

**Taxi-v3** is an environment where a taxi picks up a passenger and delivers them to a destination:

  * **State space** : 500 states (5×5 grid × 5 passenger locations × 4 destinations)
  * **Action space** : 6 actions (up, down, left, right, pickup, dropoff)
  * **Rewards** : +20 for reaching correct destination, -1 per step, -10 for illegal pickup/dropoff

### Q-Learning on Taxi-v3
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Q-Learning on Taxi-v3
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import gym
    import matplotlib.pyplot as plt
    
    # Taxi-v3 environment
    print("=== Q-Learning on Taxi-v3 Environment ===")
    
    env = gym.make('Taxi-v3', render_mode=None)
    
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    print(f"Actions: {['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']}")
    
    # Q-learning agent
    taxi_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Training
    taxi_rewards = train_q_learning(env, taxi_agent, num_episodes=5000)
    
    # Learning curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    smoothed = np.convolve(taxi_rewards, np.ones(100)/100, mode='valid')
    plt.plot(smoothed, linewidth=2, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Taxi-v3 Q-Learning Learning Curve')
    plt.grid(True, alpha=0.3)
    
    # Calculate success rate
    success_rate = []
    window = 100
    for i in range(len(taxi_rewards) - window):
        success = np.sum(np.array(taxi_rewards[i:i+window]) > 0) / window
        success_rate.append(success)
    
    plt.subplot(1, 2, 2)
    plt.plot(success_rate, linewidth=2, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Task Success Rate (100-Episode Moving Average)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('taxi_training.png', dpi=150, bbox_inches='tight')
    print("Taxi training results saved: taxi_training.png")
    plt.close()
    
    env.close()
    

### Evaluating the Trained Agent
    
    
    def evaluate_agent(env, agent, num_episodes=100, render=False):
        """Evaluate trained agent"""
        total_rewards = []
        total_steps = []
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
    
            while steps < 200:  # Maximum steps
                # Select best action (no exploration)
                action = np.argmax(agent.Q[state])
    
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
    
                if terminated or truncated:
                    break
    
            total_rewards.append(episode_reward)
            total_steps.append(steps)
    
        return total_rewards, total_steps
    
    
    # Evaluation
    print("\n=== Evaluating Trained Agent ===")
    
    env = gym.make('Taxi-v3', render_mode=None)
    eval_rewards, eval_steps = evaluate_agent(env, taxi_agent, num_episodes=100)
    
    print(f"Average reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Average steps: {np.mean(eval_steps):.2f} ± {np.std(eval_steps):.2f}")
    print(f"Success rate: {np.sum(np.array(eval_rewards) > 0) / len(eval_rewards) * 100:.1f}%")
    
    env.close()
    

* * *

## 2.7 Practice: Cliff Walking Environment

### Definition of Cliff Walking Environment

**Cliff Walking** is an environment where you must reach the goal while avoiding cliffs. It clearly demonstrates the difference between Q-learning and SARSA:

  * **4×12 grid** : Start at bottom-left, goal at bottom-right
  * **Cliff area** : Central portion of bottom edge (stepping on it gives -100 penalty)
  * **Rewards** : -1 per step, -100 for cliff, 0 for goal

### Implementation on Cliff Walking Environment
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation on Cliff Walking Environment
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import gym
    
    # Cliff Walking environment
    print("=== Cliff Walking Environment ===")
    
    env = gym.make('CliffWalking-v0')
    
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    print(f"Grid size: 4×12")
    
    # Q-learning agent
    cliff_q_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.5,
        gamma=0.99,
        epsilon=0.1
    )
    
    # SARSA agent
    cliff_sarsa_agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.5,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Training
    print("\nTraining with Q-learning...")
    q_rewards = train_q_learning(env, cliff_q_agent, num_episodes=500)
    
    env = gym.make('CliffWalking-v0')
    print("\nTraining with SARSA...")
    sarsa_rewards = train_sarsa(env, cliff_sarsa_agent, num_episodes=500)
    
    # Comparison visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    smoothed_q = np.convolve(q_rewards, np.ones(10)/10, mode='valid')
    smoothed_sarsa = np.convolve(sarsa_rewards, np.ones(10)/10, mode='valid')
    
    plt.plot(smoothed_q, label='Q-Learning', linewidth=2)
    plt.plot(smoothed_sarsa, label='SARSA', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Cliff Walking: Q-Learning vs SARSA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Visualize policy (display with arrows)
    plt.subplot(1, 2, 2)
    
    def visualize_policy(Q, shape=(4, 12)):
        """Visualize learned policy"""
        policy = np.argmax(Q, axis=1)
        policy_grid = policy.reshape(shape)
    
        # Arrow directions
        arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(np.zeros(shape), cmap='Blues', alpha=0.3)
    
        for i in range(shape[0]):
            for j in range(shape[1]):
                state = i * shape[1] + j
                action = policy[state]
    
                # Display cliff area in red
                if i == 3 and 1 <= j <= 10:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=True, color='red', alpha=0.3))
    
                # Goal
                if i == 3 and j == 11:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=True, color='green', alpha=0.3))
    
                # Arrow
                ax.text(j, i, arrows[action], ha='center', va='center',
                       fontsize=16, fontweight='bold')
    
        ax.set_xlim(-0.5, shape[1]-0.5)
        ax.set_ylim(shape[0]-0.5, -0.5)
        ax.set_xticks(range(shape[1]))
        ax.set_yticks(range(shape[0]))
        ax.grid(True)
        ax.set_title('Learned Policy (Q-Learning)')
    
    
    visualize_policy(cliff_q_agent.Q)
    
    plt.tight_layout()
    plt.savefig('cliff_walking_comparison.png', dpi=150, bbox_inches='tight')
    print("\nCliff Walking comparison saved: cliff_walking_comparison.png")
    plt.close()
    
    env.close()
    

### Path Differences between Q-Learning and SARSA

> **Important Observation** : In Cliff Walking, Q-learning learns the **shortest path (close to cliff)** , while SARSA learns a **safe path (away from cliff)**. This is because SARSA incorporates accidental cliff falls from ε-greedy exploration into its learning.

* * *

## Exercises

**Exercise 1: Comparing Convergence Speed of Q-Learning and SARSA**

Train Q-learning and SARSA on the FrozenLake environment with the same hyperparameters and compare their convergence speeds.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Train Q-learning and SARSA on the FrozenLake environment wit
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import gym
    import numpy as np
    
    # Exercise: Train Q-learning and SARSA with same settings
    # Exercise: Plot episode rewards
    # Exercise: Compare number of episodes needed for convergence
    # Expected: Convergence speed differs depending on environment
    

**Exercise 2: Optimizing ε Decay Schedules**

Implement different ε decay patterns (linear decay, exponential decay, step decay) and compare their performance on Taxi-v3.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implement different ε decay patterns (linear decay, exponent
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Exercise: Implement 3 types of ε decay schedules
    # Exercise: Evaluate each schedule on Taxi-v3
    # Exercise: Compare learning curves and final performance
    # Hint: Emphasize exploration early, exploitation later
    

**Exercise 3: Implementing Double Q-Learning**

Implement Double Q-Learning to prevent overestimation and compare performance with standard Q-learning.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implement Double Q-Learning to prevent overestimation and co
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    
    # Exercise: Implement Double Q-Learning using two Q-tables
    # Exercise: Train on FrozenLake environment
    # Exercise: Compare Q-value estimation error with standard Q-learning
    # Theory: Double algorithm reduces overestimation bias
    

**Exercise 4: Adaptive Learning Rate Adjustment**

Implement an adaptive learning rate that adjusts based on visit count and compare it with fixed learning rate.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implement an adaptive learning rate that adjusts based on vi
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Exercise: Implement adaptive learning rate α(s,a) = 1 / (1 + N(s,a))
    # Exercise: Compare performance with fixed learning rate
    # Exercise: Visualize visit counts for each state
    # Expected: Adaptive learning rate leads to more stable convergence
    

**Exercise 5: Experiments on Custom Environments**

Discretize the state space for other OpenAI Gym environments (CartPole-v1, MountainCar-v0, etc.) and apply Q-learning.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Discretize the state space for other OpenAI Gym environments
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import gym
    import numpy as np
    
    # Exercise: Implement function to discretize continuous state space
    # Exercise: Apply Q-learning on discretized CartPole environment
    # Exercise: Investigate relationship between discretization granularity and performance
    # Challenge: Appropriate discretization of continuous space is critical
    

* * *

## Summary

In this chapter, we learned Q-learning and SARSA based on temporal difference learning.

### Key Points

  * **TD Learning** : Updates at each step without waiting for episode end
  * **Q-Learning** : Off-policy type, directly learns optimal policy
  * **SARSA** : On-policy type, learning that considers exploration impact
  * **ε-greedy** : Simple policy that controls exploration-exploitation balance
  * **Learning rate α** : Controls update strength (0.1-0.3 recommended)
  * **Discount factor γ** : Importance of future rewards (0.95-0.99 recommended)
  * **Q-table** : Stores value for each state-action pair
  * **Application** : Tasks with discrete state and action spaces

### When to Use Q-Learning vs SARSA

Situation | Recommended Algorithm | Reason  
---|---|---  
**Simulation environment** | Q-Learning | Efficiently learns optimal policy  
**Real environment/Robotics** | SARSA | Learns safe policy  
**Dangerous states present** | SARSA | Risk-averse tendency  
**Fast convergence needed** | Q-Learning | Flexible with off-policy  
  
### Next Steps

In the next chapter, we will learn about **Deep Q-Network (DQN)**. We will master techniques for approximating action-value functions using neural networks for large-scale and continuous state spaces that cannot be handled by Q-tables, including Experience Replay, Target Network, and applications to Atari games.
