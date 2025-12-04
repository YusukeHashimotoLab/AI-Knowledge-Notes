---
title: "Chapter 1: Fundamentals of Reinforcement Learning"
chapter_title: "Chapter 1: Fundamentals of Reinforcement Learning"
subtitle: Understanding the basic concepts of reinforcement learning, Markov decision processes, value functions and Bellman equations, and fundamental algorithms
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
code_examples: 8
exercises: 6
---

This chapter covers the fundamentals of Fundamentals of Reinforcement Learning, which what is reinforcement learning?. You will learn differences between reinforcement learning, difference between value functions (V), and concept of policy.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the differences between reinforcement learning, supervised learning, and unsupervised learning
  * ✅ Explain the basic concepts of Markov decision processes (MDP): states, actions, rewards, and transitions
  * ✅ Understand the meaning and role of the Bellman equation
  * ✅ Explain the difference between value functions (V) and action-value functions (Q)
  * ✅ Understand the concept of policy and the definition of optimal policy
  * ✅ Understand the exploration-exploitation tradeoff
  * ✅ Implement value iteration
  * ✅ Implement policy iteration
  * ✅ Understand the basic principles of Monte Carlo methods
  * ✅ Understand and implement TD learning (Temporal Difference)

* * *

## 1.1 What is Reinforcement Learning?

### Basic Concepts of Reinforcement Learning

Reinforcement Learning (RL) is a branch of machine learning in which an agent learns optimal actions through trial and error while interacting with an environment. It is applied in a wide range of fields, including game AI, robot control, autonomous driving, and recommendation systems.

> "Reinforcement learning is the problem of learning what actions to take in order to maximize a reward signal."

#### Components of Reinforcement Learning

A reinforcement learning system consists of six key components: the **Agent** (the entity that learns and makes decisions), the **Environment** (the object with which the agent interacts), **State** (information representing the current situation of the environment), **Action** (operations the agent can choose to perform), **Reward** (immediate feedback indicating the quality of an action), and **Policy** (a mapping from states to actions that serves as the decision-making rule).
    
    
    ```mermaid
    graph LR
        A[Agent] -->|Action At| B[Environment]
        B -->|State St+1| A
        B -->|Reward Rt+1| A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
    ```

### Reinforcement Learning vs Supervised Learning vs Unsupervised Learning

Learning Method | Data Characteristics | Learning Objective | Examples  
---|---|---|---  
**Supervised Learning** | Pairs of inputs and correct labels | Build a model to predict correct answers | Image classification, speech recognition  
**Unsupervised Learning** | Data without labels | Discover structure and patterns in data | Clustering, dimensionality reduction  
**Reinforcement Learning** | Feedback of actions and rewards | Learn a policy that maximizes cumulative rewards | Game AI, robot control  
  
#### Characteristics of Reinforcement Learning

  1. **Learning through trial and error** : No correct answers are provided; learning occurs from reward signals
  2. **Delayed rewards** : The consequences of actions are often not immediately apparent
  3. **Exploration-exploitation tradeoff** : Whether to try new actions or take known good actions
  4. **Sequential decision making** : Past actions influence future states

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Characteristics of Reinforcement Learning
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== Comparison of Reinforcement Learning vs Supervised Learning ===\n")
    
    # Example of supervised learning: simple regression
    print("【Supervised Learning】")
    print("Objective: Learn a function from data")
    X_train = np.array([1, 2, 3, 4, 5])
    y_train = np.array([2, 4, 6, 8, 10])  # y = 2x relationship
    
    # Learn using least squares method
    slope = np.sum((X_train - X_train.mean()) * (y_train - y_train.mean())) / \
            np.sum((X_train - X_train.mean())**2)
    intercept = y_train.mean() - slope * X_train.mean()
    
    print(f"Learning result: y = {slope:.2f}x + {intercept:.2f}")
    print("Characteristic: Correct answers (y_train) are explicitly provided\n")
    
    # Example of reinforcement learning: simple bandit problem
    print("【Reinforcement Learning】")
    print("Objective: Learn actions that maximize rewards")
    
    class SimpleBandit:
        """3-armed bandit problem"""
        def __init__(self):
            # True expected reward of each arm (unknown to agent)
            self.true_values = np.array([0.3, 0.5, 0.7])
    
        def pull(self, action):
            """Pull an arm and get a reward"""
            # Generate reward from Bernoulli distribution
            reward = 1 if np.random.rand() < self.true_values[action] else 0
            return reward
    
    # Learn with ε-greedy algorithm
    bandit = SimpleBandit()
    n_arms = 3
    n_steps = 1000
    epsilon = 0.1
    
    Q = np.zeros(n_arms)  # Estimated value of each arm
    N = np.zeros(n_arms)  # Number of times each arm was pulled
    rewards = []
    
    for step in range(n_steps):
        # Action selection with ε-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)  # Exploration
        else:
            action = np.argmax(Q)  # Exploitation
    
        # Get reward
        reward = bandit.pull(action)
        rewards.append(reward)
    
        # Update Q-value
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
    
    print(f"True expected rewards: {bandit.true_values}")
    print(f"Learned estimates: {Q}")
    print(f"Average reward: {np.mean(rewards):.3f}")
    print("Characteristic: Correct answer unknown, learning through trial and error from reward signals\n")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Supervised learning
    axes[0].scatter(X_train, y_train, s=100, alpha=0.6, label='Training data (with correct answers)')
    X_test = np.linspace(0, 6, 100)
    y_pred = slope * X_test + intercept
    axes[0].plot(X_test, y_pred, 'r-', linewidth=2, label=f'Learned model')
    axes[0].set_xlabel('Input X', fontsize=12)
    axes[0].set_ylabel('Output y', fontsize=12)
    axes[0].set_title('Supervised Learning: Learning from correct data', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Right: Reinforcement learning
    window = 50
    cumulative_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[1].plot(cumulative_rewards, linewidth=2)
    axes[1].axhline(y=max(bandit.true_values), color='r', linestyle='--',
                    linewidth=2, label=f'Optimal reward ({max(bandit.true_values):.1f})')
    axes[1].set_xlabel('Steps', fontsize=12)
    axes[1].set_ylabel('Average reward (moving average)', fontsize=12)
    axes[1].set_title('Reinforcement Learning: Learning optimal actions through trial and error', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_vs_supervised.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'rl_vs_supervised.png'.")
    

* * *

## 1.2 Markov Decision Process (MDP)

### Definition of MDP

The Markov Decision Process (MDP) is a framework that provides the mathematical foundation for reinforcement learning. An MDP is defined by a 5-tuple $(S, A, P, R, \gamma)$:

  * $S$: State space
  * $A$: Action space
  * $P$: State transition probability $P(s'|s, a)$
  * $R$: Reward function $R(s, a, s')$
  * $\gamma$: Discount factor $\in [0, 1]$

> "Markov property: The next state depends only on the current state and action, not on past history."

#### Mathematical Expression of the Markov Property

When state $s$ satisfies the Markov property:

$$ P(S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0) = P(S_{t+1}|S_t, A_t) $$ 

### States, Actions, Rewards, and Transitions

#### 1\. State

A state represents information about the current situation of the environment:

  * **Full observability** : The agent can observe all information about the environment (e.g., chess)
  * **Partial observability** : Only some information is observable (e.g., poker)

#### 2\. Action

  * **Discrete action space** : A finite number of actions (e.g., moving up, down, left, right)
  * **Continuous action space** : Real-valued actions (e.g., robot joint angles)

#### 3\. Reward

The reward is a scalar value $r_t$ indicating the quality of the action at time $t$. The agent's goal is to maximize the expected value of the cumulative reward (return):

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$ 

#### 4\. Discount Factor $\gamma$

  * $\gamma = 0$: Only considers immediate rewards (myopic)
  * $\gamma = 1$: Weights all future rewards equally
  * $0 < \gamma < 1$: Discounts future rewards (typically 0.9-0.99)

    
    
    ```mermaid
    graph TD
        S0[State S0] -->|Action a0| S1[State S1]
        S1 -->|Reward r1| S0
        S1 -->|Action a1| S2[State S2]
        S2 -->|Reward r2| S1
        S2 -->|Action a2| S3[State S3]
        S3 -->|Reward r3| S2
    
        style S0 fill:#e3f2fd
        style S1 fill:#fff3e0
        style S2 fill:#e8f5e9
        style S3 fill:#fce4ec
    ```
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== Basics of Markov Decision Process (MDP) ===\n")
    
    class SimpleMDP:
        """
        Simple MDP example: 3x3 grid world
    
        State: (x, y) coordinates
        Action: Movement in four directions (0:up, 1:right, 2:down, 3:left)
        Reward: +1 for reaching goal, 0 otherwise
        """
        def __init__(self):
            self.grid_size = 3
            self.start_state = (0, 0)
            self.goal_state = (2, 2)
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up right down left
            self.action_names = ['Up', 'Right', 'Down', 'Left']
    
        def is_valid_state(self, state):
            """Check if state is valid"""
            x, y = state
            return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
        def step(self, state, action):
            """
            State transition function
    
            Returns:
            --------
            next_state : tuple
            reward : float
            done : bool
            """
            if state == self.goal_state:
                return state, 0, True
    
            # Calculate next state
            dx, dy = self.actions[action]
            next_state = (state[0] + dx, state[1] + dy)
    
            # If hitting a wall, stay in current state
            if not self.is_valid_state(next_state):
                next_state = state
    
            # Calculate reward
            reward = 1.0 if next_state == self.goal_state else 0.0
            done = (next_state == self.goal_state)
    
            return next_state, reward, done
    
        def get_all_states(self):
            """Get all states"""
            return [(x, y) for x in range(self.grid_size)
                    for y in range(self.grid_size)]
    
    # Instantiate MDP
    mdp = SimpleMDP()
    
    print("【Components of MDP】")
    print(f"State space S: {mdp.get_all_states()}")
    print(f"Action space A: {mdp.action_names}")
    print(f"Start state: {mdp.start_state}")
    print(f"Goal state: {mdp.goal_state}\n")
    
    # Demonstration of Markov property
    print("【Verification of Markov Property】")
    current_state = (1, 1)
    action = 1  # Right
    
    print(f"Current state: {current_state}")
    print(f"Selected action: {mdp.action_names[action]}")
    
    # Execute state transition
    next_state, reward, done = mdp.step(current_state, action)
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print("\n→ Next state is determined only by current state and action (Markov property)\n")
    
    # Visualize the effect of discount factor
    print("【Effect of Discount Factor γ】")
    gammas = [0.0, 0.5, 0.9, 0.99]
    rewards = np.array([1, 1, 1, 1, 1])  # 5 consecutive steps with reward 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Cumulative reward calculation
    axes[0].set_title('Difference in cumulative rewards by discount factor', fontsize=14, fontweight='bold')
    for gamma in gammas:
        discounted_rewards = [gamma**i * r for i, r in enumerate(rewards)]
        cumulative = np.cumsum(discounted_rewards)
        axes[0].plot(cumulative, marker='o', label=f'γ={gamma}', linewidth=2)
    
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Cumulative reward', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Right: Weight for each step
    axes[1].set_title('Weight for rewards at each step', fontsize=14, fontweight='bold')
    steps = np.arange(10)
    for gamma in gammas:
        weights = [gamma**i for i in steps]
        axes[1].plot(steps, weights, marker='o', label=f'γ={gamma}', linewidth=2)
    
    axes[1].set_xlabel('Future steps', fontsize=12)
    axes[1].set_ylabel('Weight (γ^k)', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mdp_basics.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'mdp_basics.png'.")
    

* * *

## 1.3 Bellman Equation and Value Functions

### Value Function

A value function evaluates how good a state or state-action pair is.

#### State-Value Function $V^\pi(s)$

The expected cumulative reward from state $s$ when following policy $\pi$:

$$ V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right] $$ 

#### Action-Value Function $Q^\pi(s, a)$

The expected cumulative reward when taking action $a$ in state $s$, then following policy $\pi$:

$$ Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right] $$ 

#### Relationship between V and Q

$$ V^\pi(s) = \sum_{a} \pi(a|s) Q^\pi(s, a) $$ 

### Bellman Equation

The Bellman equation recursively defines value functions. This is a fundamental equation for dynamic programming.

#### Bellman Expectation Equation

State-value function:

$$ V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^\pi(s')\right] $$ 

Action-value function:

$$ Q^\pi(s, a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')\right] $$ 

#### Bellman Optimality Equation

Optimal state-value function $V^*(s)$:

$$ V^*(s) = \max_{a} \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^*(s')\right] $$ 

Optimal action-value function $Q^*(s, a)$:

$$ Q^*(s, a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\right] $$ 

> "Intuition of Bellman equation: Current value = Immediate reward + Discounted future value"

### Policy

Policy $\pi$ is a mapping from states to actions:

  * **Deterministic policy** : $a = \pi(s)$ (determines one action for each state)
  * **Stochastic policy** : $\pi(a|s)$ (probability distribution over actions for each state)

#### Optimal Policy $\pi^*$

A policy that achieves the maximum value in all states:

$$ \pi^*(s) = \arg\max_{a} Q^*(s, a) $$ 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== Bellman Equation and Value Functions ===\n")
    
    class GridWorld:
        """
        4x4 Grid World MDP
        """
        def __init__(self, grid_size=4):
            self.size = grid_size
            self.n_states = grid_size * grid_size
            self.n_actions = 4  # up right down left
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            self.action_names = ['↑', '→', '↓', '←']
    
            # Terminal states (top-left and bottom-right)
            self.terminal_states = [(0, 0), (grid_size-1, grid_size-1)]
    
        def state_to_index(self, state):
            """Convert (x, y) to 1D index"""
            return state[0] * self.size + state[1]
    
        def index_to_state(self, index):
            """Convert 1D index to (x, y)"""
            return (index // self.size, index % self.size)
    
        def is_terminal(self, state):
            """Check if terminal state"""
            return state in self.terminal_states
    
        def get_next_state(self, state, action):
            """Get next state"""
            if self.is_terminal(state):
                return state
    
            dx, dy = self.actions[action]
            next_state = (state[0] + dx, state[1] + dy)
    
            # Check if within grid bounds
            if (0 <= next_state[0] < self.size and
                0 <= next_state[1] < self.size):
                return next_state
            else:
                return state  # Don't move if hitting wall
    
        def get_reward(self, state, action, next_state):
            """Get reward"""
            if self.is_terminal(state):
                return 0
            return -1  # -1 reward per step (motivation to find shortest path)
    
    def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
        """
        Policy evaluation: Calculate value function of given policy
    
        Parameters:
        -----------
        env : GridWorld
        policy : ndarray
            Action probability distribution for each state [n_states, n_actions]
        gamma : float
            Discount factor
        theta : float
            Convergence threshold
    
        Returns:
        --------
        V : ndarray
            State-value function
        """
        V = np.zeros(env.n_states)
    
        iteration = 0
        while True:
            delta = 0
            V_old = V.copy()
    
            for s_idx in range(env.n_states):
                state = env.index_to_state(s_idx)
    
                if env.is_terminal(state):
                    continue
    
                v = 0
                # Bellman expectation equation
                for action in range(env.n_actions):
                    next_state = env.get_next_state(state, action)
                    reward = env.get_reward(state, action, next_state)
                    next_s_idx = env.state_to_index(next_state)
    
                    # V(s) = Σ π(a|s) [R + γV(s')]
                    v += policy[s_idx, action] * (reward + gamma * V_old[next_s_idx])
    
                V[s_idx] = v
                delta = max(delta, abs(V[s_idx] - V_old[s_idx]))
    
            iteration += 1
            if delta < theta:
                break
    
        print(f"Policy evaluation converged in {iteration} iterations")
        return V
    
    # Create grid world
    env = GridWorld(grid_size=4)
    
    # Random policy (select each action with equal probability)
    random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    
    print("【Evaluation of Random Policy】")
    V_random = policy_evaluation(env, random_policy, gamma=0.9)
    
    # Display value function in 2D grid
    V_grid = V_random.reshape((env.size, env.size))
    print("\nState-value function V(s):")
    print(V_grid)
    print()
    
    # Calculate optimal policy (greedy policy)
    def compute_greedy_policy(env, V, gamma=0.9):
        """
        Compute greedy policy from value function
        """
        policy = np.zeros((env.n_states, env.n_actions))
    
        for s_idx in range(env.n_states):
            state = env.index_to_state(s_idx)
    
            if env.is_terminal(state):
                policy[s_idx] = 1.0 / env.n_actions  # Uniform for terminal state
                continue
    
            # Calculate Q-value for each action
            q_values = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(state, action, next_state)
                next_s_idx = env.state_to_index(next_state)
                q_values[action] = reward + gamma * V[next_s_idx]
    
            # Select action with maximum Q-value
            best_action = np.argmax(q_values)
            policy[s_idx, best_action] = 1.0
    
        return policy
    
    greedy_policy = compute_greedy_policy(env, V_random, gamma=0.9)
    
    # Visualize policy
    def visualize_policy(env, policy):
        """Visualize policy with arrows"""
        policy_grid = np.zeros((env.size, env.size), dtype=object)
    
        for s_idx in range(env.n_states):
            state = env.index_to_state(s_idx)
            if env.is_terminal(state):
                policy_grid[state] = 'T'
            else:
                action = np.argmax(policy[s_idx])
                policy_grid[state] = env.action_names[action]
    
        return policy_grid
    
    print("【Greedy Policy (derived from random policy)】")
    policy_grid = visualize_policy(env, greedy_policy)
    print(policy_grid)
    print()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: State-value function
    im1 = axes[0].imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title('State-Value Function V(s)\n(Random Policy)',
                      fontsize=14, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            text = axes[0].text(j, i, f'{V_grid[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=11)
    axes[0].set_xticks(range(env.size))
    axes[0].set_yticks(range(env.size))
    plt.colorbar(im1, ax=axes[0])
    
    # Right: Greedy policy
    policy_display = np.zeros((env.size, env.size))
    axes[1].imshow(policy_display, cmap='Blues', alpha=0.3)
    axes[1].set_title('Greedy Policy\n(derived from V(s))',
                      fontsize=14, fontweight='bold')
    
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if env.is_terminal(state):
                axes[1].text(j, i, 'GOAL', ha="center", va="center",
                            fontsize=12, fontweight='bold', color='red')
            else:
                s_idx = env.state_to_index(state)
                action = np.argmax(greedy_policy[s_idx])
                arrow = env.action_names[action]
                axes[1].text(j, i, arrow, ha="center", va="center",
                            fontsize=20, fontweight='bold')
    
    axes[1].set_xticks(range(env.size))
    axes[1].set_yticks(range(env.size))
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('value_function_bellman.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'value_function_bellman.png'.")
    

* * *

## 1.4 Value Iteration and Policy Iteration

### Value Iteration

Value iteration is an algorithm that iteratively applies the Bellman optimality equation to find the optimal value function.

#### Algorithm

  1. Initialize $V(s)$ to an arbitrary value (usually 0)
  2. For each state $s$, iterate: $$ V_{k+1}(s) = \max_{a} \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V_k(s')\right] $$ 
  3. Repeat until $V_k$ converges
  4. Extract optimal policy: $\pi^*(s) = \arg\max_{a} Q^*(s, a)$

### Policy Iteration

Policy iteration alternates between policy evaluation and policy improvement.

#### Algorithm

  1. **Initialization** : Choose an arbitrary policy $\pi$
  2. **Policy evaluation** : Calculate $V^\pi$
  3. **Policy improvement** : $\pi' = \text{greedy}(V^\pi)$
  4. If $\pi' = \pi$, stop; otherwise set $\pi = \pi'$ and go to step 2

Method | Characteristic | Convergence Speed | Application  
---|---|---|---  
**Value Iteration** | Directly optimizes value function | Slow | When simple implementation is needed  
**Policy Iteration** | Iteratively improves policy | Fast | When policy is important  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== Implementation of Value Iteration and Policy Iteration ===\n")
    
    class GridWorldEnv:
        """Grid World environment"""
        def __init__(self, size=4):
            self.size = size
            self.n_states = size * size
            self.n_actions = 4
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up right down left
            self.terminal_states = [(0, 0), (size-1, size-1)]
    
        def state_to_index(self, state):
            return state[0] * self.size + state[1]
    
        def index_to_state(self, index):
            return (index // self.size, index % self.size)
    
        def is_terminal(self, state):
            return state in self.terminal_states
    
        def step(self, state, action):
            if self.is_terminal(state):
                return state, 0
    
            dx, dy = self.actions[action]
            next_state = (state[0] + dx, state[1] + dy)
    
            if (0 <= next_state[0] < self.size and
                0 <= next_state[1] < self.size):
                return next_state, -1
            else:
                return state, -1  # Reward -1 even when hitting wall
    
    def value_iteration(env, gamma=0.9, theta=1e-6):
        """
        Value iteration
    
        Returns:
        --------
        V : ndarray
            Optimal state-value function
        policy : ndarray
            Optimal policy
        iterations : int
            Number of iterations until convergence
        """
        V = np.zeros(env.n_states)
        policy = np.zeros(env.n_states, dtype=int)
    
        iteration = 0
        while True:
            delta = 0
            V_old = V.copy()
    
            for s_idx in range(env.n_states):
                state = env.index_to_state(s_idx)
    
                if env.is_terminal(state):
                    continue
    
                # Calculate Q-value for each action
                q_values = np.zeros(env.n_actions)
                for action in range(env.n_actions):
                    next_state, reward = env.step(state, action)
                    next_s_idx = env.state_to_index(next_state)
                    q_values[action] = reward + gamma * V_old[next_s_idx]
    
                # Bellman optimality equation: V(s) = max_a Q(s, a)
                V[s_idx] = np.max(q_values)
                policy[s_idx] = np.argmax(q_values)
    
                delta = max(delta, abs(V[s_idx] - V_old[s_idx]))
    
            iteration += 1
            if delta < theta:
                break
    
        return V, policy, iteration
    
    def policy_iteration(env, gamma=0.9, theta=1e-6):
        """
        Policy iteration
    
        Returns:
        --------
        V : ndarray
            Optimal state-value function
        policy : ndarray
            Optimal policy
        iterations : int
            Number of policy improvements
        """
        # Initialize with random policy
        policy = np.random.randint(0, env.n_actions, size=env.n_states)
    
        iteration = 0
        while True:
            # 1. Policy evaluation
            V = np.zeros(env.n_states)
            while True:
                delta = 0
                V_old = V.copy()
    
                for s_idx in range(env.n_states):
                    state = env.index_to_state(s_idx)
    
                    if env.is_terminal(state):
                        continue
    
                    action = policy[s_idx]
                    next_state, reward = env.step(state, action)
                    next_s_idx = env.state_to_index(next_state)
    
                    V[s_idx] = reward + gamma * V_old[next_s_idx]
                    delta = max(delta, abs(V[s_idx] - V_old[s_idx]))
    
                if delta < theta:
                    break
    
            # 2. Policy improvement
            policy_stable = True
            for s_idx in range(env.n_states):
                state = env.index_to_state(s_idx)
    
                if env.is_terminal(state):
                    continue
    
                old_action = policy[s_idx]
    
                # Calculate greedy policy
                q_values = np.zeros(env.n_actions)
                for action in range(env.n_actions):
                    next_state, reward = env.step(state, action)
                    next_s_idx = env.state_to_index(next_state)
                    q_values[action] = reward + gamma * V[next_s_idx]
    
                policy[s_idx] = np.argmax(q_values)
    
                if old_action != policy[s_idx]:
                    policy_stable = False
    
            iteration += 1
            if policy_stable:
                break
    
        return V, policy, iteration
    
    # Create environment
    env = GridWorldEnv(size=4)
    
    # Execute value iteration
    print("【Value Iteration】")
    V_vi, policy_vi, iter_vi = value_iteration(env, gamma=0.9)
    print(f"Iterations until convergence: {iter_vi}")
    print(f"\nOptimal state-value function:")
    print(V_vi.reshape(env.size, env.size))
    print()
    
    # Execute policy iteration
    print("【Policy Iteration】")
    V_pi, policy_pi, iter_pi = policy_iteration(env, gamma=0.9)
    print(f"Number of policy improvements: {iter_pi}")
    print(f"\nOptimal state-value function:")
    print(V_pi.reshape(env.size, env.size))
    print()
    
    # Visualize policy
    action_symbols = ['↑', '→', '↓', '←']
    
    def visualize_policy_grid(env, policy):
        policy_grid = np.zeros((env.size, env.size), dtype=object)
        for s_idx in range(env.n_states):
            state = env.index_to_state(s_idx)
            if env.is_terminal(state):
                policy_grid[state] = 'G'
            else:
                policy_grid[state] = action_symbols[policy[s_idx]]
        return policy_grid
    
    print("【Optimal Policy (Value Iteration)】")
    policy_grid_vi = visualize_policy_grid(env, policy_vi)
    print(policy_grid_vi)
    print()
    
    print("【Optimal Policy (Policy Iteration)】")
    policy_grid_pi = visualize_policy_grid(env, policy_pi)
    print(policy_grid_pi)
    print()
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Value iteration results
    V_grid_vi = V_vi.reshape(env.size, env.size)
    im1 = axes[0, 0].imshow(V_grid_vi, cmap='RdYlGn', interpolation='nearest')
    axes[0, 0].set_title('Value Iteration: Optimal Value Function', fontsize=12, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            axes[0, 0].text(j, i, f'{V_grid_vi[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Value iteration policy
    axes[0, 1].imshow(np.zeros((env.size, env.size)), cmap='Blues', alpha=0.3)
    axes[0, 1].set_title(f'Value Iteration: Optimal Policy\n(iterations: {iter_vi})',
                         fontsize=12, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            if env.is_terminal((i, j)):
                axes[0, 1].text(j, i, 'GOAL', ha="center", va="center",
                               fontsize=10, fontweight='bold', color='red')
            else:
                s_idx = env.state_to_index((i, j))
                axes[0, 1].text(j, i, action_symbols[policy_vi[s_idx]],
                               ha="center", va="center", fontsize=16)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Policy iteration results
    V_grid_pi = V_pi.reshape(env.size, env.size)
    im2 = axes[1, 0].imshow(V_grid_pi, cmap='RdYlGn', interpolation='nearest')
    axes[1, 0].set_title('Policy Iteration: Optimal Value Function', fontsize=12, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            axes[1, 0].text(j, i, f'{V_grid_pi[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Policy iteration policy
    axes[1, 1].imshow(np.zeros((env.size, env.size)), cmap='Blues', alpha=0.3)
    axes[1, 1].set_title(f'Policy Iteration: Optimal Policy\n(policy improvements: {iter_pi})',
                         fontsize=12, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            if env.is_terminal((i, j)):
                axes[1, 1].text(j, i, 'GOAL', ha="center", va="center",
                               fontsize=10, fontweight='bold', color='red')
            else:
                s_idx = env.state_to_index((i, j))
                axes[1, 1].text(j, i, action_symbols[policy_pi[s_idx]],
                               ha="center", va="center", fontsize=16)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('value_policy_iteration.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'value_policy_iteration.png'.")
    

* * *

## 1.5 Exploration-Exploitation Tradeoff

### Exploration vs Exploitation

One of the most important challenges in reinforcement learning is the exploration-exploitation tradeoff.

> "Exploration: Try new actions to find better options"  
>  "Exploitation: Select the best known action to maximize rewards"

#### Multi-Armed Bandit Problem

There is a slot machine (bandit) with $K$ arms, and each arm $i$ has a different expected reward $\mu_i$. The goal is to maximize cumulative rewards.

### Exploration Strategies

Strategy | Description | Characteristic  
---|---|---  
**ε-greedy** | Random action with probability $\epsilon$ | Simple, easy to tune  
**Softmax** | Probabilistic selection based on Q-values | Smooth exploration  
**UCB** | Uses upper confidence bound | Theoretical guarantees  
**Thompson Sampling** | Based on Bayesian inference | Efficient exploration  
  
#### ε-greedy Policy

$$ a_t = \begin{cases} \arg\max_a Q(a) & \text{probability } 1-\epsilon \\\ \text{random action} & \text{probability } \epsilon \end{cases} $$ 

#### Softmax (Boltzmann) Policy

$$ P(a) = \frac{\exp(Q(a) / \tau)}{\sum_{a'} \exp(Q(a') / \tau)} $$ where $\tau$ is the temperature parameter (higher means more exploratory). 

#### UCB (Upper Confidence Bound)

$$ a_t = \arg\max_a \left[Q(a) + c\sqrt{\frac{\ln t}{N(a)}}\right] $$ where $N(a)$ is the number of times action $a$ was selected, and $c$ controls exploration degree. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== Exploration-Exploitation Tradeoff ===\n")
    
    class MultiArmedBandit:
        """Multi-armed bandit problem"""
        def __init__(self, n_arms=10, seed=42):
            np.random.seed(seed)
            self.n_arms = n_arms
            # True expected reward of each arm (generated from standard normal distribution)
            self.true_values = np.random.randn(n_arms)
            self.optimal_arm = np.argmax(self.true_values)
    
        def pull(self, arm):
            """Pull an arm and get a reward (expected value + noise)"""
            reward = self.true_values[arm] + np.random.randn()
            return reward
    
    class EpsilonGreedy:
        """ε-greedy algorithm"""
        def __init__(self, n_arms, epsilon=0.1):
            self.n_arms = n_arms
            self.epsilon = epsilon
            self.Q = np.zeros(n_arms)  # Estimated values
            self.N = np.zeros(n_arms)  # Selection counts
    
        def select_action(self):
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_arms)  # Exploration
            else:
                return np.argmax(self.Q)  # Exploitation
    
        def update(self, action, reward):
            self.N[action] += 1
            # Incremental update: Q(a) ← Q(a) + α[R - Q(a)]
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
    
    class UCB:
        """UCB (Upper Confidence Bound) algorithm"""
        def __init__(self, n_arms, c=2.0):
            self.n_arms = n_arms
            self.c = c
            self.Q = np.zeros(n_arms)
            self.N = np.zeros(n_arms)
            self.t = 0
    
        def select_action(self):
            self.t += 1
    
            # Select each arm at least once
            if np.min(self.N) == 0:
                return np.argmin(self.N)
    
            # Calculate UCB scores
            ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
            return np.argmax(ucb_values)
    
        def update(self, action, reward):
            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
    
    class Softmax:
        """Softmax (Boltzmann) policy"""
        def __init__(self, n_arms, tau=1.0):
            self.n_arms = n_arms
            self.tau = tau  # Temperature parameter
            self.Q = np.zeros(n_arms)
            self.N = np.zeros(n_arms)
    
        def select_action(self):
            # Calculate Softmax probabilities
            exp_values = np.exp(self.Q / self.tau)
            probs = exp_values / np.sum(exp_values)
            return np.random.choice(self.n_arms, p=probs)
    
        def update(self, action, reward):
            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
    
    def run_experiment(bandit, agent, n_steps=1000):
        """Run experiment"""
        rewards = np.zeros(n_steps)
        optimal_actions = np.zeros(n_steps)
    
        for step in range(n_steps):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
    
            rewards[step] = reward
            optimal_actions[step] = (action == bandit.optimal_arm)
    
        return rewards, optimal_actions
    
    # Create bandit
    bandit = MultiArmedBandit(n_arms=10, seed=42)
    
    print("【True Expected Rewards】")
    for i, value in enumerate(bandit.true_values):
        marker = " ← Optimal" if i == bandit.optimal_arm else ""
        print(f"Arm {i}: {value:.3f}{marker}")
    print()
    
    # Compare multiple strategies
    n_runs = 100
    n_steps = 1000
    
    strategies = {
        'ε-greedy (ε=0.01)': lambda: EpsilonGreedy(10, epsilon=0.01),
        'ε-greedy (ε=0.1)': lambda: EpsilonGreedy(10, epsilon=0.1),
        'UCB (c=2)': lambda: UCB(10, c=2.0),
        'Softmax (τ=1)': lambda: Softmax(10, tau=1.0),
    }
    
    results = {}
    
    for name, agent_fn in strategies.items():
        print(f"Running: {name}")
        all_rewards = np.zeros((n_runs, n_steps))
        all_optimal = np.zeros((n_runs, n_steps))
    
        for run in range(n_runs):
            bandit_run = MultiArmedBandit(n_arms=10, seed=run)
            agent = agent_fn()
            rewards, optimal = run_experiment(bandit_run, agent, n_steps)
            all_rewards[run] = rewards
            all_optimal[run] = optimal
    
        results[name] = {
            'rewards': all_rewards.mean(axis=0),
            'optimal': all_optimal.mean(axis=0)
        }
    
    print("\nExperiment complete\n")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Average reward
    axes[0].set_title('Average Reward Over Time', fontsize=14, fontweight='bold')
    for name, data in results.items():
        axes[0].plot(data['rewards'], label=name, linewidth=2, alpha=0.8)
    
    axes[0].axhline(y=max(bandit.true_values), color='r', linestyle='--',
                    linewidth=2, label='Optimal reward', alpha=0.5)
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Average reward', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Right: Optimal action selection rate
    axes[1].set_title('Optimal Action Selection Rate', fontsize=14, fontweight='bold')
    for name, data in results.items():
        axes[1].plot(data['optimal'], label=name, linewidth=2, alpha=0.8)
    
    axes[1].set_xlabel('Steps', fontsize=12)
    axes[1].set_ylabel('Probability of selecting optimal action', fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exploration_exploitation.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'exploration_exploitation.png'.")
    

* * *

## 1.6 Monte Carlo Methods and TD Learning

### Monte Carlo Methods

Monte Carlo methods update value functions after experiencing entire episodes. They are model-free and can learn without knowing the environment's dynamics.

#### First-Visit MC

Update using the return $G_t$ when first visiting state $s$:

$$ V(s) \leftarrow V(s) + \alpha [G_t - V(s)] $$ 

#### Every-Visit MC

Use returns from all times when visiting state $s$.

### TD Learning (Temporal Difference Learning)

TD learning updates value functions at each step without waiting for episode completion.

#### TD(0) Algorithm

$$ V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$ Here $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called the **TD error**. 

### MC vs TD Comparison

Feature | Monte Carlo Method | TD Learning  
---|---|---  
**Update timing** | After episode completion | Each step  
**Required information** | Actual return $G_t$ | Estimate of next state $V(S_{t+1})$  
**Bias** | Unbiased | Biased (uses estimates)  
**Variance** | High variance | Low variance  
**Convergence speed** | Slow | Fast  
**Application** | Episodic tasks only | Continuing tasks possible  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== Monte Carlo Methods and TD Learning ===\n")
    
    class RandomWalk:
        """
        1D random walk environment
    
        States: [0, 1, 2, 3, 4, 5, 6]
        - State 0: Left end (terminal, reward 0)
        - States 1-5: Normal states
        - State 6: Right end (terminal, reward +1)
        """
        def __init__(self):
            self.n_states = 7
            self.start_state = 3  # Start from center
    
        def reset(self):
            return self.start_state
    
        def step(self, state):
            """Move randomly left or right"""
            if state == 0 or state == 6:
                return state, 0, True  # Terminal state
    
            # Move left or right with 50% probability
            if np.random.rand() < 0.5:
                next_state = state - 1
            else:
                next_state = state + 1
    
            # Reward and termination check
            if next_state == 6:
                return next_state, 1.0, True
            elif next_state == 0:
                return next_state, 0.0, True
            else:
                return next_state, 0.0, False
    
    def monte_carlo_evaluation(env, n_episodes=1000, alpha=0.1):
        """
        Value function estimation using Monte Carlo method
        """
        V = np.zeros(env.n_states)
        V[6] = 1.0  # True value of right end
    
        for episode in range(n_episodes):
            # Generate episode
            states = []
            rewards = []
    
            state = env.reset()
            states.append(state)
    
            while True:
                next_state, reward, done = env.step(state)
                rewards.append(reward)
    
                if done:
                    break
    
                states.append(next_state)
                state = next_state
    
            # Calculate returns (backward from end)
            G = 0
            visited = set()
    
            for t in range(len(states) - 1, -1, -1):
                G = rewards[t] + G  # Assuming γ=1
                s = states[t]
    
                # First-Visit MC
                if s not in visited:
                    visited.add(s)
                    # Incremental update
                    V[s] = V[s] + alpha * (G - V[s])
    
        return V
    
    def td_learning(env, n_episodes=1000, alpha=0.1, gamma=1.0):
        """
        Value function estimation using TD(0)
        """
        V = np.zeros(env.n_states)
        V[6] = 1.0  # True value of right end
    
        for episode in range(n_episodes):
            state = env.reset()
    
            while True:
                next_state, reward, done = env.step(state)
    
                # TD(0) update
                td_target = reward + gamma * V[next_state]
                td_error = td_target - V[state]
                V[state] = V[state] + alpha * td_error
    
                if done:
                    break
    
                state = next_state
    
        return V
    
    # True value function (can be calculated analytically)
    true_values = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])
    
    # Random walk environment
    env = RandomWalk()
    
    print("【True Value Function】")
    print("States:", list(range(7)))
    print("Values:", [f"{v:.3f}" for v in true_values])
    print()
    
    # Execute Monte Carlo method
    print("Running Monte Carlo method...")
    V_mc = monte_carlo_evaluation(env, n_episodes=5000, alpha=0.01)
    
    print("【Estimation by Monte Carlo Method】")
    print("States:", list(range(7)))
    print("Values:", [f"{v:.3f}" for v in V_mc])
    print()
    
    # Execute TD learning
    print("Running TD learning...")
    V_td = td_learning(env, n_episodes=5000, alpha=0.01)
    
    print("【Estimation by TD Learning】")
    print("States:", list(range(7)))
    print("Values:", [f"{v:.3f}" for v in V_td])
    print()
    
    # Compare learning curves
    def evaluate_learning_curve(env, method, n_runs=20, episode_checkpoints=None):
        """Evaluate learning curves"""
        if episode_checkpoints is None:
            episode_checkpoints = [0, 1, 10, 100, 500, 1000, 2000, 5000]
    
        errors = {ep: [] for ep in episode_checkpoints}
    
        for run in range(n_runs):
            for n_ep in episode_checkpoints:
                if n_ep == 0:
                    V = np.zeros(7)
                    V[6] = 1.0
                else:
                    if method == 'MC':
                        V = monte_carlo_evaluation(env, n_episodes=n_ep, alpha=0.01)
                    else:  # TD
                        V = td_learning(env, n_episodes=n_ep, alpha=0.01)
    
                # Calculate RMSE
                rmse = np.sqrt(np.mean((V - true_values)**2))
                errors[n_ep].append(rmse)
    
        # Calculate averages
        avg_errors = {ep: np.mean(errors[ep]) for ep in episode_checkpoints}
        return avg_errors
    
    print("Evaluating learning curves (this may take some time)...")
    episode_checkpoints = [0, 1, 10, 100, 500, 1000, 2000, 5000]
    mc_errors = evaluate_learning_curve(env, 'MC', n_runs=10,
                                        episode_checkpoints=episode_checkpoints)
    td_errors = evaluate_learning_curve(env, 'TD', n_runs=10,
                                        episode_checkpoints=episode_checkpoints)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Value function comparison
    states = np.arange(7)
    axes[0].plot(states, true_values, 'k-', linewidth=3, marker='o',
                 markersize=8, label='True value', alpha=0.7)
    axes[0].plot(states, V_mc, 'b--', linewidth=2, marker='s',
                 markersize=6, label='MC (5000 episodes)', alpha=0.8)
    axes[0].plot(states, V_td, 'r-.', linewidth=2, marker='^',
                 markersize=6, label='TD (5000 episodes)', alpha=0.8)
    
    axes[0].set_xlabel('State', fontsize=12)
    axes[0].set_ylabel('Estimated value', fontsize=12)
    axes[0].set_title('Random Walk: Value Function Estimation', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(states)
    
    # Right: Learning curves
    episodes = list(mc_errors.keys())
    mc_rmse = [mc_errors[ep] for ep in episodes]
    td_rmse = [td_errors[ep] for ep in episodes]
    
    axes[1].plot(episodes, mc_rmse, 'b-', linewidth=2, marker='s',
                 markersize=6, label='Monte Carlo method', alpha=0.8)
    axes[1].plot(episodes, td_rmse, 'r-', linewidth=2, marker='^',
                 markersize=6, label='TD learning', alpha=0.8)
    
    axes[1].set_xlabel('Number of episodes', fontsize=12)
    axes[1].set_ylabel('RMSE (root mean squared error)', fontsize=12)
    axes[1].set_title('Comparison of Learning Speed', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mc_vs_td.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'mc_vs_td.png'.")
    

* * *

## 1.7 Practice: Reinforcement Learning in Grid World

### Implementation of Grid World Environment

Here, we build a more complex Grid World environment and apply the learned algorithms in an integrated manner.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    print("=== Practical Reinforcement Learning in Grid World ===\n")
    
    class GridWorldEnv:
        """
        Customizable Grid World environment
    
        - Can place walls, goals, and holes (pitfalls)
        - Agent moves in four directions
        - Stochastic transitions (may not move in intended direction)
        """
        def __init__(self, size=5, slip_prob=0.1):
            self.size = size
            self.slip_prob = slip_prob  # Slip probability
    
            # Grid configuration
            self.grid = np.zeros((size, size), dtype=int)
            # 0: normal, 1: wall, 2: goal, 3: hole
    
            # Default environment setup
            self.grid[1, 1] = 1  # Wall
            self.grid[1, 2] = 1  # Wall
            self.grid[2, 3] = 3  # Hole
            self.grid[4, 4] = 2  # Goal
    
            self.start_pos = (0, 0)
            self.current_pos = self.start_pos
    
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up right down left
            self.action_names = ['↑', '→', '↓', '←']
            self.n_actions = 4
    
        def reset(self):
            """Reset environment"""
            self.current_pos = self.start_pos
            return self.current_pos
    
        def is_valid_pos(self, pos):
            """Check if position is valid"""
            x, y = pos
            if not (0 <= x < self.size and 0 <= y < self.size):
                return False
            if self.grid[x, y] == 1:  # Wall
                return False
            return True
    
        def step(self, action):
            """
            Execute action
    
            Returns:
            --------
            next_pos : tuple
            reward : float
            done : bool
            """
            # Handle slip
            if np.random.rand() < self.slip_prob:
                action = np.random.randint(self.n_actions)
    
            dx, dy = self.actions[action]
            next_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
    
            # Stay in place for invalid moves
            if not self.is_valid_pos(next_pos):
                next_pos = self.current_pos
    
            # Reward and termination check
            cell_type = self.grid[next_pos]
            if cell_type == 2:  # Goal
                reward = 10.0
                done = True
            elif cell_type == 3:  # Hole
                reward = -10.0
                done = True
            else:
                reward = -0.1  # Small penalty per step
                done = False
    
            self.current_pos = next_pos
            return next_pos, reward, done
    
        def render(self, policy=None, values=None):
            """Visualize environment"""
            fig, ax = plt.subplots(figsize=(8, 8))
    
            # Draw grid
            for i in range(self.size):
                for j in range(self.size):
                    cell_type = self.grid[i, j]
    
                    if cell_type == 1:  # Wall
                        color = 'gray'
                        ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                              facecolor=color))
                    elif cell_type == 2:  # Goal
                        color = 'gold'
                        ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                              facecolor=color))
                        ax.text(j+0.5, self.size-1-i+0.5, 'GOAL',
                               ha='center', va='center', fontsize=10,
                               fontweight='bold', color='red')
                    elif cell_type == 3:  # Hole
                        color = 'black'
                        ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                              facecolor=color))
                        ax.text(j+0.5, self.size-1-i+0.5, 'HOLE',
                               ha='center', va='center', fontsize=10,
                               fontweight='bold', color='white')
                    else:  # Normal
                        # Display value function
                        if values is not None:
                            value = values[i, j]
                            norm_value = (value - values.min()) / \
                                        (values.max() - values.min() + 1e-8)
                            color = plt.cm.RdYlGn(norm_value)
                            ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                                  facecolor=color, alpha=0.6))
                            ax.text(j+0.5, self.size-1-i+0.7, f'{value:.1f}',
                                   ha='center', va='center', fontsize=8)
    
                        # Display policy
                        if policy is not None and values is not None:
                            arrow = policy[i, j]
                            ax.text(j+0.5, self.size-1-i+0.3, arrow,
                                   ha='center', va='center', fontsize=14,
                                   fontweight='bold')
    
            # Mark start position
            ax.plot(self.start_pos[1]+0.5, self.size-1-self.start_pos[0]+0.5,
                   'go', markersize=15, label='Start')
    
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_aspect('equal')
            ax.set_xticks(range(self.size+1))
            ax.set_yticks(range(self.size+1))
            ax.grid(True)
            ax.legend()
    
            return fig
    
    class QLearningAgent:
        """Q-learning agent"""
        def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
            self.env = env
            self.alpha = alpha  # Learning rate
            self.gamma = gamma  # Discount factor
            self.epsilon = epsilon  # Exploration rate
    
            # Q table
            self.Q = np.zeros((env.size, env.size, env.n_actions))
    
        def select_action(self, state):
            """Select action with ε-greedy policy"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.env.n_actions)
            else:
                x, y = state
                return np.argmax(self.Q[x, y])
    
        def update(self, state, action, reward, next_state, done):
            """Update Q-value"""
            x, y = state
            nx, ny = next_state
    
            if done:
                td_target = reward
            else:
                td_target = reward + self.gamma * np.max(self.Q[nx, ny])
    
            td_error = td_target - self.Q[x, y, action]
            self.Q[x, y, action] += self.alpha * td_error
    
        def get_policy(self):
            """Get current policy"""
            policy = np.zeros((self.env.size, self.env.size), dtype=object)
            values = np.zeros((self.env.size, self.env.size))
    
            for i in range(self.env.size):
                for j in range(self.env.size):
                    if self.env.grid[i, j] == 1:  # Wall
                        policy[i, j] = '■'
                        values[i, j] = 0
                    elif self.env.grid[i, j] == 2:  # Goal
                        policy[i, j] = 'G'
                        values[i, j] = 10
                    elif self.env.grid[i, j] == 3:  # Hole
                        policy[i, j] = 'H'
                        values[i, j] = -10
                    else:
                        best_action = np.argmax(self.Q[i, j])
                        policy[i, j] = self.env.action_names[best_action]
                        values[i, j] = np.max(self.Q[i, j])
    
            return policy, values
    
        def train(self, n_episodes=1000):
            """Execute learning"""
            episode_rewards = []
            episode_lengths = []
    
            for episode in range(n_episodes):
                state = self.env.reset()
                total_reward = 0
                steps = 0
    
                while steps < 100:  # Maximum number of steps
                    action = self.select_action(state)
                    next_state, reward, done = self.env.step(action)
                    self.update(state, action, reward, next_state, done)
    
                    total_reward += reward
                    steps += 1
                    state = next_state
    
                    if done:
                        break
    
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
    
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_length = np.mean(episode_lengths[-100:])
                    print(f"Episode {episode+1}: "
                          f"Average reward={avg_reward:.2f}, Average steps={avg_length:.1f}")
    
            return episode_rewards, episode_lengths
    
    # Create Grid World environment
    env = GridWorldEnv(size=5, slip_prob=0.1)
    
    print("【Grid World Environment】")
    print("Grid size: 5x5")
    print("Slip probability: 0.1")
    print("Goal: (4, 4) → Reward +10")
    print("Hole: (2, 3) → Reward -10")
    print("Each step: Reward -0.1\n")
    
    # Create and train Q-learning agent
    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    
    print("【Training with Q-learning】")
    rewards, lengths = agent.train(n_episodes=500)
    
    print("\nTraining complete\n")
    
    # Get learned policy
    policy, values = agent.get_policy()
    
    print("【Learned Policy】")
    print(policy)
    print()
    
    # Visualization
    fig1 = env.render(policy=policy, values=values)
    plt.title('Policy and Value Function Learned by Q-learning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gridworld_qlearning.png', dpi=150, bbox_inches='tight')
    print("Saved policy visualization to 'gridworld_qlearning.png'.")
    
    # Learning curves
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Reward progression
    window = 20
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed_rewards, linewidth=2)
    axes[0].set_xlabel('Episodes', fontsize=12)
    axes[0].set_ylabel('Average reward (moving average)', fontsize=12)
    axes[0].set_title('Reward Progression', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Right: Step count progression
    smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_lengths, linewidth=2, color='orange')
    axes[1].set_xlabel('Episodes', fontsize=12)
    axes[1].set_ylabel('Average steps (moving average)', fontsize=12)
    axes[1].set_title('Episode Length Progression', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gridworld_learning_curves.png', dpi=150, bbox_inches='tight')
    print("Saved learning curves to 'gridworld_learning_curves.png'.")
    

* * *

## Summary

In this chapter, we learned the fundamentals of reinforcement learning:

**1\. Definition of reinforcement learning** — A framework for learning optimal actions through trial and error.

**2\. MDP** — Formulation using states, actions, rewards, transition probabilities, and discount factor.

**3\. Bellman equation** — Recursive definition of value functions and the foundation of dynamic programming.

**4\. Value iteration and policy iteration** — Dynamic programming algorithms for finding optimal policies.

**5\. Exploration and exploitation** — Strategies like epsilon-greedy, UCB, and Softmax for balancing exploration and exploitation.

**6\. Monte Carlo methods and TD learning** — Model-free learning methods that do not require knowledge of environment dynamics.

**7\. Practice** — Implementation of Q-learning in Grid World environment.

In the next chapter, we will learn about more advanced reinforcement learning algorithms (SARSA, Q-learning, DQN).

Exercises

#### Problem 1: Understanding MDP

Explain how the agent's behavior changes when the discount factor $\gamma$ is close to 0 versus close to 1.

#### Problem 2: Bellman Equation

Derive the relationship between the state-value function $V^\pi(s)$ and the action-value function $Q^\pi(s, a)$.

#### Problem 3: Exploration Strategies

In the ε-greedy policy, how does the agent behave in the extreme cases of $\epsilon = 0$ and $\epsilon = 1$?

#### Problem 4: MC and TD

Explain the differences between Monte Carlo methods and TD learning from the perspectives of bias and variance.

#### Problem 5: Value Iteration

Modify the provided code to execute value iteration on a 3x3 grid world.

#### Problem 6: ε-greedy Implementation

For the multi-armed bandit problem, implement an ε-decay strategy that decreases $\epsilon$ over time (e.g., $\epsilon_t = \epsilon_0 / (1 + t)$).
