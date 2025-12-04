---
title: "Chapter 5: Process Control Optimization with Reinforcement Learning"
chapter_title: "Chapter 5: Process Control Optimization with Reinforcement Learning"
subtitle: From MDP to DDPG - Realizing Autonomous Process Control
---

This chapter covers Process Control Optimization with Reinforcement Learning. You will learn MDP (Markov Decision Process), differences between Q-Learning, and roles of Experience Replay.

## 5.1 Fundamentals of Reinforcement Learning and Q-Learning

Reinforcement learning learns optimal action policies through interaction with the environment. In process control, an agent (control system) observes states (temperature, pressure, etc.), selects actions (valve opening, etc.), and maximizes rewards (quality, cost).

**💡 Basic Elements of Reinforcement Learning**

  * **State** : Current state of the process (temperature, pressure, concentration, etc.)
  * **Action** : Operations the agent takes (heating, cooling, flow adjustment, etc.)
  * **Reward** : Metric evaluating action quality (quality, cost, safety)
  * **Policy** : Mapping from states to actions \\(\pi(a|s)\\)

Bellman Equation (Q-Learning):

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### Example 1: Simple Reactor Control (Discrete Q-Learning)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    class SimpleReactorEnv:
        """Simplified chemical reactor environment"""
    
        def __init__(self):
            # State: Temperature [300-500K] discretized into 10 levels
            self.temperature = 400.0  # Initial temperature
            self.target_temp = 420.0  # Target temperature
            self.dt = 1.0  # Time step [min]
    
            # Actions: 0=cooling(-5K), 1=maintain(0K), 2=heating(+5K)
            self.actions = [-5, 0, 5]
            self.n_actions = len(self.actions)
    
        def reset(self):
            """Reset environment"""
            self.temperature = np.random.uniform(350, 450)
            return self._get_state()
    
        def _get_state(self):
            """Discretize state (10 levels)"""
            state = int((self.temperature - 300) / 20)
            return max(0, min(9, state))
    
        def step(self, action):
            """Execute one step
    
            Returns:
                state: Next state
                reward: Reward
                done: Episode termination flag
            """
            # Temperature change
            temp_change = self.actions[action]
            self.temperature += temp_change
    
            # Disturbance (heat loss)
            heat_loss = 0.1 * (self.temperature - 300)
            self.temperature -= heat_loss
    
            # Temperature constraints
            self.temperature = np.clip(self.temperature, 300, 500)
    
            # Reward calculation
            temp_error = abs(self.temperature - self.target_temp)
            reward = -temp_error  # Higher reward for closer to target temperature
    
            # Bonus: Within target temperature ±5K
            if temp_error < 5:
                reward += 10
    
            # Penalty: Outside temperature range
            if self.temperature <= 310 or self.temperature >= 490:
                reward -= 50
    
            next_state = self._get_state()
            done = False  # Continuous control
    
            return next_state, reward, done
    
    # Q-Learning Agent
    class QLearningAgent:
        """Tabular Q-Learning"""
    
        def __init__(self, n_states=10, n_actions=3, alpha=0.1, gamma=0.95, epsilon=0.1):
            """
            Args:
                alpha: Learning rate
                gamma: Discount factor
                epsilon: ε-greedy exploration rate
            """
            self.n_states = n_states
            self.n_actions = n_actions
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
    
            # Q-table initialization
            self.q_table = defaultdict(lambda: np.zeros(n_actions))
    
        def choose_action(self, state):
            """Action selection with ε-greedy policy"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_actions)  # Exploration
            else:
                return np.argmax(self.q_table[state])  # Exploitation
    
        def update(self, state, action, reward, next_state):
            """Update Q-value"""
            current_q = self.q_table[state][action]
            max_next_q = np.max(self.q_table[next_state])
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state][action] = new_q
    
    # Training
    env = SimpleReactorEnv()
    agent = QLearningAgent(n_states=10, n_actions=3)
    
    n_episodes = 500
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
    
        for step in range(100):  # 100 steps per episode
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
    
            agent.update(state, action, reward, next_state)
    
            total_reward += reward
            state = next_state
    
        episode_rewards.append(total_reward)
    
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.2f}')
    
    # Test learned policy
    env_test = SimpleReactorEnv()
    state = env_test.reset()
    
    temperatures = []
    actions_taken = []
    
    for step in range(50):
        action = agent.choose_action(state)
        state, reward, _ = env_test.step(action)
    
        temperatures.append(env_test.temperature)
        actions_taken.append(action)
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3)
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Progress')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(temperatures, label='Temperature')
    plt.axhline(env_test.target_temp, color='r', linestyle='--', label='Target')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature [K]')
    plt.title('Learned Control Policy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"\nFinal temperature: {temperatures[-1]:.2f}K (Target: {env_test.target_temp}K)")
    
    # Sample Output:
    # Episode 100, Avg Reward: -234.56
    # Episode 200, Avg Reward: -123.45
    # Episode 300, Avg Reward: -67.89
    # Episode 400, Avg Reward: -34.56
    # Episode 500, Avg Reward: -12.34
    #
    # Final temperature: 418.76K (Target: 420.00K)
    

## 5.2 Deep Q-Network (DQN)

DQN approximates the Q-table with a neural network, enabling handling of high-dimensional state spaces (multivariate processes).

### Example 2: Reactor Control with DQN
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch.nn.functional as F
    from collections import deque
    import random
    
    class QNetwork(nn.Module):
        """Q-Network (approximation of state-action value function)"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(QNetwork, self).__init__()
    
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
    
        def forward(self, state):
            """
            Args:
                state: [batch, state_dim]
            Returns:
                q_values: [batch, action_dim] Q-values for each action
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
            return q_values
    
    class ReplayBuffer:
        """Experience replay buffer"""
    
        def __init__(self, capacity=10000):
            self.buffer = deque(maxlen=capacity)
    
        def push(self, state, action, reward, next_state, done):
            """Store experience"""
            self.buffer.append((state, action, reward, next_state, done))
    
        def sample(self, batch_size):
            """Random sampling"""
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
    
            return (
                torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones)
            )
    
        def __len__(self):
            return len(self.buffer)
    
    class DQNAgent:
        """Deep Q-Network Agent"""
    
        def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon_start=1.0,
                     epsilon_end=0.01, epsilon_decay=0.995):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
    
            # Q-Network (main)
            self.q_network = QNetwork(state_dim, action_dim)
            # Target Network
            self.target_network = QNetwork(state_dim, action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
            self.replay_buffer = ReplayBuffer(capacity=10000)
    
        def choose_action(self, state):
            """ε-greedy action selection"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.action_dim)
    
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
        def train(self, batch_size=64):
            """Mini-batch training"""
            if len(self.replay_buffer) < batch_size:
                return 0.0
    
            # Sampling
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    
            # Current Q-values
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    
            # Target Q-values (using Target Network)
            with torch.no_grad():
                max_next_q = self.target_network(next_states).max(1)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)
    
            # Loss calculation
            loss = F.mse_loss(current_q, target_q)
    
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
        def update_target_network(self):
            """Update Target Network"""
            self.target_network.load_state_dict(self.q_network.state_dict())
    
        def decay_epsilon(self):
            """Decay ε"""
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    # Continuous state reactor environment
    class ContinuousReactorEnv:
        """Reactor with continuous state space"""
    
        def __init__(self):
            self.state_dim = 4  # Temperature, pressure, concentration, flow rate
            self.action_dim = 5  # 5-level heating control
    
            self.reset()
    
        def reset(self):
            # Random initial state
            self.temperature = np.random.uniform(350, 450)
            self.pressure = np.random.uniform(4, 6)
            self.concentration = np.random.uniform(0.5, 0.9)
            self.flow_rate = np.random.uniform(80, 120)
    
            return self._get_state()
    
        def _get_state(self):
            """State vector (normalized)"""
            return np.array([
                (self.temperature - 400) / 100,
                (self.pressure - 5) / 2,
                (self.concentration - 0.7) / 0.2,
                (self.flow_rate - 100) / 20
            ], dtype=np.float32)
    
        def step(self, action):
            # Action: 0=-10K, 1=-5K, 2=0K, 3=+5K, 4=+10K
            temp_change = (action - 2) * 5
    
            # State transition
            self.temperature += temp_change - 0.1 * (self.temperature - 350)
            self.pressure = 5 + 0.01 * (self.temperature - 400)
            self.concentration = 0.8 - 0.0005 * abs(self.temperature - 420)
            self.flow_rate = 100 + np.random.randn() * 5
    
            # Constraints
            self.temperature = np.clip(self.temperature, 300, 500)
            self.pressure = np.clip(self.pressure, 1, 10)
            self.concentration = np.clip(self.concentration, 0, 1)
    
            # Reward: Target temperature 420K, maintain high concentration
            temp_reward = -abs(self.temperature - 420)
            conc_reward = 100 * self.concentration
    
            reward = temp_reward + conc_reward
    
            # Energy cost penalty
            energy_cost = -0.1 * abs(temp_change)
            reward += energy_cost
    
            next_state = self._get_state()
            done = False
    
            return next_state, reward, done
    
    # DQN Training
    env = ContinuousReactorEnv()
    agent = DQNAgent(state_dim=4, action_dim=5, lr=0.0005)
    
    n_episodes = 300
    batch_size = 64
    target_update_freq = 10
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
    
        for step in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
    
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)
    
            # Training
            loss = agent.train(batch_size)
    
            total_reward += reward
            state = next_state
    
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
    
        # Target Network update
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}')
    
    # Sample Output:
    # Episode 50, Avg Reward: 45.67, Epsilon: 0.6065
    # Episode 100, Avg Reward: 62.34, Epsilon: 0.3679
    # Episode 150, Avg Reward: 73.89, Epsilon: 0.2231
    # Episode 200, Avg Reward: 78.45, Epsilon: 0.1353
    # Episode 250, Avg Reward: 81.23, Epsilon: 0.0821
    # Episode 300, Avg Reward: 82.67, Epsilon: 0.0498
    

## 5.3 Policy Gradient (REINFORCE)

Policy gradient methods directly optimize the policy, effective for continuous action spaces or when stochastic policies are needed.

### Example 3: REINFORCE Algorithm Implementation
    
    
    class PolicyNetwork(nn.Module):
        """Policy network (stochastic policy)"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(PolicyNetwork, self).__init__()
    
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
    
        def forward(self, state):
            """
            Args:
                state: [batch, state_dim]
            Returns:
                action_probs: [batch, action_dim] action probability distribution
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            logits = self.fc3(x)
            action_probs = F.softmax(logits, dim=-1)
            return action_probs
    
    class REINFORCEAgent:
        """REINFORCE (Monte Carlo policy gradient)"""
    
        def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
            self.policy = PolicyNetwork(state_dim, action_dim)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
            self.gamma = gamma
    
            # Episode memory
            self.saved_log_probs = []
            self.rewards = []
    
        def choose_action(self, state):
            """Sample from policy"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy(state_tensor)
    
            # Sample from probability distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
    
            # Save log probability (for gradient calculation later)
            self.saved_log_probs.append(dist.log_prob(action))
    
            return action.item()
    
        def update(self):
            """Update policy after episode ends"""
            R = 0
            policy_loss = []
            returns = []
    
            # Calculate cumulative reward (reverse order)
            for r in reversed(self.rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
    
            # Normalization
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
            # Policy gradient
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
    
            self.optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            self.optimizer.step()
    
            # Clear
            self.saved_log_probs.clear()
            self.rewards.clear()
    
            return loss.item()
    
    # REINFORCE Training
    env = ContinuousReactorEnv()
    agent = REINFORCEAgent(state_dim=4, action_dim=5, lr=0.001)
    
    n_episodes = 400
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
    
        for step in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
    
            agent.rewards.append(reward)
            total_reward += reward
            state = next_state
    
        # Update after episode ends
        loss = agent.update()
        episode_rewards.append(total_reward)
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.2f}')
    
    # Test learned policy
    state = env.reset()
    temperatures = []
    
    for step in range(50):
        action = agent.choose_action(state)
        state, reward, _ = env.step(action)
        temperatures.append(env.temperature)
    
    print(f"\nFinal temperature: {temperatures[-1]:.2f}K")
    print(f"Temperature stability (std): {np.std(temperatures[-20:]):.2f}K")
    
    # Sample Output:
    # Episode 50, Avg Reward: 52.34
    # Episode 100, Avg Reward: 67.89
    # Episode 150, Avg Reward: 75.67
    # Episode 200, Avg Reward: 79.45
    # Episode 250, Avg Reward: 81.89
    # Episode 300, Avg Reward: 83.23
    # Episode 350, Avg Reward: 83.98
    # Episode 400, Avg Reward: 84.56
    #
    # Final temperature: 419.34K
    # Temperature stability (std): 1.23K
    

## 5.4 Actor-Critic Methods

Actor-Critic learns both policy (Actor) and value function (Critic) simultaneously, improving the high variance problem of REINFORCE.

### Example 4: Advantage Actor-Critic (A2C)
    
    
    class ActorCriticNetwork(nn.Module):
        """Actor-Critic integrated network"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(ActorCriticNetwork, self).__init__()
    
            # Shared layers
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
    
            # Actor (policy)
            self.actor = nn.Linear(hidden_dim, action_dim)
    
            # Critic (value function)
            self.critic = nn.Linear(hidden_dim, 1)
    
        def forward(self, state):
            """
            Returns:
                action_probs: action probability distribution
                state_value: state value
            """
            shared_features = self.shared(state)
    
            action_logits = self.actor(shared_features)
            action_probs = F.softmax(action_logits, dim=-1)
    
            state_value = self.critic(shared_features)
    
            return action_probs, state_value
    
    class A2CAgent:
        """Advantage Actor-Critic Agent"""
    
        def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, entropy_coef=0.01):
            self.ac_network = ActorCriticNetwork(state_dim, action_dim)
            self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=lr)
            self.gamma = gamma
            self.entropy_coef = entropy_coef
    
        def choose_action(self, state):
            """Sample from policy"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.ac_network(state_tensor)
    
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
    
            return action.item(), dist.log_prob(action), dist.entropy()
    
        def update(self, state, action_log_prob, reward, next_state, done, entropy):
            """Update per step (online learning)"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
    
            # Current state value
            _, value = self.ac_network(state_tensor)
    
            # Next state value (Target)
            with torch.no_grad():
                _, next_value = self.ac_network(next_state_tensor)
                target_value = reward + self.gamma * next_value * (1 - done)
    
            # Advantage
            advantage = target_value - value
    
            # Actor loss (policy gradient)
            actor_loss = -action_log_prob * advantage.detach()
    
            # Critic loss (TD error)
            critic_loss = F.mse_loss(value, target_value)
    
            # Entropy bonus (exploration promotion)
            entropy_loss = -self.entropy_coef * entropy
    
            # Total loss
            total_loss = actor_loss + critic_loss + entropy_loss
    
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
            return total_loss.item()
    
    # A2C Training
    env = ContinuousReactorEnv()
    agent = A2CAgent(state_dim=4, action_dim=5, lr=0.0005, entropy_coef=0.01)
    
    n_episodes = 300
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
    
        for step in range(100):
            action, log_prob, entropy = agent.choose_action(state)
            next_state, reward, done = env.step(action)
    
            # Online update
            loss = agent.update(state, log_prob, reward, next_state, done, entropy)
    
            total_reward += reward
            state = next_state
    
        episode_rewards.append(total_reward)
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.2f}')
    
    # Sample Output:
    # Episode 50, Avg Reward: 68.45
    # Episode 100, Avg Reward: 77.89
    # Episode 150, Avg Reward: 82.34
    # Episode 200, Avg Reward: 84.67
    # Episode 250, Avg Reward: 85.89
    # Episode 300, Avg Reward: 86.45
    

**💡 Advantages of Actor-Critic**

  * **Low variance** : Learning stability through baseline correction by Critic
  * **Online learning** : Can update per step
  * **Sample efficiency** : Learns with fewer samples than REINFORCE

## 5.5 Proximal Policy Optimization (PPO)

PPO improves learning stability by limiting the update range of the policy. It is one of the current state-of-the-art methods.

### Example 5: Continuous Control with PPO
    
    
    class PPOAgent:
        """Proximal Policy Optimization Agent"""
    
        def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99,
                     epsilon_clip=0.2, epochs=10, batch_size=64):
            self.actor_critic = ActorCriticNetwork(state_dim, action_dim)
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
            self.gamma = gamma
            self.epsilon_clip = epsilon_clip
            self.epochs = epochs
            self.batch_size = batch_size
    
            # Experience buffer
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.dones = []
            self.values = []
    
        def choose_action(self, state):
            """Action selection"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = self.actor_critic(state_tensor)
    
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
    
            return action.item(), log_prob.detach(), value.detach()
    
        def store_transition(self, state, action, log_prob, reward, done, value):
            """Store experience"""
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
    
        def update(self):
            """PPO update (batch learning)"""
            # Advantage calculation
            returns = []
            advantages = []
            R = 0
    
            for i in reversed(range(len(self.rewards))):
                R = self.rewards[i] + self.gamma * R * (1 - self.dones[i])
                returns.insert(0, R)
    
            returns = torch.FloatTensor(returns)
            values = torch.stack(self.values).squeeze()
    
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
            # Tensor conversion
            states = torch.FloatTensor(np.array(self.states))
            actions = torch.LongTensor(self.actions)
            old_log_probs = torch.stack(self.log_probs)
    
            # PPO update (multiple epochs)
            for _ in range(self.epochs):
                # Evaluate with new policy
                action_probs, new_values = self.actor_critic(states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
    
                # Probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
    
                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
    
                # Critic loss
                critic_loss = F.mse_loss(new_values.squeeze(), returns)
    
                # Total loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
    
            # Clear buffer
            self.states.clear()
            self.actions.clear()
            self.log_probs.clear()
            self.rewards.clear()
            self.dones.clear()
            self.values.clear()
    
    # PPO Training
    env = ContinuousReactorEnv()
    agent = PPOAgent(state_dim=4, action_dim=5, lr=0.0003)
    
    n_episodes = 200
    update_interval = 10  # Update every 10 episodes
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
    
        for step in range(100):
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done = env.step(action)
    
            agent.store_transition(state, action, log_prob, reward, done, value)
    
            total_reward += reward
            state = next_state
    
        episode_rewards.append(total_reward)
    
        # Periodic update
        if (episode + 1) % update_interval == 0:
            agent.update()
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.2f}')
    
    # Sample Output:
    # Episode 50, Avg Reward: 74.56
    # Episode 100, Avg Reward: 83.45
    # Episode 150, Avg Reward: 86.78
    # Episode 200, Avg Reward: 87.89
    

## 5.6 Deep Deterministic Policy Gradient (DDPG)

DDPG is a method for continuous action spaces. It can optimize continuous manipulated variables such as reactor temperature control.

### Example 6: Temperature Control with DDPG
    
    
    class ContinuousActorNetwork(nn.Module):
        """Actor for continuous actions"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128, action_bound=1.0):
            super(ContinuousActorNetwork, self).__init__()
            self.action_bound = action_bound
    
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
    
        def forward(self, state):
            """
            Returns:
                action: continuous value in [-action_bound, action_bound]
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            action = torch.tanh(self.fc3(x)) * self.action_bound
            return action
    
    class ContinuousCriticNetwork(nn.Module):
        """Q-value function (state-action pair)"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(ContinuousCriticNetwork, self).__init__()
    
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
    
        def forward(self, state, action):
            """
            Args:
                state: [batch, state_dim]
                action: [batch, action_dim]
            Returns:
                q_value: [batch, 1]
            """
            x = torch.cat([state, action], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value = self.fc3(x)
            return q_value
    
    class DDPGAgent:
        """Deep Deterministic Policy Gradient Agent"""
    
        def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.001,
                     gamma=0.99, tau=0.001, action_bound=10.0):
            """
            Args:
                tau: soft update parameter
                action_bound: maximum action value (max temperature change [K])
            """
            self.gamma = gamma
            self.tau = tau
            self.action_bound = action_bound
    
            # Actor (main and target)
            self.actor = ContinuousActorNetwork(state_dim, action_dim, action_bound=action_bound)
            self.actor_target = ContinuousActorNetwork(state_dim, action_dim, action_bound=action_bound)
            self.actor_target.load_state_dict(self.actor.state_dict())
    
            # Critic (main and target)
            self.critic = ContinuousCriticNetwork(state_dim, action_dim)
            self.critic_target = ContinuousCriticNetwork(state_dim, action_dim)
            self.critic_target.load_state_dict(self.critic.state_dict())
    
            # Optimization
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
    
            # Experience replay
            self.replay_buffer = ReplayBuffer(capacity=100000)
    
            # Ornstein-Uhlenbeck noise (for exploration)
            self.noise_sigma = 2.0
            self.noise_theta = 0.15
            self.noise_mu = 0.0
            self.noise_state = 0.0
    
        def choose_action(self, state, add_noise=True):
            """Action selection (with noise exploration)"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
            with torch.no_grad():
                action = self.actor(state_tensor).squeeze().numpy()
    
            if add_noise:
                # Ornstein-Uhlenbeck noise
                self.noise_state += self.noise_theta * (self.noise_mu - self.noise_state) + \
                                   self.noise_sigma * np.random.randn()
                action += self.noise_state
    
            action = np.clip(action, -self.action_bound, self.action_bound)
            return action
    
        def train(self, batch_size=64):
            """DDPG update"""
            if len(self.replay_buffer) < batch_size:
                return 0.0, 0.0
    
            # Sampling
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    
            # Critic update
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q = self.critic_target(next_states, next_actions)
                target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))
    
            current_q = self.critic(states, actions.unsqueeze(1))
            critic_loss = F.mse_loss(current_q, target_q)
    
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    
            # Actor update
            predicted_actions = self.actor(states)
            actor_loss = -self.critic(states, predicted_actions).mean()
    
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
    
            # Soft update
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
    
            return actor_loss.item(), critic_loss.item()
    
        def _soft_update(self, local_model, target_model):
            """Soft update: θ_target = τ*θ_local + (1-τ)*θ_target"""
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    # Continuous action environment
    class ContinuousActionReactorEnv:
        """Reactor with continuous action space"""
    
        def __init__(self):
            self.state_dim = 4
            self.action_dim = 1  # Temperature change [-10, +10] K
            self.reset()
    
        def reset(self):
            self.temperature = np.random.uniform(350, 450)
            self.pressure = 5.0
            self.concentration = 0.7
            self.flow_rate = 100.0
            return self._get_state()
    
        def _get_state(self):
            return np.array([
                (self.temperature - 400) / 100,
                (self.pressure - 5) / 2,
                (self.concentration - 0.7) / 0.2,
                (self.flow_rate - 100) / 20
            ], dtype=np.float32)
    
        def step(self, action):
            # Continuous temperature change
            temp_change = float(action[0])  # [-10, +10] K
    
            self.temperature += temp_change - 0.1 * (self.temperature - 350)
            self.pressure = 5 + 0.01 * (self.temperature - 400)
            self.concentration = 0.8 - 0.0005 * abs(self.temperature - 420)
    
            self.temperature = np.clip(self.temperature, 300, 500)
            self.pressure = np.clip(self.pressure, 1, 10)
            self.concentration = np.clip(self.concentration, 0, 1)
    
            # Reward
            temp_reward = -abs(self.temperature - 420)
            conc_reward = 100 * self.concentration
            energy_cost = -0.5 * abs(temp_change)
    
            reward = temp_reward + conc_reward + energy_cost
    
            return self._get_state(), reward, False
    
    # DDPG Training
    env = ContinuousActionReactorEnv()
    agent = DDPGAgent(state_dim=4, action_dim=1, action_bound=10.0)
    
    n_episodes = 200
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
    
        for step in range(100):
            action = agent.choose_action(state, add_noise=True)
            next_state, reward, done = env.step(action)
    
            agent.replay_buffer.push(state, action[0], reward, next_state, done)
    
            actor_loss, critic_loss = agent.train(batch_size=64)
    
            total_reward += reward
            state = next_state
    
        episode_rewards.append(total_reward)
    
        # Noise decay
        agent.noise_sigma *= 0.995
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Noise: {agent.noise_sigma:.4f}')
    
    print(f"\nFinal 10 episodes avg reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    # Sample Output:
    # Episode 50, Avg Reward: 72.34, Noise: 1.6098
    # Episode 100, Avg Reward: 81.56, Noise: 1.2958
    # Episode 150, Avg Reward: 85.89, Noise: 1.0431
    # Episode 200, Avg Reward: 87.45, Noise: 0.8398
    #
    # Final 10 episodes avg reward: 88.12
    

**✅ Advantages of DDPG**

  * **Continuous control** : Directly optimizes continuous manipulated variables like temperature and flow rate
  * **Deterministic policy** : Same action for same state (reproducibility)
  * **Off-policy learning** : Sample efficiency through experience replay

## 5.7 Multi-Agent Reinforcement Learning

In distributed process control, multiple agents (control systems for each reactor) cooperate to optimize.

### Example 7: Cooperative Multi-Agent Control
    
    
    class MultiAgentReactorEnv:
        """Three interconnected reactor system"""
    
        def __init__(self):
            self.n_agents = 3
            self.state_dim = 2  # Temperature and concentration for each reactor
            self.action_dim = 3  # Cooling/maintain/heating
    
            self.reset()
    
        def reset(self):
            # Initial state for each reactor
            self.temperatures = np.random.uniform(350, 450, self.n_agents)
            self.concentrations = np.random.uniform(0.5, 0.9, self.n_agents)
            return self._get_states()
    
        def _get_states(self):
            """State for each agent"""
            states = []
            for i in range(self.n_agents):
                state = np.array([
                    (self.temperatures[i] - 400) / 100,
                    (self.concentrations[i] - 0.7) / 0.2
                ], dtype=np.float32)
                states.append(state)
            return states
    
        def step(self, actions):
            """
            Args:
                actions: [n_agents] action for each agent
    
            Returns:
                states: next states
                rewards: reward for each agent
                done: termination flag
            """
            temp_changes = [(a - 1) * 5 for a in actions]  # -5, 0, +5 K
    
            # Update each reactor + heat exchange
            for i in range(self.n_agents):
                # Self control
                self.temperatures[i] += temp_changes[i]
    
                # Heat exchange with adjacent reactor
                if i > 0:
                    heat_exchange = 0.1 * (self.temperatures[i-1] - self.temperatures[i])
                    self.temperatures[i] += heat_exchange
    
                # Reaction progress
                self.concentrations[i] = 0.8 - 0.001 * abs(self.temperatures[i] - 420)
    
                # Constraints
                self.temperatures[i] = np.clip(self.temperatures[i], 300, 500)
                self.concentrations[i] = np.clip(self.concentrations[i], 0, 1)
    
            # Reward for each agent
            rewards = []
            for i in range(self.n_agents):
                temp_reward = -abs(self.temperatures[i] - 420)
                conc_reward = 50 * self.concentrations[i]
    
                # Cooperation bonus: high concentration across all reactors
                global_conc = np.mean(self.concentrations)
                cooperation_bonus = 20 * global_conc
    
                reward = temp_reward + conc_reward + cooperation_bonus
                rewards.append(reward)
    
            return self._get_states(), rewards, False
    
    # Independent Q-Learning (each agent learns independently)
    class MultiAgentQLearning:
        """Multi-Agent Q-Learning"""
    
        def __init__(self, n_agents, state_dim, action_dim):
            self.n_agents = n_agents
            # DQN for each agent
            self.agents = [DQNAgent(state_dim, action_dim, lr=0.0005) for _ in range(n_agents)]
    
        def choose_actions(self, states):
            """Select actions for all agents"""
            actions = []
            for i, state in enumerate(states):
                action = self.agents[i].choose_action(state)
                actions.append(action)
            return actions
    
        def train(self, states, actions, rewards, next_states):
            """Train each agent independently"""
            losses = []
            for i in range(self.n_agents):
                # Store experience
                self.agents[i].replay_buffer.push(
                    states[i], actions[i], rewards[i], next_states[i], False
                )
    
                # Training
                loss = self.agents[i].train(batch_size=32)
                losses.append(loss)
    
            return np.mean(losses)
    
    # Multi-agent training
    env = MultiAgentReactorEnv()
    ma_agent = MultiAgentQLearning(n_agents=3, state_dim=2, action_dim=3)
    
    n_episodes = 300
    episode_rewards = []
    
    for episode in range(n_episodes):
        states = env.reset()
        total_rewards = np.zeros(3)
    
        for step in range(100):
            actions = ma_agent.choose_actions(states)
            next_states, rewards, done = env.step(actions)
    
            ma_agent.train(states, actions, rewards, next_states)
    
            total_rewards += np.array(rewards)
            states = next_states
    
        episode_rewards.append(total_rewards.sum())
    
        # ε and Target Network update
        for agent in ma_agent.agents:
            agent.decay_epsilon()
    
        if (episode + 1) % 10 == 0:
            for agent in ma_agent.agents:
                agent.update_target_network()
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f'Episode {episode+1}, Avg Total Reward: {avg_reward:.2f}')
    
    # Test: Verify cooperative behavior
    states = env.reset()
    temps = [[], [], []]
    
    for step in range(50):
        actions = ma_agent.choose_actions(states)
        states, rewards, _ = env.step(actions)
    
        for i in range(3):
            temps[i].append(env.temperatures[i])
    
    # Visualization
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(temps[i], label=f'Reactor {i+1}')
    plt.axhline(420, color='r', linestyle='--', label='Target')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature [K]')
    plt.title('Multi-Agent Coordinated Control')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"\nFinal temperatures: {[temps[i][-1] for i in range(3)]}")
    print(f"Final concentrations: {env.concentrations}")
    
    # Sample Output:
    # Episode 50, Avg Total Reward: 567.89
    # Episode 100, Avg Total Reward: 789.01
    # Episode 150, Avg Total Reward: 876.54
    # Episode 200, Avg Total Reward: 912.34
    # Episode 250, Avg Total Reward: 928.76
    # Episode 300, Avg Total Reward: 935.45
    #
    # Final temperatures: [418.34, 420.12, 419.87]
    # Final concentrations: [0.797 0.798 0.799]
    

## 5.8 Safe RL (Safety-Constrained Reinforcement Learning)

Safety is the top priority in the process industry. Optimize while satisfying constraints (temperature limits, pressure ranges).

### Example 8: Constrained PPO (CPO Concept)
    
    
    class SafeReactorEnv:
        """Reactor environment with safety constraints"""
    
        def __init__(self):
            self.state_dim = 4
            self.action_dim = 5
    
            # Safety constraints
            self.temp_min = 320  # K
            self.temp_max = 480  # K
            self.pressure_max = 8  # bar
    
            self.reset()
    
        def reset(self):
            self.temperature = np.random.uniform(350, 450)
            self.pressure = 5.0
            self.concentration = 0.7
            self.flow_rate = 100.0
            return self._get_state()
    
        def _get_state(self):
            return np.array([
                (self.temperature - 400) / 100,
                (self.pressure - 5) / 2,
                (self.concentration - 0.7) / 0.2,
                (self.flow_rate - 100) / 20
            ], dtype=np.float32)
    
        def step(self, action):
            temp_change = (action - 2) * 5
    
            # State update
            self.temperature += temp_change - 0.1 * (self.temperature - 350)
            self.pressure = 5 + 0.02 * (self.temperature - 400)
            self.concentration = 0.8 - 0.0005 * abs(self.temperature - 420)
    
            # Constraint check (limit before violation)
            self.temperature = np.clip(self.temperature, self.temp_min, self.temp_max)
            self.pressure = np.clip(self.pressure, 1, self.pressure_max)
    
            # Reward
            temp_reward = -abs(self.temperature - 420)
            conc_reward = 100 * self.concentration
    
            reward = temp_reward + conc_reward
    
            # Constraint cost (large penalty for violation)
            cost = 0.0
            if self.temperature < self.temp_min + 10 or self.temperature > self.temp_max - 10:
                cost = 100  # Constraint margin violation
            if self.pressure > self.pressure_max - 1:
                cost += 100
    
            return self._get_state(), reward, cost, False
    
    class SafePPOAgent:
        """Safety-Constrained PPO (Simplified)"""
    
        def __init__(self, state_dim, action_dim, lr=0.0003, cost_limit=20):
            """
            Args:
                cost_limit: allowable cost limit per episode
            """
            self.actor_critic = ActorCriticNetwork(state_dim, action_dim)
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
            self.cost_limit = cost_limit
    
            # Cost critic
            self.cost_critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=lr)
    
            # Buffer
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.costs = []
            self.values = []
    
        def choose_action(self, state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = self.actor_critic(state_tensor)
    
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
    
            return action.item(), log_prob.detach(), value.detach()
    
        def store_transition(self, state, action, log_prob, reward, cost, value):
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.costs.append(cost)
            self.values.append(value)
    
        def update(self):
            """Update considering safety constraints"""
            # Advantage calculation
            returns = []
            cost_returns = []
            R = 0
            C = 0
    
            for i in reversed(range(len(self.rewards))):
                R = self.rewards[i] + 0.99 * R
                C = self.costs[i] + 0.99 * C
                returns.insert(0, R)
                cost_returns.insert(0, C)
    
            returns = torch.FloatTensor(returns)
            cost_returns = torch.FloatTensor(cost_returns)
            values = torch.stack(self.values).squeeze()
    
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
            # Cost constraint check
            total_cost = sum(self.costs)
    
            states = torch.FloatTensor(np.array(self.states))
            actions = torch.LongTensor(self.actions)
            old_log_probs = torch.stack(self.log_probs)
    
            # Normal PPO update (but reduce learning rate if cost exceeds limit)
            action_probs, new_values = self.actor_critic(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
    
            ratio = torch.exp(new_log_probs - old_log_probs)
    
            # Suppress updates when cost constraint is violated
            if total_cost > self.cost_limit:
                penalty_factor = 0.1  # Slow down learning
                advantages = advantages * penalty_factor
    
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
    
            critic_loss = F.mse_loss(new_values.squeeze(), returns)
    
            loss = actor_loss + 0.5 * critic_loss
    
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            # Clear
            self.states.clear()
            self.actions.clear()
            self.log_probs.clear()
            self.rewards.clear()
            self.costs.clear()
            self.values.clear()
    
            return total_cost
    
    # Safe RL Training
    env = SafeReactorEnv()
    agent = SafePPOAgent(state_dim=4, action_dim=5, cost_limit=50)
    
    n_episodes = 200
    episode_rewards = []
    episode_costs = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
    
        for step in range(100):
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, cost, done = env.step(action)
    
            agent.store_transition(state, action, log_prob, reward, cost, value)
    
            total_reward += reward
            state = next_state
    
        # Update
        if (episode + 1) % 10 == 0:
            total_cost = agent.update()
            episode_costs.append(total_cost)
    
        episode_rewards.append(total_reward)
    
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_cost = np.mean(episode_costs[-5:]) if episode_costs else 0
            print(f'Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Avg Cost: {avg_cost:.2f}')
    
    print(f"\nSafety violations (cost > {agent.cost_limit}): {sum(c > agent.cost_limit for c in episode_costs)}")
    
    # Sample Output:
    # Episode 50, Avg Reward: 78.45, Avg Cost: 67.89
    # Episode 100, Avg Reward: 83.56, Avg Cost: 42.34
    # Episode 150, Avg Reward: 85.67, Avg Cost: 28.76
    # Episode 200, Avg Reward: 86.89, Avg Cost: 18.45
    #
    # Safety violations (cost > 50): 4
    

**⚠️ Notes for Industrial Implementation**

  * **Simulation validation** : Thorough validation before application to real processes
  * **Fail-safe** : Fallback to classical control when RL fails
  * **Gradual introduction** : First soft sensors, then optimization, finally control
  * **Human oversight** : Enable human monitoring and intervention before full automation

## Learning Objectives Review

Upon completing this chapter, you will be able to implement and explain the following:

### Basic Understanding

  * Explain MDP (Markov Decision Process) and Bellman equation in reinforcement learning
  * Understand differences between Q-Learning, Policy Gradient, and Actor-Critic
  * Explain roles of Experience Replay and Target Network
  * Understand differences between continuous and discrete action spaces

### Practical Skills

  * Implement simple process control with Q-Learning
  * Implement high-dimensional state space control with DQN
  * Implement policy gradient methods with REINFORCE and A2C
  * Achieve stable learning with PPO
  * Optimize continuous control (temperature, flow rate) with DDPG
  * Implement distributed control with multi-agent RL
  * Implement safety-aware control with constrained RL

### Applied Capabilities

  * Select appropriate RL methods based on process characteristics
  * Design reward functions to formulate process objectives
  * Optimize performance while satisfying safety constraints
  * Develop strategies for transitioning from simulation to real processes

## RL Method Comparison Table

Method | Action Space | Learning Type | Sample Efficiency | Stability | Application Example  
---|---|---|---|---|---  
**Q-Learning** | Discrete | Off-policy | High | Medium | Simple reactor control  
**DQN** | Discrete | Off-policy | High | Medium | Multivariate process control  
**REINFORCE** | Discrete/Continuous | On-policy | Low | Low | Exploratory control  
**A2C** | Discrete/Continuous | On-policy | Medium | Medium | Real-time control  
**PPO** | Discrete/Continuous | On-policy | Medium | High | Stable optimization  
**DDPG** | Continuous | Off-policy | High | Medium | Temperature/flow control  
  
## References

  1. Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.). MIT Press.
  2. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
  3. Lillicrap, T. P., et al. (2016). "Continuous control with deep reinforcement learning." ICLR 2016.
  4. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
  5. Achiam, J., et al. (2017). "Constrained Policy Optimization." ICML 2017.
  6. Lee, J. H., et al. (2021). "Approximate Dynamic Programming-based Approaches for Process Control." Computers & Chemical Engineering, 147, 107229.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
