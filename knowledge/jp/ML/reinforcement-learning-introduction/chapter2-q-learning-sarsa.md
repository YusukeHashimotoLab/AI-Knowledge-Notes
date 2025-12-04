---
title: 第2章：Q学習とSARSA
chapter_title: 第2章：Q学習とSARSA
subtitle: 時間差分学習による価値関数の推定と行動方策の最適化
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 時間差分（TD）学習の基本原理とモンテカルロ法との違いを理解する
  * ✅ Q学習アルゴリズム（オフポリシー型）の仕組みと更新式を説明できる
  * ✅ SARSAアルゴリズム（オンポリシー型）の特徴と適用場面を理解する
  * ✅ ε-greedy方策による探索と活用のバランス調整ができる
  * ✅ 学習率と割引率のハイパーパラメータの影響を説明できる
  * ✅ OpenAI GymのTaxi-v3やCliff Walking環境で実装できる

* * *

## 2.1 時間差分（TD）学習の基礎

### モンテカルロ法の課題

第1章で学んだモンテカルロ法は、**エピソード終了まで待つ必要がある** という制約がありました：

$$ V(s_t) \leftarrow V(s_t) + \alpha [G_t - V(s_t)] $$ 

ここで $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$ は実際の収益（リターン）です。

### 時間差分学習の基本アイデア

**時間差分（Temporal Difference: TD）学習** は、エピソード終了を待たずに、**1ステップごとに価値関数を更新** します：

$$ V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)] $$ 

ここで：

  * $r_{t+1} + \gamma V(s_{t+1})$：**TD目標** （TD target）
  * $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$：**TD誤差** （TD error）
  * $\alpha$：学習率（learning rate）

    
    
    ```mermaid
    graph LR
        S1["状態 s_t"] --> A["行動 a_t"]
        A --> S2["状態 s_t+1"]
        S2 --> R["報酬 r_t+1"]
        R --> Update["V(s_t) 更新"]
        S2 --> Update
    
        style S1 fill:#b3e5fc
        style S2 fill:#c5e1a5
        style R fill:#fff9c4
        style Update fill:#ffab91
    ```

### TD(0)の実装
    
    
    import numpy as np
    import gym
    
    def td_0_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.99):
        """
        TD(0)による状態価値関数の推定
    
        Args:
            env: 環境
            policy: 方策 (状態 -> 行動確率分布)
            num_episodes: エピソード数
            alpha: 学習率
            gamma: 割引率
    
        Returns:
            V: 状態価値関数
        """
        # 状態価値関数の初期化
        V = np.zeros(env.observation_space.n)
    
        for episode in range(num_episodes):
            state, _ = env.reset()
    
            while True:
                # 方策に従って行動選択
                action = np.random.choice(env.action_space.n, p=policy[state])
    
                # 環境との相互作用
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    
                # TD(0)更新
                td_target = reward + gamma * V[next_state]
                td_error = td_target - V[state]
                V[state] = V[state] + alpha * td_error
    
                if done:
                    break
    
                state = next_state
    
        return V
    
    
    # 使用例：FrozenLake環境
    print("=== TD(0)による価値関数推定 ===")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # ランダム方策
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    
    # TD(0)実行
    V = td_0_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.99)
    
    print(f"状態価値関数:\n{V.reshape(4, 4)}")
    env.close()
    

### モンテカルロ法とTD学習の比較

項目 | モンテカルロ法 | TD学習  
---|---|---  
**更新タイミング** | エピソード終了後 | 各ステップ後  
**収益の計算** | 実際の収益 $G_t$ | 推定収益 $r + \gamma V(s')$  
**バイアス** | なし（不偏推定） | あり（初期値に依存）  
**分散** | 高い | 低い  
**継続タスク** | 適用不可 | 適用可能  
**収束速度** | 遅い | 速い  
  
> 「TD学習は、ブートストラップ（自己の推定値を使って更新）により、効率的な学習を実現します」

* * *

## 2.2 Q学習（Q-Learning）

### 行動価値関数Q(s, a)

状態価値関数 $V(s)$ の代わりに、**行動価値関数** $Q(s, a)$ を学習します：

$$ Q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a] $$ 

これは「状態 $s$ で行動 $a$ を取った後の期待収益」を表します。

### Q学習の更新式

**Q学習** は、TD学習を行動価値関数に適用したものです：

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right] $$ 

重要なポイント：

  * $\max_{a'} Q(s_{t+1}, a')$：次状態での**最良の行動** の価値を使う
  * **オフポリシー型** ：実際の行動と更新に使う行動が異なる
  * 最適方策を直接学習できる

    
    
    ```mermaid
    graph TB
        Start["状態 s, 行動 a"] --> Execute["環境実行"]
        Execute --> Observe["s', r 観測"]
        Observe --> MaxQ["max_a' Q(s', a')"]
        MaxQ --> Target["TD目標 = r + γ max Q(s', a')"]
        Target --> Update["Q(s,a) 更新"]
        Update --> Next["次ステップ"]
    
        style Start fill:#b3e5fc
        style MaxQ fill:#fff59d
        style Update fill:#ffab91
    ```

### Q学習アルゴリズムの実装
    
    
    import numpy as np
    import gym
    
    class QLearningAgent:
        """Q学習エージェント"""
    
        def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
            """
            Args:
                n_states: 状態数
                n_actions: 行動数
                alpha: 学習率
                gamma: 割引率
                epsilon: ε-greedy のε
            """
            self.Q = np.zeros((n_states, n_actions))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.n_actions = n_actions
    
        def select_action(self, state):
            """ε-greedy方策で行動選択"""
            if np.random.rand() < self.epsilon:
                # ランダム行動（探索）
                return np.random.randint(self.n_actions)
            else:
                # 最良行動（活用）
                return np.argmax(self.Q[state])
    
        def update(self, state, action, reward, next_state, done):
            """Q値の更新"""
            if done:
                # 終端状態
                td_target = reward
            else:
                # Q学習の更新式
                td_target = reward + self.gamma * np.max(self.Q[next_state])
    
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
    
    
    def train_q_learning(env, agent, num_episodes=1000):
        """Q学習の訓練"""
        episode_rewards = []
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
    
            while True:
                # 行動選択
                action = agent.select_action(state)
    
                # 環境実行
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    
                # Q値更新
                agent.update(state, action, reward, next_state, done)
    
                total_reward += reward
    
                if done:
                    break
    
                state = next_state
    
            episode_rewards.append(total_reward)
    
            # 進捗表示
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
        return episode_rewards
    
    
    # 使用例：FrozenLake
    print("\n=== Q学習の訓練 ===")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    rewards = train_q_learning(env, agent, num_episodes=1000)
    
    print(f"\n学習済みQ表（一部）:")
    print(agent.Q[:16].reshape(4, 4, -1)[:, :, 0])  # 行動0のQ値
    env.close()
    

### Q表（Q-Table）の可視化
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def visualize_q_table(Q, env_shape=(4, 4)):
        """Q表を可視化"""
        n_states = Q.shape[0]
        n_actions = Q.shape[1]
    
        fig, axes = plt.subplots(1, n_actions, figsize=(16, 4))
    
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    
        for action in range(n_actions):
            Q_action = Q[:, action].reshape(env_shape)
    
            sns.heatmap(Q_action, annot=True, fmt='.2f', cmap='YlOrRd',
                       ax=axes[action], cbar=True, square=True)
            axes[action].set_title(f'Q値: {action_names[action]}')
            axes[action].set_xlabel('列')
            axes[action].set_ylabel('行')
    
        plt.tight_layout()
        plt.savefig('q_table_visualization.png', dpi=150, bbox_inches='tight')
        print("Q表を保存: q_table_visualization.png")
        plt.close()
    
    
    # Q表の可視化
    visualize_q_table(agent.Q)
    

### 学習曲線の可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_learning_curve(rewards, window=100):
        """学習曲線をプロット"""
        # 移動平均を計算
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
        plt.figure(figsize=(12, 5))
    
        # エピソード報酬
        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.3, label='エピソード報酬')
        plt.plot(range(window-1, len(rewards)), smoothed_rewards,
                 linewidth=2, label=f'{window}エピソード移動平均')
        plt.xlabel('エピソード')
        plt.ylabel('報酬')
        plt.title('Q学習の学習曲線')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # 累積報酬
        plt.subplot(1, 2, 2)
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, linewidth=2, color='green')
        plt.xlabel('エピソード')
        plt.ylabel('累積報酬')
        plt.title('累積報酬')
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('q_learning_curve.png', dpi=150, bbox_inches='tight')
        print("学習曲線を保存: q_learning_curve.png")
        plt.close()
    
    
    plot_learning_curve(rewards)
    

* * *

## 2.3 SARSA（State-Action-Reward-State-Action）

### SARSAの基本原理

**SARSA** は、Q学習の**オンポリシー版** です。実際に取る行動を使って更新します：

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right] $$ 

重要な違い：

  * Q学習：$\max_{a'} Q(s_{t+1}, a')$ を使う（最良の行動）
  * SARSA：$Q(s_{t+1}, a_{t+1})$ を使う（実際に取る行動）

    
    
    ```mermaid
    graph LR
        S1["S_t"] --> A1["A_t"]
        A1 --> R["R_t+1"]
        R --> S2["S_t+1"]
        S2 --> A2["A_t+1"]
        A2 --> Update["Q(S_t, A_t) 更新"]
    
        style S1 fill:#b3e5fc
        style A1 fill:#c5e1a5
        style R fill:#fff9c4
        style S2 fill:#b3e5fc
        style A2 fill:#c5e1a5
        style Update fill:#ffab91
    ```

### Q学習とSARSAの比較

項目 | Q学習 | SARSA  
---|---|---  
**学習タイプ** | オフポリシー | オンポリシー  
**更新式** | $r + \gamma \max_a Q(s', a)$ | $r + \gamma Q(s', a')$  
**探索の影響** | 学習に影響しない | 学習に影響する  
**収束先** | 最適方策 | 現在の方策の価値  
**安全性** | リスクを考慮しない | リスクを考慮  
**適用場面** | シミュレーション環境 | 実環境での学習  
  
### SARSAの実装
    
    
    import numpy as np
    import gym
    
    class SARSAAgent:
        """SARSAエージェント"""
    
        def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
            """
            Args:
                n_states: 状態数
                n_actions: 行動数
                alpha: 学習率
                gamma: 割引率
                epsilon: ε-greedy のε
            """
            self.Q = np.zeros((n_states, n_actions))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.n_actions = n_actions
    
        def select_action(self, state):
            """ε-greedy方策で行動選択"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_actions)
            else:
                return np.argmax(self.Q[state])
    
        def update(self, state, action, reward, next_state, next_action, done):
            """Q値の更新（SARSA）"""
            if done:
                td_target = reward
            else:
                # SARSAの更新式（次に実際に取る行動を使用）
                td_target = reward + self.gamma * self.Q[next_state, next_action]
    
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
    
    
    def train_sarsa(env, agent, num_episodes=1000):
        """SARSAの訓練"""
        episode_rewards = []
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            action = agent.select_action(state)  # 初期行動選択
            total_reward = 0
    
            while True:
                # 環境実行
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    
                if not done:
                    # 次の行動を選択（SARSAの特徴）
                    next_action = agent.select_action(next_state)
                else:
                    next_action = None
    
                # Q値更新
                agent.update(state, action, reward, next_state, next_action, done)
    
                total_reward += reward
    
                if done:
                    break
    
                state = next_state
                action = next_action  # 次の行動に移行
    
            episode_rewards.append(total_reward)
    
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
        return episode_rewards
    
    
    # 使用例
    print("\n=== SARSAの訓練 ===")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    sarsa_agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    sarsa_rewards = train_sarsa(env, sarsa_agent, num_episodes=1000)
    
    print(f"\n学習済みQ表（SARSA）:")
    print(sarsa_agent.Q[:16].reshape(4, 4, -1)[:, :, 0])
    env.close()
    

* * *

## 2.4 ε-greedy探索戦略

### 探索と活用のトレードオフ

強化学習では、**探索（Exploration）** と**活用（Exploitation）** のバランスが重要です：

  * **探索** ：新しい状態・行動を試して環境を理解する
  * **活用** ：現在の知識で最良の行動を選択する

### ε-greedy方策

最もシンプルな探索戦略：

$$ a = \begin{cases} \text{random action} & \text{確率 } \epsilon \\\ \arg\max_a Q(s, a) & \text{確率 } 1 - \epsilon \end{cases} $$ 

### εの減衰（Epsilon Decay）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class EpsilonGreedy:
        """ε-greedy方策（減衰機能付き）"""
    
        def __init__(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
            """
            Args:
                epsilon_start: 初期ε
                epsilon_end: 最小ε
                epsilon_decay: 減衰率
            """
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
    
        def select_action(self, Q, state, n_actions):
            """行動選択"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(n_actions)
            else:
                return np.argmax(Q[state])
    
        def decay(self):
            """εを減衰"""
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    
    # εの減衰パターンを可視化
    print("\n=== ε減衰パターンの可視化 ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 異なる減衰率
    decay_rates = [0.99, 0.995, 0.999]
    
    for i, decay_rate in enumerate(decay_rates):
        epsilon_greedy = EpsilonGreedy(epsilon_start=1.0, epsilon_end=0.01,
                                       epsilon_decay=decay_rate)
        epsilons = [epsilon_greedy.epsilon]
    
        for _ in range(1000):
            epsilon_greedy.decay()
            epsilons.append(epsilon_greedy.epsilon)
    
        axes[i].plot(epsilons, linewidth=2)
        axes[i].set_xlabel('エピソード')
        axes[i].set_ylabel('ε')
        axes[i].set_title(f'減衰率 = {decay_rate}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('epsilon_decay.png', dpi=150, bbox_inches='tight')
    print("ε減衰パターンを保存: epsilon_decay.png")
    plt.close()
    

### 他の探索戦略

#### ソフトマックス（Boltzmann）探索

行動の価値に基づいて確率的に選択：

$$ P(a | s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)} $$ 

$\tau$ は温度パラメータ（高いほどランダム）

#### Upper Confidence Bound (UCB)

不確実性を考慮した探索：

$$ a = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a)}} \right] $$ 

$N(s,a)$ は行動 $a$ の選択回数、$c$ は探索係数

* * *

## 2.5 ハイパーパラメータの影響

### 学習率（Learning Rate）α

学習率 $\alpha$ は更新の強さを制御します：

  * **大きいα（例：0.5）** ：速く学習するが不安定
  * **小さいα（例：0.01）** ：安定だが収束が遅い
  * **推奨値** ：0.1 〜 0.3

### 割引率（Discount Factor）γ

割引率 $\gamma$ は将来の報酬の重要度を決定：

  * **γ = 0** ：即座の報酬のみ考慮（近視眼的）
  * **γ → 1** ：遠い将来まで考慮（先見的）
  * **推奨値** ：0.95 〜 0.99

### ハイパーパラメータ調査の実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import gym
    
    def hyperparameter_search(env_name, param_name, param_values, num_episodes=500):
        """ハイパーパラメータの影響を調査"""
        results = {}
    
        for value in param_values:
            print(f"\n{param_name} = {value} で訓練中...")
    
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
    
    
    # 学習率の影響調査
    print("=== 学習率αの影響調査 ===")
    
    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
    alpha_results = hyperparameter_search('FrozenLake-v1', 'alpha', alpha_values)
    
    # 可視化
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    for alpha, rewards in alpha_results.items():
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(smoothed, label=f'α = {alpha}', linewidth=2)
    
    plt.xlabel('エピソード')
    plt.ylabel('平均報酬')
    plt.title('学習率αの影響')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 割引率の影響調査
    gamma_values = [0.5, 0.9, 0.95, 0.99, 0.999]
    gamma_results = hyperparameter_search('FrozenLake-v1', 'gamma', gamma_values)
    
    plt.subplot(1, 2, 2)
    for gamma, rewards in gamma_results.items():
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(smoothed, label=f'γ = {gamma}', linewidth=2)
    
    plt.xlabel('エピソード')
    plt.ylabel('平均報酬')
    plt.title('割引率γの影響')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_impact.png', dpi=150, bbox_inches='tight')
    print("\nハイパーパラメータの影響を保存: hyperparameter_impact.png")
    plt.close()
    

* * *

## 2.6 実践：Taxi-v3環境

### Taxi-v3環境の概要

**Taxi-v3** は、タクシーが乗客をピックアップして目的地まで送る環境です：

  * **状態空間** ：500状態（5×5グリッド × 5乗客位置 × 4目的地）
  * **行動空間** ：6行動（上下左右、乗客の乗降）
  * **報酬** ：正しい目的地に到着 +20、1ステップごと -1、不正な乗降 -10

### Taxi-v3でのQ学習
    
    
    import numpy as np
    import gym
    import matplotlib.pyplot as plt
    
    # Taxi-v3環境
    print("=== Taxi-v3環境でのQ学習 ===")
    
    env = gym.make('Taxi-v3', render_mode=None)
    
    print(f"状態空間: {env.observation_space.n}")
    print(f"行動空間: {env.action_space.n}")
    print(f"行動: {['南', '北', '東', '西', '乗車', '降車']}")
    
    # Q学習エージェント
    taxi_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    # 訓練
    taxi_rewards = train_q_learning(env, taxi_agent, num_episodes=5000)
    
    # 学習曲線
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    smoothed = np.convolve(taxi_rewards, np.ones(100)/100, mode='valid')
    plt.plot(smoothed, linewidth=2, color='blue')
    plt.xlabel('エピソード')
    plt.ylabel('平均報酬')
    plt.title('Taxi-v3 Q学習の学習曲線')
    plt.grid(True, alpha=0.3)
    
    # 成功率の計算
    success_rate = []
    window = 100
    for i in range(len(taxi_rewards) - window):
        success = np.sum(np.array(taxi_rewards[i:i+window]) > 0) / window
        success_rate.append(success)
    
    plt.subplot(1, 2, 2)
    plt.plot(success_rate, linewidth=2, color='green')
    plt.xlabel('エピソード')
    plt.ylabel('成功率')
    plt.title('タスク成功率（100エピソード移動平均）')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('taxi_training.png', dpi=150, bbox_inches='tight')
    print("Taxi訓練結果を保存: taxi_training.png")
    plt.close()
    
    env.close()
    

### 学習済みエージェントの評価
    
    
    def evaluate_agent(env, agent, num_episodes=100, render=False):
        """学習済みエージェントを評価"""
        total_rewards = []
        total_steps = []
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
    
            while steps < 200:  # 最大ステップ数
                # 最良の行動を選択（探索なし）
                action = np.argmax(agent.Q[state])
    
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
    
                if terminated or truncated:
                    break
    
            total_rewards.append(episode_reward)
            total_steps.append(steps)
    
        return total_rewards, total_steps
    
    
    # 評価
    print("\n=== 学習済みエージェントの評価 ===")
    
    env = gym.make('Taxi-v3', render_mode=None)
    eval_rewards, eval_steps = evaluate_agent(env, taxi_agent, num_episodes=100)
    
    print(f"平均報酬: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"平均ステップ数: {np.mean(eval_steps):.2f} ± {np.std(eval_steps):.2f}")
    print(f"成功率: {np.sum(np.array(eval_rewards) > 0) / len(eval_rewards) * 100:.1f}%")
    
    env.close()
    

* * *

## 2.7 実践：Cliff Walking環境

### Cliff Walking環境の定義

**Cliff Walking** は、崖を避けてゴールに到達する環境です。Q学習とSARSAの違いを明確に示す例です：

  * **4×12グリッド** ：左下がスタート、右下がゴール
  * **崖エリア** ：下端の中央部分（踏むと -100 の罰則）
  * **報酬** ：各ステップ -1、崖 -100、ゴール 0

### Cliff Walking環境の実装
    
    
    import numpy as np
    import gym
    
    # Cliff Walking環境
    print("=== Cliff Walking環境 ===")
    
    env = gym.make('CliffWalking-v0')
    
    print(f"状態空間: {env.observation_space.n}")
    print(f"行動空間: {env.action_space.n}")
    print(f"グリッドサイズ: 4×12")
    
    # Q学習エージェント
    cliff_q_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.5,
        gamma=0.99,
        epsilon=0.1
    )
    
    # SARSAエージェント
    cliff_sarsa_agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.5,
        gamma=0.99,
        epsilon=0.1
    )
    
    # 訓練
    print("\nQ学習で訓練中...")
    q_rewards = train_q_learning(env, cliff_q_agent, num_episodes=500)
    
    env = gym.make('CliffWalking-v0')
    print("\nSARSAで訓練中...")
    sarsa_rewards = train_sarsa(env, cliff_sarsa_agent, num_episodes=500)
    
    # 比較可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    smoothed_q = np.convolve(q_rewards, np.ones(10)/10, mode='valid')
    smoothed_sarsa = np.convolve(sarsa_rewards, np.ones(10)/10, mode='valid')
    
    plt.plot(smoothed_q, label='Q学習', linewidth=2)
    plt.plot(smoothed_sarsa, label='SARSA', linewidth=2)
    plt.xlabel('エピソード')
    plt.ylabel('報酬')
    plt.title('Cliff Walking: Q学習 vs SARSA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 方策の可視化（矢印で表示）
    plt.subplot(1, 2, 2)
    
    def visualize_policy(Q, shape=(4, 12)):
        """学習した方策を可視化"""
        policy = np.argmax(Q, axis=1)
        policy_grid = policy.reshape(shape)
    
        # 矢印の方向
        arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(np.zeros(shape), cmap='Blues', alpha=0.3)
    
        for i in range(shape[0]):
            for j in range(shape[1]):
                state = i * shape[1] + j
                action = policy[state]
    
                # 崖エリアを赤く表示
                if i == 3 and 1 <= j <= 10:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=True, color='red', alpha=0.3))
    
                # ゴール
                if i == 3 and j == 11:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=True, color='green', alpha=0.3))
    
                # 矢印
                ax.text(j, i, arrows[action], ha='center', va='center',
                       fontsize=16, fontweight='bold')
    
        ax.set_xlim(-0.5, shape[1]-0.5)
        ax.set_ylim(shape[0]-0.5, -0.5)
        ax.set_xticks(range(shape[1]))
        ax.set_yticks(range(shape[0]))
        ax.grid(True)
        ax.set_title('学習した方策（Q学習）')
    
    
    visualize_policy(cliff_q_agent.Q)
    
    plt.tight_layout()
    plt.savefig('cliff_walking_comparison.png', dpi=150, bbox_inches='tight')
    print("\nCliff Walking比較を保存: cliff_walking_comparison.png")
    plt.close()
    
    env.close()
    

### Q学習とSARSAの経路の違い

> **重要な観察** ：Cliff Walkingでは、Q学習は**最短経路（崖の近く）** を学習しますが、SARSAは**安全な経路（崖から離れる）** を学習します。これはε-greedy探索による偶発的な崖への転落をSARSAが学習に反映するためです。

* * *

## 演習問題

**演習1：Q学習とSARSAの収束速度比較**

FrozenLake環境でQ学習とSARSAを同じハイパーパラメータで訓練し、収束速度を比較してください。
    
    
    import gym
    import numpy as np
    
    # TODO: Q学習とSARSAを同じ設定で訓練
    # TODO: エピソードごとの報酬をプロット
    # TODO: 収束に必要なエピソード数を比較
    # 期待: 環境によって収束速度が異なる
    

**演習2：ε減衰スケジュールの最適化**

異なるε減衰パターン（線形減衰、指数減衰、ステップ減衰）を実装し、Taxi-v3での性能を比較してください。
    
    
    import numpy as np
    
    # TODO: 3種類のε減衰スケジュールを実装
    # TODO: Taxi-v3で各スケジュールを評価
    # TODO: 学習曲線と最終性能を比較
    # ヒント: 初期は探索重視、後半は活用重視
    

**演習3：Double Q-Learning の実装**

過大評価を防ぐDouble Q-Learningを実装し、通常のQ学習と性能を比較してください。
    
    
    import numpy as np
    
    # TODO: 2つのQ表を使うDouble Q-Learningを実装
    # TODO: FrozenLake環境で訓練
    # TODO: Q値の推定誤差を通常のQ学習と比較
    # 理論: Doubleアルゴリズムは過大評価バイアスを軽減
    

**演習4：学習率の適応的調整**

訪問回数に応じて学習率を調整する適応的学習率を実装し、固定学習率と比較してください。
    
    
    import numpy as np
    
    # TODO: α(s,a) = 1 / (1 + N(s,a)) の適応的学習率を実装
    # TODO: 固定学習率と性能を比較
    # TODO: 各状態での訪問回数を可視化
    # 期待: 適応的学習率で収束が安定する
    

**演習5：独自環境での実験**

OpenAI Gymの別の環境（CartPole-v1、MountainCar-v0など）で状態の離散化を行い、Q学習を適用してください。
    
    
    import gym
    import numpy as np
    
    # TODO: 連続状態空間を離散化する関数を実装
    # TODO: 離散化したCartPole環境でQ学習
    # TODO: 離散化の粒度と性能の関係を調査
    # 課題: 連続空間の適切な離散化が重要
    

* * *

## まとめ

この章では、時間差分学習に基づくQ学習とSARSAを学びました。

### 重要ポイント

  * **TD学習** ：エピソード終了を待たずにステップごとに更新
  * **Q学習** ：オフポリシー型、最適方策を直接学習
  * **SARSA** ：オンポリシー型、探索の影響を考慮した学習
  * **ε-greedy** ：探索と活用のバランスを制御するシンプルな方策
  * **学習率α** ：更新の強さを制御（0.1〜0.3が推奨）
  * **割引率γ** ：将来の報酬の重要度（0.95〜0.99が推奨）
  * **Q表** ：状態-行動ペアごとの価値を保存
  * **適用場面** ：離散状態・行動空間のタスク

### Q学習とSARSAの使い分け

状況 | 推奨アルゴリズム | 理由  
---|---|---  
**シミュレーション環境** | Q学習 | 最適方策を効率的に学習  
**実環境・ロボット** | SARSA | 安全な方策を学習  
**危険な状態がある** | SARSA | リスク回避の傾向  
**高速な収束が必要** | Q学習 | オフポリシーで柔軟  
  
### 次のステップ

次章では、**Deep Q-Network (DQN)** について学びます。Q表では扱えない大規模・連続状態空間に対して、ニューラルネットワークで行動価値関数を近似する手法、Experience Replay、Target Network、Atariゲームでの応用などを習得します。
