---
title: 第1章：強化学習の基礎
chapter_title: 第1章：強化学習の基礎
subtitle: 強化学習の基本概念、マルコフ決定過程、価値関数とベルマン方程式、基本アルゴリズムの理解
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 8
exercises: 6
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 強化学習と教師あり学習・教師なし学習の違いを理解する
  * ✅ マルコフ決定過程（MDP）の基本概念（状態、行動、報酬、遷移）を説明できる
  * ✅ ベルマン方程式の意味と役割を理解する
  * ✅ 価値関数（V）と行動価値関数（Q）の違いを説明できる
  * ✅ 方策（Policy）の概念と最適方策の定義を理解する
  * ✅ 探索（Exploration）と活用（Exploitation）のトレードオフを理解する
  * ✅ 価値反復法（Value Iteration）を実装できる
  * ✅ 方策反復法（Policy Iteration）を実装できる
  * ✅ モンテカルロ法の基本原理を理解する
  * ✅ TD学習（Temporal Difference）の仕組みを理解し実装できる

* * *

## 1.1 強化学習とは何か

### 強化学習の基本概念

強化学習（Reinforcement Learning）は、エージェントが環境と相互作用しながら、試行錯誤を通じて最適な行動を学習する機械学習の一分野です。ゲームAI、ロボット制御、自動運転、推薦システムなど、幅広い分野で応用されています。

> 「強化学習は、報酬信号を最大化するために、どのような行動を取るべきかを学習する問題である。」

#### 強化学習の構成要素

  * **エージェント（Agent）** ：学習し行動を決定する主体
  * **環境（Environment）** ：エージェントが相互作用する対象
  * **状態（State）** ：環境の現在の状況を表す情報
  * **行動（Action）** ：エージェントが選択できる動作
  * **報酬（Reward）** ：行動の良し悪しを示す即時的なフィードバック
  * **方策（Policy）** ：状態から行動への写像（意思決定ルール）

    
    
    ```mermaid
    graph LR
        A[エージェント] -->|行動 At| B[環境]
        B -->|状態 St+1| A
        B -->|報酬 Rt+1| A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
    ```

### 強化学習 vs 教師あり学習 vs 教師なし学習

学習方法 | データの特徴 | 学習の目的 | 具体例  
---|---|---|---  
**教師あり学習** | 入力と正解ラベルのペア | 正解を予測するモデルを構築 | 画像分類、音声認識  
**教師なし学習** | ラベルなしデータ | データの構造やパターンを発見 | クラスタリング、次元削減  
**強化学習** | 行動と報酬のフィードバック | 累積報酬を最大化する方策を学習 | ゲームAI、ロボット制御  
  
#### 強化学習の特徴

  1. **試行錯誤による学習** ：正解は与えられず、報酬信号から学習
  2. **遅延報酬** ：行動の結果が即座に分からないことが多い
  3. **探索と活用のトレードオフ** ：新しい行動を試すか、既知の良い行動を取るか
  4. **逐次的意思決定** ：過去の行動が未来の状態に影響する

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== 強化学習 vs 教師あり学習の比較 ===\n")
    
    # 教師あり学習の例：単純な回帰
    print("【教師あり学習】")
    print("目的: データから関数を学習する")
    X_train = np.array([1, 2, 3, 4, 5])
    y_train = np.array([2, 4, 6, 8, 10])  # y = 2x の関係
    
    # 最小二乗法で学習
    slope = np.sum((X_train - X_train.mean()) * (y_train - y_train.mean())) / \
            np.sum((X_train - X_train.mean())**2)
    intercept = y_train.mean() - slope * X_train.mean()
    
    print(f"学習結果: y = {slope:.2f}x + {intercept:.2f}")
    print("特徴: 正解（y_train）が明示的に与えられる\n")
    
    # 強化学習の例：簡単なバンディット問題
    print("【強化学習】")
    print("目的: 報酬を最大化する行動を学習する")
    
    class SimpleBandit:
        """3本腕のバンディット問題"""
        def __init__(self):
            # 各腕の真の期待報酬（エージェントには未知）
            self.true_values = np.array([0.3, 0.5, 0.7])
    
        def pull(self, action):
            """腕を引いて報酬を得る"""
            # ベルヌーイ分布から報酬を生成
            reward = 1 if np.random.rand() < self.true_values[action] else 0
            return reward
    
    # ε-greedy アルゴリズムで学習
    bandit = SimpleBandit()
    n_arms = 3
    n_steps = 1000
    epsilon = 0.1
    
    Q = np.zeros(n_arms)  # 各腕の推定価値
    N = np.zeros(n_arms)  # 各腕を引いた回数
    rewards = []
    
    for step in range(n_steps):
        # ε-greedy 方策で行動選択
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)  # 探索
        else:
            action = np.argmax(Q)  # 活用
    
        # 報酬を得る
        reward = bandit.pull(action)
        rewards.append(reward)
    
        # Q値を更新
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
    
    print(f"真の期待報酬: {bandit.true_values}")
    print(f"学習した推定値: {Q}")
    print(f"平均報酬: {np.mean(rewards):.3f}")
    print("特徴: 正解は不明、報酬信号から試行錯誤で学習\n")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左：教師あり学習
    axes[0].scatter(X_train, y_train, s=100, alpha=0.6, label='訓練データ（正解あり）')
    X_test = np.linspace(0, 6, 100)
    y_pred = slope * X_test + intercept
    axes[0].plot(X_test, y_pred, 'r-', linewidth=2, label=f'学習したモデル')
    axes[0].set_xlabel('入力 X', fontsize=12)
    axes[0].set_ylabel('出力 y', fontsize=12)
    axes[0].set_title('教師あり学習：正解データから学習', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 右：強化学習
    window = 50
    cumulative_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[1].plot(cumulative_rewards, linewidth=2)
    axes[1].axhline(y=max(bandit.true_values), color='r', linestyle='--',
                    linewidth=2, label=f'最適報酬 ({max(bandit.true_values):.1f})')
    axes[1].set_xlabel('ステップ数', fontsize=12)
    axes[1].set_ylabel('平均報酬（移動平均）', fontsize=12)
    axes[1].set_title('強化学習：試行錯誤で最適行動を学習', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_vs_supervised.png', dpi=150, bbox_inches='tight')
    print("可視化を 'rl_vs_supervised.png' に保存しました。")
    

* * *

## 1.2 マルコフ決定過程（MDP）

### MDPの定義

マルコフ決定過程（Markov Decision Process）は、強化学習の数学的基盤を提供するフレームワークです。MDPは5つ組 $(S, A, P, R, \gamma)$ で定義されます：

  * $S$：状態空間（State space）
  * $A$：行動空間（Action space）
  * $P$：状態遷移確率 $P(s'|s, a)$
  * $R$：報酬関数 $R(s, a, s')$
  * $\gamma$：割引率（Discount factor）$\in [0, 1]$

> 「マルコフ性：次の状態は現在の状態と行動のみに依存し、過去の履歴には依存しない。」

#### マルコフ性の数学的表現

状態 $s$ がマルコフ性を満たすとき：

$$ P(S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0) = P(S_{t+1}|S_t, A_t) $$ 

### 状態、行動、報酬、遷移

#### 1\. 状態（State）

状態は環境の現在の状況を表す情報です：

  * **完全観測** ：エージェントが環境の全情報を観測できる（例：チェス）
  * **部分観測** ：一部の情報のみ観測可能（例：ポーカー）

#### 2\. 行動（Action）

  * **離散行動空間** ：有限個の行動（例：上下左右の移動）
  * **連続行動空間** ：実数値の行動（例：ロボットの関節角度）

#### 3\. 報酬（Reward）

報酬は、時刻 $t$ での行動の良さを示すスカラー値 $r_t$ です。エージェントの目標は累積報酬（リターン）の期待値を最大化することです：

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$ 

#### 4\. 割引率 $\gamma$

  * $\gamma = 0$：即時報酬のみを考慮（近視眼的）
  * $\gamma = 1$：全ての将来報酬を等しく重視
  * $0 < \gamma < 1$：将来の報酬を割り引く（一般的には0.9〜0.99）

    
    
    ```mermaid
    graph TD
        S0[状態 S0] -->|行動 a0| S1[状態 S1]
        S1 -->|報酬 r1| S0
        S1 -->|行動 a1| S2[状態 S2]
        S2 -->|報酬 r2| S1
        S2 -->|行動 a2| S3[状態 S3]
        S3 -->|報酬 r3| S2
    
        style S0 fill:#e3f2fd
        style S1 fill:#fff3e0
        style S2 fill:#e8f5e9
        style S3 fill:#fce4ec
    ```
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== マルコフ決定過程（MDP）の基本 ===\n")
    
    class SimpleMDP:
        """
        簡単なMDPの例：3x3のグリッドワールド
    
        状態: (x, y) 座標
        行動: 上下左右の移動（0:上, 1:右, 2:下, 3:左）
        報酬: ゴール到達で+1、その他0
        """
        def __init__(self):
            self.grid_size = 3
            self.start_state = (0, 0)
            self.goal_state = (2, 2)
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
            self.action_names = ['上', '右', '下', '左']
    
        def is_valid_state(self, state):
            """状態が有効かチェック"""
            x, y = state
            return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
        def step(self, state, action):
            """
            状態遷移関数
    
            Returns:
            --------
            next_state : tuple
            reward : float
            done : bool
            """
            if state == self.goal_state:
                return state, 0, True
    
            # 次の状態を計算
            dx, dy = self.actions[action]
            next_state = (state[0] + dx, state[1] + dy)
    
            # 壁にぶつかる場合は元の状態に留まる
            if not self.is_valid_state(next_state):
                next_state = state
    
            # 報酬を計算
            reward = 1.0 if next_state == self.goal_state else 0.0
            done = (next_state == self.goal_state)
    
            return next_state, reward, done
    
        def get_all_states(self):
            """全ての状態を取得"""
            return [(x, y) for x in range(self.grid_size)
                    for y in range(self.grid_size)]
    
    # MDPのインスタンス化
    mdp = SimpleMDP()
    
    print("【MDP の構成要素】")
    print(f"状態空間 S: {mdp.get_all_states()}")
    print(f"行動空間 A: {mdp.action_names}")
    print(f"開始状態: {mdp.start_state}")
    print(f"ゴール状態: {mdp.goal_state}\n")
    
    # マルコフ性のデモンストレーション
    print("【マルコフ性の検証】")
    current_state = (1, 1)
    action = 1  # 右
    
    print(f"現在の状態: {current_state}")
    print(f"選択した行動: {mdp.action_names[action]}")
    
    # 状態遷移を実行
    next_state, reward, done = mdp.step(current_state, action)
    print(f"次の状態: {next_state}")
    print(f"報酬: {reward}")
    print(f"終了: {done}")
    print("\n→ 次の状態は現在の状態と行動のみで決まる（マルコフ性）\n")
    
    # 割引率の影響を可視化
    print("【割引率 γ の影響】")
    gammas = [0.0, 0.5, 0.9, 0.99]
    rewards = np.array([1, 1, 1, 1, 1])  # 5ステップ連続で報酬1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左：累積報酬の計算
    axes[0].set_title('割引率による累積報酬の違い', fontsize=14, fontweight='bold')
    for gamma in gammas:
        discounted_rewards = [gamma**i * r for i, r in enumerate(rewards)]
        cumulative = np.cumsum(discounted_rewards)
        axes[0].plot(cumulative, marker='o', label=f'γ={gamma}', linewidth=2)
    
    axes[0].set_xlabel('ステップ数', fontsize=12)
    axes[0].set_ylabel('累積報酬', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 右：各ステップでの重み
    axes[1].set_title('各ステップの報酬に対する重み', fontsize=14, fontweight='bold')
    steps = np.arange(10)
    for gamma in gammas:
        weights = [gamma**i for i in steps]
        axes[1].plot(steps, weights, marker='o', label=f'γ={gamma}', linewidth=2)
    
    axes[1].set_xlabel('将来のステップ数', fontsize=12)
    axes[1].set_ylabel('重み (γ^k)', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mdp_basics.png', dpi=150, bbox_inches='tight')
    print("可視化を 'mdp_basics.png' に保存しました。")
    

* * *

## 1.3 ベルマン方程式と価値関数

### 価値関数（Value Function）

価値関数は、ある状態または状態-行動ペアがどれだけ良いかを評価する関数です。

#### 状態価値関数 $V^\pi(s)$

方策 $\pi$ に従ったときの、状態 $s$ からの期待累積報酬：

$$ V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right] $$ 

#### 行動価値関数 $Q^\pi(s, a)$

状態 $s$ で行動 $a$ を取り、その後方策 $\pi$ に従ったときの期待累積報酬：

$$ Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right] $$ 

#### V と Q の関係

$$ V^\pi(s) = \sum_{a} \pi(a|s) Q^\pi(s, a) $$ 

### ベルマン方程式

ベルマン方程式は、価値関数を再帰的に定義します。これは動的計画法の基礎となる重要な方程式です。

#### ベルマン期待方程式（Bellman Expectation Equation）

状態価値関数：

$$ V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^\pi(s')\right] $$ 

行動価値関数：

$$ Q^\pi(s, a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')\right] $$ 

#### ベルマン最適方程式（Bellman Optimality Equation）

最適状態価値関数 $V^*(s)$：

$$ V^*(s) = \max_{a} \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^*(s')\right] $$ 

最適行動価値関数 $Q^*(s, a)$：

$$ Q^*(s, a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\right] $$ 

> 「ベルマン方程式の直感：現在の価値 = 即時報酬 + 将来の価値の割引」

### 方策（Policy）

方策 $\pi$ は、状態から行動への写像です：

  * **決定的方策** ：$a = \pi(s)$（各状態で1つの行動を決定）
  * **確率的方策** ：$\pi(a|s)$（各状態で行動の確率分布）

#### 最適方策 $\pi^*$

全ての状態で最大の価値を達成する方策：

$$ \pi^*(s) = \arg\max_{a} Q^*(s, a) $$ 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== ベルマン方程式と価値関数 ===\n")
    
    class GridWorld:
        """
        4x4 グリッドワールド MDP
        """
        def __init__(self, grid_size=4):
            self.size = grid_size
            self.n_states = grid_size * grid_size
            self.n_actions = 4  # 上右下左
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            self.action_names = ['↑', '→', '↓', '←']
    
            # 終端状態（左上と右下）
            self.terminal_states = [(0, 0), (grid_size-1, grid_size-1)]
    
        def state_to_index(self, state):
            """(x, y) を 1次元インデックスに変換"""
            return state[0] * self.size + state[1]
    
        def index_to_state(self, index):
            """1次元インデックスを (x, y) に変換"""
            return (index // self.size, index % self.size)
    
        def is_terminal(self, state):
            """終端状態かチェック"""
            return state in self.terminal_states
    
        def get_next_state(self, state, action):
            """次の状態を取得"""
            if self.is_terminal(state):
                return state
    
            dx, dy = self.actions[action]
            next_state = (state[0] + dx, state[1] + dy)
    
            # グリッドの範囲内かチェック
            if (0 <= next_state[0] < self.size and
                0 <= next_state[1] < self.size):
                return next_state
            else:
                return state  # 壁にぶつかる場合は移動しない
    
        def get_reward(self, state, action, next_state):
            """報酬を取得"""
            if self.is_terminal(state):
                return 0
            return -1  # 各ステップで-1の報酬（最短経路を見つける動機付け）
    
    def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
        """
        方策評価：与えられた方策の価値関数を計算
    
        Parameters:
        -----------
        env : GridWorld
        policy : ndarray
            各状態での行動確率分布 [n_states, n_actions]
        gamma : float
            割引率
        theta : float
            収束判定の閾値
    
        Returns:
        --------
        V : ndarray
            状態価値関数
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
                # ベルマン期待方程式
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
    
        print(f"方策評価が {iteration} 回の反復で収束しました")
        return V
    
    # グリッドワールドの作成
    env = GridWorld(grid_size=4)
    
    # ランダム方策（各行動を等確率で選択）
    random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    
    print("【ランダム方策の評価】")
    V_random = policy_evaluation(env, random_policy, gamma=0.9)
    
    # 価値関数を2次元グリッドで表示
    V_grid = V_random.reshape((env.size, env.size))
    print("\n状態価値関数 V(s):")
    print(V_grid)
    print()
    
    # 最適な方策（貪欲方策）を計算
    def compute_greedy_policy(env, V, gamma=0.9):
        """
        価値関数から貪欲方策を計算
        """
        policy = np.zeros((env.n_states, env.n_actions))
    
        for s_idx in range(env.n_states):
            state = env.index_to_state(s_idx)
    
            if env.is_terminal(state):
                policy[s_idx] = 1.0 / env.n_actions  # 終端状態では均等
                continue
    
            # 各行動のQ値を計算
            q_values = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(state, action, next_state)
                next_s_idx = env.state_to_index(next_state)
                q_values[action] = reward + gamma * V[next_s_idx]
    
            # 最大Q値を持つ行動を選択
            best_action = np.argmax(q_values)
            policy[s_idx, best_action] = 1.0
    
        return policy
    
    greedy_policy = compute_greedy_policy(env, V_random, gamma=0.9)
    
    # 方策の可視化
    def visualize_policy(env, policy):
        """方策を矢印で可視化"""
        policy_grid = np.zeros((env.size, env.size), dtype=object)
    
        for s_idx in range(env.n_states):
            state = env.index_to_state(s_idx)
            if env.is_terminal(state):
                policy_grid[state] = 'T'
            else:
                action = np.argmax(policy[s_idx])
                policy_grid[state] = env.action_names[action]
    
        return policy_grid
    
    print("【貪欲方策（ランダム方策から導出）】")
    policy_grid = visualize_policy(env, greedy_policy)
    print(policy_grid)
    print()
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左：状態価値関数
    im1 = axes[0].imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title('状態価値関数 V(s)\n（ランダム方策）',
                      fontsize=14, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            text = axes[0].text(j, i, f'{V_grid[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=11)
    axes[0].set_xticks(range(env.size))
    axes[0].set_yticks(range(env.size))
    plt.colorbar(im1, ax=axes[0])
    
    # 右：貪欲方策
    policy_display = np.zeros((env.size, env.size))
    axes[1].imshow(policy_display, cmap='Blues', alpha=0.3)
    axes[1].set_title('貪欲方策\n（V(s)から導出）',
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
    print("可視化を 'value_function_bellman.png' に保存しました。")
    

* * *

## 1.4 価値反復法と方策反復法

### 価値反復法（Value Iteration）

価値反復法は、ベルマン最適方程式を反復的に適用して最適価値関数を求めるアルゴリズムです。

#### アルゴリズム

  1. $V(s)$ を任意の値（通常は0）で初期化
  2. 各状態 $s$ について、以下を反復： $$ V_{k+1}(s) = \max_{a} \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V_k(s')\right] $$ 
  3. $V_k$ が収束するまで繰り返す
  4. 最適方策を抽出：$\pi^*(s) = \arg\max_{a} Q^*(s, a)$

### 方策反復法（Policy Iteration）

方策反復法は、方策評価と方策改善を交互に行います。

#### アルゴリズム

  1. **初期化** ：任意の方策 $\pi$ を選ぶ
  2. **方策評価** ：$V^\pi$ を計算
  3. **方策改善** ：$\pi' = \text{greedy}(V^\pi)$
  4. $\pi' = \pi$ なら終了、そうでなければ $\pi = \pi'$ として2へ

手法 | 特徴 | 収束速度 | 適用場面  
---|---|---|---  
**価値反復法** | 価値関数を直接最適化 | 遅い | シンプルな実装が必要  
**方策反復法** | 方策を繰り返し改善 | 速い | 方策が重要な場合  
      
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== 価値反復法と方策反復法の実装 ===\n")
    
    class GridWorldEnv:
        """グリッドワールド環境"""
        def __init__(self, size=4):
            self.size = size
            self.n_states = size * size
            self.n_actions = 4
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
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
                return state, -1  # 壁にぶつかっても報酬-1
    
    def value_iteration(env, gamma=0.9, theta=1e-6):
        """
        価値反復法
    
        Returns:
        --------
        V : ndarray
            最適状態価値関数
        policy : ndarray
            最適方策
        iterations : int
            収束までの反復回数
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
    
                # 各行動のQ値を計算
                q_values = np.zeros(env.n_actions)
                for action in range(env.n_actions):
                    next_state, reward = env.step(state, action)
                    next_s_idx = env.state_to_index(next_state)
                    q_values[action] = reward + gamma * V_old[next_s_idx]
    
                # ベルマン最適方程式：V(s) = max_a Q(s, a)
                V[s_idx] = np.max(q_values)
                policy[s_idx] = np.argmax(q_values)
    
                delta = max(delta, abs(V[s_idx] - V_old[s_idx]))
    
            iteration += 1
            if delta < theta:
                break
    
        return V, policy, iteration
    
    def policy_iteration(env, gamma=0.9, theta=1e-6):
        """
        方策反復法
    
        Returns:
        --------
        V : ndarray
            最適状態価値関数
        policy : ndarray
            最適方策
        iterations : int
            方策改善の回数
        """
        # ランダム方策で初期化
        policy = np.random.randint(0, env.n_actions, size=env.n_states)
    
        iteration = 0
        while True:
            # 1. 方策評価
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
    
            # 2. 方策改善
            policy_stable = True
            for s_idx in range(env.n_states):
                state = env.index_to_state(s_idx)
    
                if env.is_terminal(state):
                    continue
    
                old_action = policy[s_idx]
    
                # 貪欲方策を計算
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
    
    # 環境の作成
    env = GridWorldEnv(size=4)
    
    # 価値反復法の実行
    print("【価値反復法】")
    V_vi, policy_vi, iter_vi = value_iteration(env, gamma=0.9)
    print(f"収束までの反復回数: {iter_vi}")
    print(f"\n最適状態価値関数:")
    print(V_vi.reshape(env.size, env.size))
    print()
    
    # 方策反復法の実行
    print("【方策反復法】")
    V_pi, policy_pi, iter_pi = policy_iteration(env, gamma=0.9)
    print(f"方策改善の回数: {iter_pi}")
    print(f"\n最適状態価値関数:")
    print(V_pi.reshape(env.size, env.size))
    print()
    
    # 方策の可視化
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
    
    print("【最適方策（価値反復法）】")
    policy_grid_vi = visualize_policy_grid(env, policy_vi)
    print(policy_grid_vi)
    print()
    
    print("【最適方策（方策反復法）】")
    policy_grid_pi = visualize_policy_grid(env, policy_pi)
    print(policy_grid_pi)
    print()
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 価値反復法の結果
    V_grid_vi = V_vi.reshape(env.size, env.size)
    im1 = axes[0, 0].imshow(V_grid_vi, cmap='RdYlGn', interpolation='nearest')
    axes[0, 0].set_title('価値反復法：最適価値関数', fontsize=12, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            axes[0, 0].text(j, i, f'{V_grid_vi[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 価値反復法の方策
    axes[0, 1].imshow(np.zeros((env.size, env.size)), cmap='Blues', alpha=0.3)
    axes[0, 1].set_title(f'価値反復法：最適方策\n(反復回数: {iter_vi})',
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
    
    # 方策反復法の結果
    V_grid_pi = V_pi.reshape(env.size, env.size)
    im2 = axes[1, 0].imshow(V_grid_pi, cmap='RdYlGn', interpolation='nearest')
    axes[1, 0].set_title('方策反復法：最適価値関数', fontsize=12, fontweight='bold')
    for i in range(env.size):
        for j in range(env.size):
            axes[1, 0].text(j, i, f'{V_grid_pi[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 方策反復法の方策
    axes[1, 1].imshow(np.zeros((env.size, env.size)), cmap='Blues', alpha=0.3)
    axes[1, 1].set_title(f'方策反復法：最適方策\n(方策改善回数: {iter_pi})',
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
    print("可視化を 'value_policy_iteration.png' に保存しました。")
    

* * *

## 1.5 探索と活用のトレードオフ

### 探索（Exploration）vs 活用（Exploitation）

強化学習における最も重要な課題の一つが、探索と活用のトレードオフです。

> 「探索：新しい行動を試して、より良い選択肢を見つける」  
>  「活用：既知の最良の行動を選択して、報酬を最大化する」

#### 多腕バンディット問題

$K$ 本の腕を持つスロットマシン（バンディット）があり、各腕 $i$ は異なる期待報酬 $\mu_i$ を持ちます。目標は、累積報酬を最大化することです。

### 探索戦略

戦略 | 説明 | 特徴  
---|---|---  
**ε-greedy** | 確率 $\epsilon$ でランダム行動 | シンプル、調整が容易  
**Softmax** | Q値に基づく確率的選択 | 滑らかな探索  
**UCB** | 上側信頼限界を使用 | 理論的保証あり  
**Thompson Sampling** | ベイズ推定に基づく | 効率的な探索  
  
#### ε-greedy 方策

$$ a_t = \begin{cases} \arg\max_a Q(a) & \text{確率 } 1-\epsilon \\\ \text{random action} & \text{確率 } \epsilon \end{cases} $$ 

#### Softmax（ボルツマン）方策

$$ P(a) = \frac{\exp(Q(a) / \tau)}{\sum_{a'} \exp(Q(a') / \tau)} $$ ここで $\tau$ は温度パラメータ（高いほど探索的）。 

#### UCB（Upper Confidence Bound）

$$ a_t = \arg\max_a \left[Q(a) + c\sqrt{\frac{\ln t}{N(a)}}\right] $$ ここで $N(a)$ は行動 $a$ を選んだ回数、$c$ は探索度合いを制御。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== 探索と活用のトレードオフ ===\n")
    
    class MultiArmedBandit:
        """多腕バンディット問題"""
        def __init__(self, n_arms=10, seed=42):
            np.random.seed(seed)
            self.n_arms = n_arms
            # 各腕の真の期待報酬（標準正規分布から生成）
            self.true_values = np.random.randn(n_arms)
            self.optimal_arm = np.argmax(self.true_values)
    
        def pull(self, arm):
            """腕を引いて報酬を得る（期待値 + ノイズ）"""
            reward = self.true_values[arm] + np.random.randn()
            return reward
    
    class EpsilonGreedy:
        """ε-greedy アルゴリズム"""
        def __init__(self, n_arms, epsilon=0.1):
            self.n_arms = n_arms
            self.epsilon = epsilon
            self.Q = np.zeros(n_arms)  # 推定価値
            self.N = np.zeros(n_arms)  # 選択回数
    
        def select_action(self):
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_arms)  # 探索
            else:
                return np.argmax(self.Q)  # 活用
    
        def update(self, action, reward):
            self.N[action] += 1
            # 増分更新：Q(a) ← Q(a) + α[R - Q(a)]
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
    
    class UCB:
        """UCB（Upper Confidence Bound）アルゴリズム"""
        def __init__(self, n_arms, c=2.0):
            self.n_arms = n_arms
            self.c = c
            self.Q = np.zeros(n_arms)
            self.N = np.zeros(n_arms)
            self.t = 0
    
        def select_action(self):
            self.t += 1
    
            # 全ての腕を少なくとも1回は選択
            if np.min(self.N) == 0:
                return np.argmin(self.N)
    
            # UCB スコアを計算
            ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
            return np.argmax(ucb_values)
    
        def update(self, action, reward):
            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
    
    class Softmax:
        """Softmax（ボルツマン）方策"""
        def __init__(self, n_arms, tau=1.0):
            self.n_arms = n_arms
            self.tau = tau  # 温度パラメータ
            self.Q = np.zeros(n_arms)
            self.N = np.zeros(n_arms)
    
        def select_action(self):
            # Softmax 確率を計算
            exp_values = np.exp(self.Q / self.tau)
            probs = exp_values / np.sum(exp_values)
            return np.random.choice(self.n_arms, p=probs)
    
        def update(self, action, reward):
            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
    
    def run_experiment(bandit, agent, n_steps=1000):
        """実験を実行"""
        rewards = np.zeros(n_steps)
        optimal_actions = np.zeros(n_steps)
    
        for step in range(n_steps):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
    
            rewards[step] = reward
            optimal_actions[step] = (action == bandit.optimal_arm)
    
        return rewards, optimal_actions
    
    # バンディットの作成
    bandit = MultiArmedBandit(n_arms=10, seed=42)
    
    print("【真の期待報酬】")
    for i, value in enumerate(bandit.true_values):
        marker = " ← 最適" if i == bandit.optimal_arm else ""
        print(f"腕 {i}: {value:.3f}{marker}")
    print()
    
    # 複数の戦略を比較
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
        print(f"実行中: {name}")
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
    
    print("\n実験完了\n")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左：平均報酬
    axes[0].set_title('平均報酬の推移', fontsize=14, fontweight='bold')
    for name, data in results.items():
        axes[0].plot(data['rewards'], label=name, linewidth=2, alpha=0.8)
    
    axes[0].axhline(y=max(bandit.true_values), color='r', linestyle='--',
                    linewidth=2, label='最適報酬', alpha=0.5)
    axes[0].set_xlabel('ステップ数', fontsize=12)
    axes[0].set_ylabel('平均報酬', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # 右：最適行動の選択率
    axes[1].set_title('最適行動の選択率', fontsize=14, fontweight='bold')
    for name, data in results.items():
        axes[1].plot(data['optimal'], label=name, linewidth=2, alpha=0.8)
    
    axes[1].set_xlabel('ステップ数', fontsize=12)
    axes[1].set_ylabel('最適行動を選ぶ確率', fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exploration_exploitation.png', dpi=150, bbox_inches='tight')
    print("可視化を 'exploration_exploitation.png' に保存しました。")
    

* * *

## 1.6 モンテカルロ法とTD学習

### モンテカルロ法（Monte Carlo Methods）

モンテカルロ法は、エピソード全体を経験してから価値関数を更新する手法です。モデルフリーで、環境のダイナミクスを知らなくても学習できます。

#### First-Visit MC

状態 $s$ に最初に訪れたときのリターン $G_t$ を使って更新：

$$ V(s) \leftarrow V(s) + \alpha [G_t - V(s)] $$ 

#### Every-Visit MC

状態 $s$ に訪れる全てのタイミングでリターンを使用。

### TD学習（Temporal Difference Learning）

TD学習は、エピソードの終了を待たずに、1ステップごとに価値関数を更新します。

#### TD(0) アルゴリズム

$$ V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$ ここで $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ を **TD誤差** と呼びます。 

### MC vs TD の比較

特徴 | モンテカルロ法 | TD学習  
---|---|---  
**更新タイミング** | エピソード終了後 | 各ステップ  
**必要な情報** | 実際のリターン $G_t$ | 次状態の推定値 $V(S_{t+1})$  
**バイアス** | 無バイアス | バイアスあり（推定値を使用）  
**分散** | 高分散 | 低分散  
**収束速度** | 遅い | 速い  
**適用** | エピソディックタスクのみ | 継続タスクも可能  
      
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== モンテカルロ法とTD学習 ===\n")
    
    class RandomWalk:
        """
        1次元ランダムウォーク環境
    
        状態: [0, 1, 2, 3, 4, 5, 6]
        - 状態0: 左端（終端、報酬0）
        - 状態1-5: 通常状態
        - 状態6: 右端（終端、報酬+1）
        """
        def __init__(self):
            self.n_states = 7
            self.start_state = 3  # 中央からスタート
    
        def reset(self):
            return self.start_state
    
        def step(self, state):
            """ランダムに左右に移動"""
            if state == 0 or state == 6:
                return state, 0, True  # 終端状態
    
            # 50%の確率で左右に移動
            if np.random.rand() < 0.5:
                next_state = state - 1
            else:
                next_state = state + 1
    
            # 報酬と終了判定
            if next_state == 6:
                return next_state, 1.0, True
            elif next_state == 0:
                return next_state, 0.0, True
            else:
                return next_state, 0.0, False
    
    def monte_carlo_evaluation(env, n_episodes=1000, alpha=0.1):
        """
        モンテカルロ法による価値関数の推定
        """
        V = np.zeros(env.n_states)
        V[6] = 1.0  # 右端の真の価値
    
        for episode in range(n_episodes):
            # エピソードの生成
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
    
            # リターンの計算（後ろから前へ）
            G = 0
            visited = set()
    
            for t in range(len(states) - 1, -1, -1):
                G = rewards[t] + G  # γ=1 を仮定
                s = states[t]
    
                # First-Visit MC
                if s not in visited:
                    visited.add(s)
                    # 増分更新
                    V[s] = V[s] + alpha * (G - V[s])
    
        return V
    
    def td_learning(env, n_episodes=1000, alpha=0.1, gamma=1.0):
        """
        TD(0) による価値関数の推定
        """
        V = np.zeros(env.n_states)
        V[6] = 1.0  # 右端の真の価値
    
        for episode in range(n_episodes):
            state = env.reset()
    
            while True:
                next_state, reward, done = env.step(state)
    
                # TD(0) 更新
                td_target = reward + gamma * V[next_state]
                td_error = td_target - V[state]
                V[state] = V[state] + alpha * td_error
    
                if done:
                    break
    
                state = next_state
    
        return V
    
    # 真の価値関数（解析的に計算可能）
    true_values = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])
    
    # ランダムウォーク環境
    env = RandomWalk()
    
    print("【真の価値関数】")
    print("状態:", list(range(7)))
    print("価値:", [f"{v:.3f}" for v in true_values])
    print()
    
    # モンテカルロ法の実行
    print("モンテカルロ法を実行中...")
    V_mc = monte_carlo_evaluation(env, n_episodes=5000, alpha=0.01)
    
    print("【モンテカルロ法による推定】")
    print("状態:", list(range(7)))
    print("価値:", [f"{v:.3f}" for v in V_mc])
    print()
    
    # TD学習の実行
    print("TD学習を実行中...")
    V_td = td_learning(env, n_episodes=5000, alpha=0.01)
    
    print("【TD学習による推定】")
    print("状態:", list(range(7)))
    print("価値:", [f"{v:.3f}" for v in V_td])
    print()
    
    # 学習曲線の比較
    def evaluate_learning_curve(env, method, n_runs=20, episode_checkpoints=None):
        """学習曲線を評価"""
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
    
                # RMSEを計算
                rmse = np.sqrt(np.mean((V - true_values)**2))
                errors[n_ep].append(rmse)
    
        # 平均を計算
        avg_errors = {ep: np.mean(errors[ep]) for ep in episode_checkpoints}
        return avg_errors
    
    print("学習曲線を評価中（これには時間がかかります）...")
    episode_checkpoints = [0, 1, 10, 100, 500, 1000, 2000, 5000]
    mc_errors = evaluate_learning_curve(env, 'MC', n_runs=10,
                                        episode_checkpoints=episode_checkpoints)
    td_errors = evaluate_learning_curve(env, 'TD', n_runs=10,
                                        episode_checkpoints=episode_checkpoints)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左：価値関数の比較
    states = np.arange(7)
    axes[0].plot(states, true_values, 'k-', linewidth=3, marker='o',
                 markersize=8, label='真の価値', alpha=0.7)
    axes[0].plot(states, V_mc, 'b--', linewidth=2, marker='s',
                 markersize=6, label='MC (5000エピソード)', alpha=0.8)
    axes[0].plot(states, V_td, 'r-.', linewidth=2, marker='^',
                 markersize=6, label='TD (5000エピソード)', alpha=0.8)
    
    axes[0].set_xlabel('状態', fontsize=12)
    axes[0].set_ylabel('推定価値', fontsize=12)
    axes[0].set_title('ランダムウォーク：価値関数の推定', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(states)
    
    # 右：学習曲線
    episodes = list(mc_errors.keys())
    mc_rmse = [mc_errors[ep] for ep in episodes]
    td_rmse = [td_errors[ep] for ep in episodes]
    
    axes[1].plot(episodes, mc_rmse, 'b-', linewidth=2, marker='s',
                 markersize=6, label='モンテカルロ法', alpha=0.8)
    axes[1].plot(episodes, td_rmse, 'r-', linewidth=2, marker='^',
                 markersize=6, label='TD学習', alpha=0.8)
    
    axes[1].set_xlabel('エピソード数', fontsize=12)
    axes[1].set_ylabel('RMSE（平均二乗誤差）', fontsize=12)
    axes[1].set_title('学習速度の比較', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mc_vs_td.png', dpi=150, bbox_inches='tight')
    print("可視化を 'mc_vs_td.png' に保存しました。")
    

* * *

## 1.7 実践：Grid World での強化学習

### Grid World 環境の実装

ここでは、より複雑な Grid World 環境を構築し、学んだアルゴリズムを統合的に適用します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    print("=== Grid World での強化学習実践 ===\n")
    
    class GridWorldEnv:
        """
        カスタマイズ可能な Grid World 環境
    
        - 壁、ゴール、穴（落とし穴）を配置可能
        - エージェントは上下左右に移動
        - 確率的な遷移（意図した方向に進まない可能性）
        """
        def __init__(self, size=5, slip_prob=0.1):
            self.size = size
            self.slip_prob = slip_prob  # スリップ確率
    
            # グリッドの設定
            self.grid = np.zeros((size, size), dtype=int)
            # 0: 通常, 1: 壁, 2: ゴール, 3: 穴
    
            # デフォルトの環境設定
            self.grid[1, 1] = 1  # 壁
            self.grid[1, 2] = 1  # 壁
            self.grid[2, 3] = 3  # 穴
            self.grid[4, 4] = 2  # ゴール
    
            self.start_pos = (0, 0)
            self.current_pos = self.start_pos
    
            self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
            self.action_names = ['↑', '→', '↓', '←']
            self.n_actions = 4
    
        def reset(self):
            """環境をリセット"""
            self.current_pos = self.start_pos
            return self.current_pos
    
        def is_valid_pos(self, pos):
            """位置が有効かチェック"""
            x, y = pos
            if not (0 <= x < self.size and 0 <= y < self.size):
                return False
            if self.grid[x, y] == 1:  # 壁
                return False
            return True
    
        def step(self, action):
            """
            行動を実行
    
            Returns:
            --------
            next_pos : tuple
            reward : float
            done : bool
            """
            # スリップの処理
            if np.random.rand() < self.slip_prob:
                action = np.random.randint(self.n_actions)
    
            dx, dy = self.actions[action]
            next_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
    
            # 無効な移動の場合は元の位置に留まる
            if not self.is_valid_pos(next_pos):
                next_pos = self.current_pos
    
            # 報酬と終了判定
            cell_type = self.grid[next_pos]
            if cell_type == 2:  # ゴール
                reward = 10.0
                done = True
            elif cell_type == 3:  # 穴
                reward = -10.0
                done = True
            else:
                reward = -0.1  # 各ステップでの小さなペナルティ
                done = False
    
            self.current_pos = next_pos
            return next_pos, reward, done
    
        def render(self, policy=None, values=None):
            """環境を可視化"""
            fig, ax = plt.subplots(figsize=(8, 8))
    
            # グリッドの描画
            for i in range(self.size):
                for j in range(self.size):
                    cell_type = self.grid[i, j]
    
                    if cell_type == 1:  # 壁
                        color = 'gray'
                        ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                              facecolor=color))
                    elif cell_type == 2:  # ゴール
                        color = 'gold'
                        ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                              facecolor=color))
                        ax.text(j+0.5, self.size-1-i+0.5, 'GOAL',
                               ha='center', va='center', fontsize=10,
                               fontweight='bold', color='red')
                    elif cell_type == 3:  # 穴
                        color = 'black'
                        ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                              facecolor=color))
                        ax.text(j+0.5, self.size-1-i+0.5, 'HOLE',
                               ha='center', va='center', fontsize=10,
                               fontweight='bold', color='white')
                    else:  # 通常
                        # 価値関数の表示
                        if values is not None:
                            value = values[i, j]
                            norm_value = (value - values.min()) / \
                                        (values.max() - values.min() + 1e-8)
                            color = plt.cm.RdYlGn(norm_value)
                            ax.add_patch(Rectangle((j, self.size-1-i), 1, 1,
                                                  facecolor=color, alpha=0.6))
                            ax.text(j+0.5, self.size-1-i+0.7, f'{value:.1f}',
                                   ha='center', va='center', fontsize=8)
    
                        # 方策の表示
                        if policy is not None and values is not None:
                            arrow = policy[i, j]
                            ax.text(j+0.5, self.size-1-i+0.3, arrow,
                                   ha='center', va='center', fontsize=14,
                                   fontweight='bold')
    
            # スタート位置をマーク
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
        """Q学習エージェント"""
        def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
            self.env = env
            self.alpha = alpha  # 学習率
            self.gamma = gamma  # 割引率
            self.epsilon = epsilon  # 探索率
    
            # Q テーブル
            self.Q = np.zeros((env.size, env.size, env.n_actions))
    
        def select_action(self, state):
            """ε-greedy 方策で行動選択"""
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.env.n_actions)
            else:
                x, y = state
                return np.argmax(self.Q[x, y])
    
        def update(self, state, action, reward, next_state, done):
            """Q値を更新"""
            x, y = state
            nx, ny = next_state
    
            if done:
                td_target = reward
            else:
                td_target = reward + self.gamma * np.max(self.Q[nx, ny])
    
            td_error = td_target - self.Q[x, y, action]
            self.Q[x, y, action] += self.alpha * td_error
    
        def get_policy(self):
            """現在の方策を取得"""
            policy = np.zeros((self.env.size, self.env.size), dtype=object)
            values = np.zeros((self.env.size, self.env.size))
    
            for i in range(self.env.size):
                for j in range(self.env.size):
                    if self.env.grid[i, j] == 1:  # 壁
                        policy[i, j] = '■'
                        values[i, j] = 0
                    elif self.env.grid[i, j] == 2:  # ゴール
                        policy[i, j] = 'G'
                        values[i, j] = 10
                    elif self.env.grid[i, j] == 3:  # 穴
                        policy[i, j] = 'H'
                        values[i, j] = -10
                    else:
                        best_action = np.argmax(self.Q[i, j])
                        policy[i, j] = self.env.action_names[best_action]
                        values[i, j] = np.max(self.Q[i, j])
    
            return policy, values
    
        def train(self, n_episodes=1000):
            """学習を実行"""
            episode_rewards = []
            episode_lengths = []
    
            for episode in range(n_episodes):
                state = self.env.reset()
                total_reward = 0
                steps = 0
    
                while steps < 100:  # 最大ステップ数
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
                    print(f"エピソード {episode+1}: "
                          f"平均報酬={avg_reward:.2f}, 平均ステップ数={avg_length:.1f}")
    
            return episode_rewards, episode_lengths
    
    # Grid World 環境の作成
    env = GridWorldEnv(size=5, slip_prob=0.1)
    
    print("【Grid World 環境】")
    print("グリッドサイズ: 5x5")
    print("スリップ確率: 0.1")
    print("ゴール: (4, 4) → 報酬 +10")
    print("穴: (2, 3) → 報酬 -10")
    print("各ステップ: 報酬 -0.1\n")
    
    # Q学習エージェントの作成と訓練
    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    
    print("【Q学習による訓練】")
    rewards, lengths = agent.train(n_episodes=500)
    
    print("\n訓練完了\n")
    
    # 学習した方策の取得
    policy, values = agent.get_policy()
    
    print("【学習した方策】")
    print(policy)
    print()
    
    # 可視化
    fig1 = env.render(policy=policy, values=values)
    plt.title('Q学習で学習した方策と価値関数', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gridworld_qlearning.png', dpi=150, bbox_inches='tight')
    print("方策の可視化を 'gridworld_qlearning.png' に保存しました。")
    
    # 学習曲線
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左：報酬の推移
    window = 20
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed_rewards, linewidth=2)
    axes[0].set_xlabel('エピソード数', fontsize=12)
    axes[0].set_ylabel('平均報酬（移動平均）', fontsize=12)
    axes[0].set_title('報酬の推移', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 右：ステップ数の推移
    smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_lengths, linewidth=2, color='orange')
    axes[1].set_xlabel('エピソード数', fontsize=12)
    axes[1].set_ylabel('平均ステップ数（移動平均）', fontsize=12)
    axes[1].set_title('エピソード長の推移', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gridworld_learning_curves.png', dpi=150, bbox_inches='tight')
    print("学習曲線を 'gridworld_learning_curves.png' に保存しました。")
    

* * *

## まとめ

この章では、強化学習の基礎を学びました：

  * **強化学習の定義** ：試行錯誤を通じて最適な行動を学習するフレームワーク
  * **MDP** ：状態、行動、報酬、遷移確率、割引率による定式化
  * **ベルマン方程式** ：価値関数の再帰的定義と動的計画法の基礎
  * **価値反復法・方策反復法** ：最適方策を求める動的計画法アルゴリズム
  * **探索と活用** ：ε-greedy、UCB、Softmax などの戦略
  * **モンテカルロ法とTD学習** ：モデルフリーな学習手法
  * **実践** ：Grid World での Q学習の実装

次章では、より高度な強化学習アルゴリズム（SARSA、Q学習、DQN）について学びます。

演習問題

#### 問題1：MDPの理解

割引率 $\gamma$ が 0 に近い場合と 1 に近い場合で、エージェントの行動がどのように変わるか説明してください。

#### 問題2：ベルマン方程式

状態価値関数 $V^\pi(s)$ と行動価値関数 $Q^\pi(s, a)$ の関係式を導出してください。

#### 問題3：探索戦略

ε-greedy 方策において、$\epsilon = 0$ と $\epsilon = 1$ の極端な場合、エージェントはどのように振る舞いますか？

#### 問題4：MCとTD

モンテカルロ法とTD学習の違いを、バイアスと分散の観点から説明してください。

#### 問題5：価値反復法

提供されたコードを修正して、3x3のグリッドワールドで価値反復法を実行してください。

#### 問題6：ε-greedy の実装

多腕バンディット問題で、$\epsilon$ を時間とともに減衰させる ε-decay 戦略を実装してください（例：$\epsilon_t = \epsilon_0 / (1 + t)$）。
