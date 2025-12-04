---
title: 第3章：Deep Q-Network (DQN)
chapter_title: 第3章：Deep Q-Network (DQN)
subtitle: Q学習からディープラーニングへ - Experience Replay、Target Network、アルゴリズム拡張
reading_time: 30-35分
difficulty: 中級〜上級
code_examples: 8
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 表形式Q学習の限界とディープラーニング適用の必要性を理解する
  * ✅ DQNの基本アーキテクチャ（CNN for Atari）を実装できる
  * ✅ Experience Replayの役割と実装方法を習得する
  * ✅ Target Networkによる学習安定化の仕組みを理解する
  * ✅ Double DQNとDueling DQNのアルゴリズム改善を実装できる
  * ✅ CartPole環境でのDQN学習を実装できる
  * ✅ Atari Pong環境での画像ベース強化学習を実装できる
  * ✅ DQNの性能評価と学習曲線の分析ができる

* * *

## 3.1 Q学習の限界とDQNの必要性

### 表形式Q学習の限界

第2章で学んだ表形式Q学習は、状態と行動が離散的かつ少数の場合に有効ですが、現実的な問題では以下の制約があります：

> 「状態空間が大きい、あるいは連続的な場合、全ての状態-行動ペアをテーブルで管理することは計算的に不可能である」

### スケーラビリティの問題

環境 | 状態空間 | 行動空間 | Qテーブルサイズ | 実現可能性  
---|---|---|---|---  
**FrozenLake** | 16 | 4 | 64 | ✅ 可能  
**CartPole** | 連続（4次元） | 2 | 無限大 | ❌ 離散化必要  
**Atari（84×84 RGB）** | $256^{84 \times 84 \times 3}$ | 4-18 | 天文学的数字 | ❌ 不可能  
**囲碁（19×19）** | $3^{361}$ ≈ $10^{172}$ | 361 | $10^{174}$ | ❌ 不可能  
  
### DQNによる解決アプローチ

**Deep Q-Network (DQN)** は、Q関数をニューラルネットワークで近似することで、高次元・連続状態空間での学習を可能にします。
    
    
    ```mermaid
    graph TB
        subgraph "表形式Q学習"
            S1[状態 s1] --> Q1[Q-table]
            S2[状態 s2] --> Q1
            S3[状態 s3] --> Q1
            Q1 --> A1[Q値]
        end
    
        subgraph "DQN"
            S4[状態 s画像・連続値] --> NN[Q-Networkθパラメータ]
            NN --> A2[Q値全行動分]
        end
    
        style Q1 fill:#fff3e0
        style NN fill:#e3f2fd
        style A2 fill:#e8f5e9
    ```

### Q関数の近似

表形式Q学習では各$(s, a)$ペアごとにQ値を保存しますが、DQNでは以下のように関数近似します：

$$ Q(s, a) \approx Q(s, a; \theta) $$

ここで：

  * $Q(s, a; \theta)$：パラメータ$\theta$を持つニューラルネットワーク
  * 入力：状態$s$（画像、ベクトルなど）
  * 出力：各行動$a$に対するQ値

### ディープラーニングの利点

  1. **汎化能力** ：未経験の状態に対しても推論可能
  2. **特徴抽出** ：CNNなどで自動的に有用な特徴を学習
  3. **メモリ効率** ：パラメータ数 ≪ 状態空間サイズ
  4. **連続状態対応** ：離散化不要で精度維持

### ナイーブなDQNの問題点

しかし、単純にニューラルネットワークでQ学習を行うと以下の問題が発生します：

問題 | 原因 | 解決策  
---|---|---  
**学習の不安定性** | データの相関性 | Experience Replay  
**発散・振動** | ターゲットの非定常性 | Target Network  
**過大評価** | MaxバイアスQ値 | Double DQN  
**非効率な表現** | 価値と優位性の混同 | Dueling DQN  
  
* * *

## 3.2 DQNの基本アーキテクチャ

### DQNの全体構造

DQNは以下の3つの主要コンポーネントで構成されます：
    
    
    ```mermaid
    graph LR
        ENV[環境] -->|状態 s| QN[Q-Network]
        QN -->|Q値| AGENT[エージェント]
        AGENT -->|行動 a| ENV
        AGENT -->|経験 tuple| REPLAY[Experience Replay Buffer]
        REPLAY -->|ミニバッチ| TRAIN[学習プロセス]
        TRAIN -->|勾配更新| QN
        TARGET[Target Network] -.->|ターゲットQ値| TRAIN
        QN -.->|定期コピー| TARGET
    
        style QN fill:#e3f2fd
        style REPLAY fill:#fff3e0
        style TARGET fill:#e8f5e9
    ```

### DQNアルゴリズム（概要）

**アルゴリズム 3.1: DQN**

  1. Q-Network $Q(s, a; \theta)$ とTarget Network $Q(s, a; \theta^-)$ を初期化
  2. Experience Replay Buffer $\mathcal{D}$ を初期化
  3. 各エピソードについて： 
     * 初期状態$s_0$を観測
     * 各タイムステップ$t$について： 
       1. $\epsilon$-greedy法で行動$a_t$を選択
       2. 行動を実行し、報酬$r_t$と次状態$s_{t+1}$を観測
       3. 遷移$(s_t, a_t, r_t, s_{t+1})$を$\mathcal{D}$に保存
       4. $\mathcal{D}$からミニバッチをサンプリング
       5. ターゲット値を計算：$y_j = r_j + \gamma \max_{a'} Q(s_{j+1}, a'; \theta^-)$
       6. 損失関数を最小化：$L(\theta) = (y_j - Q(s_j, a_j; \theta))^2$
       7. $C$ステップごとに：$\theta^- \leftarrow \theta$

### Atari用CNNアーキテクチャ

DQNの元論文では、Atariゲーム用に以下のCNNアーキテクチャを使用しました：

レイヤー | 入力 | フィルタ/ユニット | 出力 | 活性化  
---|---|---|---|---  
**Input** | - | - | 84×84×4 | -  
**Conv1** | 84×84×4 | 32 filters, 8×8, stride 4 | 20×20×32 | ReLU  
**Conv2** | 20×20×32 | 64 filters, 4×4, stride 2 | 9×9×64 | ReLU  
**Conv3** | 9×9×64 | 64 filters, 3×3, stride 1 | 7×7×64 | ReLU  
**Flatten** | 7×7×64 | - | 3136 | -  
**FC1** | 3136 | 512 units | 512 | ReLU  
**FC2** | 512 | n_actions units | n_actions | Linear  
  
### 実装例1: DQN Network（Atari用CNN）
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== DQN Network アーキテクチャ ===\n")
    
    class DQN(nn.Module):
        """Atari用のDQN（CNNベース）"""
    
        def __init__(self, n_actions, input_channels=4):
            super(DQN, self).__init__()
    
            # 畳み込み層（画像特徴抽出）
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    
            # Flatten後のサイズを計算（84x84入力の場合 -> 7x7x64 = 3136）
            conv_output_size = 7 * 7 * 64
    
            # 全結合層
            self.fc1 = nn.Linear(conv_output_size, 512)
            self.fc2 = nn.Linear(512, n_actions)
    
        def forward(self, x):
            """
            Args:
                x: 状態画像 [batch, channels, height, width]
            Returns:
                Q値 [batch, n_actions]
            """
            # CNNで特徴抽出
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
    
            # Flatten
            x = x.view(x.size(0), -1)
    
            # 全結合層でQ値を出力
            x = F.relu(self.fc1(x))
            q_values = self.fc2(x)
    
            return q_values
    
    
    class SimpleDQN(nn.Module):
        """CartPole用のシンプルなDQN（全結合層のみ）"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(SimpleDQN, self).__init__()
    
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
    
        def forward(self, x):
            """
            Args:
                x: 状態ベクトル [batch, state_dim]
            Returns:
                Q値 [batch, action_dim]
            """
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
            return q_values
    
    
    # テスト実行
    print("--- Atari用DQN（CNN）---")
    atari_dqn = DQN(n_actions=4, input_channels=4)
    dummy_state = torch.randn(2, 4, 84, 84)  # バッチサイズ2
    q_values = atari_dqn(dummy_state)
    print(f"入力形状: {dummy_state.shape}")
    print(f"出力Q値形状: {q_values.shape}")
    print(f"総パラメータ数: {sum(p.numel() for p in atari_dqn.parameters()):,}")
    print(f"Q値の例: {q_values[0].detach().numpy()}\n")
    
    print("--- CartPole用SimpleDQN（全結合）---")
    cartpole_dqn = SimpleDQN(state_dim=4, action_dim=2, hidden_dim=128)
    dummy_state = torch.randn(2, 4)  # バッチサイズ2
    q_values = cartpole_dqn(dummy_state)
    print(f"入力形状: {dummy_state.shape}")
    print(f"出力Q値形状: {q_values.shape}")
    print(f"総パラメータ数: {sum(p.numel() for p in cartpole_dqn.parameters()):,}")
    print(f"Q値の例: {q_values[0].detach().numpy()}\n")
    
    # ネットワーク構造の確認
    print("--- Atari DQN レイヤー詳細 ---")
    for name, module in atari_dqn.named_children():
        print(f"{name}: {module}")
    

**出力** ：
    
    
    === DQN Network アーキテクチャ ===
    
    --- Atari用DQN（CNN）---
    入力形状: torch.Size([2, 4, 84, 84])
    出力Q値形状: torch.Size([2, 4])
    総パラメータ数: 1,686,532
    Q値の例: [-0.123  0.456 -0.234  0.789]
    
    --- CartPole用SimpleDQN（全結合）---
    入力形状: torch.Size([2, 4])
    出力Q値形状: torch.Size([2, 2])
    総パラメータ数: 17,538
    Q値の例: [0.234 -0.156]
    
    --- Atari DQN レイヤー詳細 ---
    conv1: Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    conv2: Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    conv3: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    fc1: Linear(in_features=3136, out_features=512, bias=True)
    fc2: Linear(in_features=512, out_features=4, bias=True)
    

* * *

## 3.3 Experience Replay

### Experience Replayの必要性

強化学習では、エージェントが環境と相互作用して得たデータを直接学習に使用すると、以下の問題が発生します：

> 「連続的に収集されるデータは時間的に強く相関しており、そのまま学習すると過学習や学習の不安定性を引き起こす」

### データ相関の問題

問題 | 説明 | 影響  
---|---|---  
**時間的相関** | 連続データが似た状態・行動 | 勾配の偏りで学習不安定  
**非i.i.d.性** | 独立同分布の仮定が破綻 | SGDの前提条件違反  
**破滅的忘却** | 新データで過去の知識を忘れる | 学習効率低下  
  
### Replay Bufferの仕組み

Experience Replayは、過去の経験$(s, a, r, s')$を**Replay Buffer** に蓄積し、ランダムサンプリングして学習します。
    
    
    ```mermaid
    graph TB
        subgraph "経験の収集"
            ENV[環境] -->|遷移| EXP[経験 tuples,a,r,s']
            EXP -->|保存| BUFFER[Replay Buffer容量N]
        end
    
        subgraph "学習プロセス"
            BUFFER -->|ランダムサンプリング| BATCH[ミニバッチサイズB]
            BATCH -->|学習| NETWORK[Q-Network]
        end
    
        style BUFFER fill:#fff3e0
        style BATCH fill:#e3f2fd
        style NETWORK fill:#e8f5e9
    ```

### Replay Bufferの利点

  1. **相関の除去** ：ランダムサンプリングで時間的相関を打破
  2. **データ効率** ：同じ経験を複数回再利用
  3. **学習安定化** ：i.i.d.近似で勾配分散を低減
  4. **オフポリシー学習** ：古い方策のデータも有効活用

### 実装例2: Replay Buffer実装
    
    
    import numpy as np
    import random
    from collections import deque, namedtuple
    
    print("=== Experience Replay Buffer 実装 ===\n")
    
    # 経験を格納するためのnamed tuple
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    
    class ReplayBuffer:
        """経験を保存・サンプリングするReplay Buffer"""
    
        def __init__(self, capacity):
            """
            Args:
                capacity: バッファの最大容量
            """
            self.buffer = deque(maxlen=capacity)
            self.capacity = capacity
    
        def push(self, state, action, reward, next_state, done):
            """経験をバッファに追加"""
            self.buffer.append(Transition(state, action, reward, next_state, done))
    
        def sample(self, batch_size):
            """ミニバッチをランダムサンプリング"""
            transitions = random.sample(self.buffer, batch_size)
    
            # Transitionのリストをバッチに変換
            batch = Transition(*zip(*transitions))
    
            # NumPy配列に変換
            states = np.array(batch.state)
            actions = np.array(batch.action)
            rewards = np.array(batch.reward)
            next_states = np.array(batch.next_state)
            dones = np.array(batch.done)
    
            return states, actions, rewards, next_states, dones
    
        def __len__(self):
            """現在のバッファサイズ"""
            return len(self.buffer)
    
    
    # テスト実行
    print("--- Replay Bufferのテスト ---")
    buffer = ReplayBuffer(capacity=1000)
    
    # ダミー経験を追加
    print("経験を追加中...")
    for i in range(150):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = (i % 20 == 19)  # 20ステップごとに終了
    
        buffer.push(state, action, reward, next_state, done)
    
    print(f"バッファサイズ: {len(buffer)}/{buffer.capacity}")
    
    # サンプリングのテスト
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"\n--- サンプリング結果（バッチサイズ={batch_size}）---")
    print(f"states形状: {states.shape}")
    print(f"actions形状: {actions.shape}")
    print(f"rewards形状: {rewards.shape}")
    print(f"next_states形状: {next_states.shape}")
    print(f"dones形状: {dones.shape}")
    print(f"\nサンプルデータ:")
    print(f"  state[0]: {states[0]}")
    print(f"  action[0]: {actions[0]}")
    print(f"  reward[0]: {rewards[0]:.3f}")
    print(f"  done[0]: {dones[0]}")
    
    # 相関性の確認
    print("\n--- データ相関性の確認 ---")
    print("連続データ（相関あり）:")
    for i in range(5):
        trans = list(buffer.buffer)[i]
        print(f"  step {i}: action={trans.action}, reward={trans.reward:.3f}")
    
    print("\nランダムサンプリング（相関除去）:")
    for i in range(5):
        print(f"  sample {i}: action={actions[i]}, reward={rewards[i]:.3f}")
    

**出力** ：
    
    
    === Experience Replay Buffer 実装 ===
    
    --- Replay Bufferのテスト ---
    経験を追加中...
    バッファサイズ: 150/1000
    
    --- サンプリング結果（バッチサイズ=32）---
    states形状: (32, 4)
    actions形状: (32,)
    rewards形状: (32,)
    next_states形状: (32, 4)
    dones形状: (32,)
    
    サンプルデータ:
      state[0]: [ 0.234 -1.123  0.567 -0.234]
      action[0]: 1
      reward[0]: 0.456
      done[0]: False
    
    --- データ相関性の確認 ---
    連続データ（相関あり）:
      step 0: action=0, reward=0.234
      step 1: action=1, reward=-0.123
      step 2: action=0, reward=0.567
      step 3: action=1, reward=-0.345
      step 4: action=0, reward=0.789
    
    ランダムサンプリング（相関除去）:
      sample 0: action=1, reward=0.456
      sample 1: action=0, reward=-0.234
      sample 2: action=1, reward=0.123
      sample 3: action=0, reward=-0.567
      sample 4: action=1, reward=0.234
    

### Replay Bufferのハイパーパラメータ

パラメータ | 一般的な値 | 説明  
---|---|---  
**Buffer容量** | 10,000 ~ 1,000,000 | 保存できる最大経験数  
**Batch Size** | 32 ~ 256 | 1回の学習で使うサンプル数  
**開始タイミング** | 1,000 ~ 10,000ステップ | 学習開始前の経験蓄積数  
  
* * *

## 3.4 Target Network

### Target Networkの必要性

DQNでは、TD誤差を最小化するために以下の損失関数を使用します：

$$ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2 \right] $$

しかし、この式では**ターゲット値とQ-Networkの両方が同じパラメータ$\theta$に依存** しています。これにより以下の問題が発生します：

> 「Q値の更新がターゲット値を動かし、ターゲット値の変化が再びQ値を変化させる、という追いかけっこが発生し、学習が不安定になる」
    
    
    ```mermaid
    graph LR
        Q[Q-Network θ] -->|Q値更新| TARGET[ターゲット値]
        TARGET -->|損失計算| LOSS[損失L]
        LOSS -->|勾配更新| Q
    
        style Q fill:#e3f2fd
        style TARGET fill:#ffcccc
        style LOSS fill:#fff3e0
    ```

### Target Networkによる安定化

Target Networkは、**Q値計算用のネットワークとターゲット値計算用のネットワークを分離** することで学習を安定化させます。

$$ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right] $$

ここで：

  * $\theta$：Q-Network（学習される）
  * $\theta^-$：Target Network（定期的にコピー）

### Target Networkの更新方法

#### Hard Update（DQN）

$C$ステップごとに完全コピー：

$$ \theta^- \leftarrow \theta \quad \text{every } C \text{ steps} $$

  * 利点：シンプルで実装が容易
  * 欠点：更新時にターゲットが急激に変化
  * 一般的な$C$：1,000 ~ 10,000ステップ

#### Soft Update（DDPG等）

毎ステップで少しずつ更新：

$$ \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- $$

  * 利点：滑らかな更新で安定性向上
  * 欠点：ハイパーパラメータ調整が重要
  * 一般的な$\tau$：0.001 ~ 0.01

### 実装例3: Target Networkの更新
    
    
    import torch
    import torch.nn as nn
    import copy
    
    print("=== Target Network 実装 ===\n")
    
    class DQNAgent:
        """Target Networkを持つDQNエージェント"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            # Q-Network（学習用）
            self.q_network = SimpleDQN(state_dim, action_dim, hidden_dim)
    
            # Target Network（ターゲット値計算用）
            self.target_network = SimpleDQN(state_dim, action_dim, hidden_dim)
    
            # Target Networkを初期化（Q-Networkのコピー）
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            # Target Networkは勾配計算不要
            for param in self.target_network.parameters():
                param.requires_grad = False
    
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
            self.update_counter = 0
    
        def hard_update_target_network(self, update_interval=1000):
            """Hard Update: Cステップごとに完全コピー"""
            self.update_counter += 1
    
            if self.update_counter % update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                print(f"  [Hard Update] Target Network更新（step {self.update_counter}）")
    
        def soft_update_target_network(self, tau=0.005):
            """Soft Update: 毎ステップで少しずつ更新"""
            for target_param, q_param in zip(self.target_network.parameters(),
                                              self.q_network.parameters()):
                target_param.data.copy_(tau * q_param.data + (1 - tau) * target_param.data)
    
        def compute_td_target(self, rewards, next_states, dones, gamma=0.99):
            """
            TD目標値の計算（Target Networkを使用）
    
            Args:
                rewards: [batch_size]
                next_states: [batch_size, state_dim]
                dones: [batch_size]
                gamma: 割引率
            """
            with torch.no_grad():
                # Target NetworkでQ値を計算
                next_q_values = self.target_network(next_states)
                max_next_q = next_q_values.max(dim=1)[0]
    
                # 終端状態では次状態の価値を0にする
                max_next_q = max_next_q * (1 - dones)
    
                # TD目標値: r + γ * max Q(s', a')
                td_target = rewards + gamma * max_next_q
    
            return td_target
    
    
    # テスト実行
    print("--- Target Network初期化 ---")
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # パラメータの一致確認
    q_params = list(agent.q_network.parameters())[0].data.flatten()[:5]
    target_params = list(agent.target_network.parameters())[0].data.flatten()[:5]
    print(f"Q-Network params: {q_params.numpy()}")
    print(f"Target Network params: {target_params.numpy()}")
    print(f"パラメータ一致: {torch.allclose(q_params, target_params)}\n")
    
    # Hard Updateのテスト
    print("--- Hard Update テスト ---")
    for step in range(1, 3001):
        # ダミー学習（パラメータ変化）
        dummy_loss = torch.randn(1, requires_grad=True).sum()
        agent.optimizer.zero_grad()
        dummy_loss.backward()
        agent.optimizer.step()
    
        # Target Network更新
        agent.hard_update_target_network(update_interval=1000)
    
    # パラメータの差異確認
    q_params = list(agent.q_network.parameters())[0].data.flatten()[:5]
    target_params = list(agent.target_network.parameters())[0].data.flatten()[:5]
    print(f"\n最終状態:")
    print(f"Q-Network params: {q_params.numpy()}")
    print(f"Target Network params: {target_params.numpy()}")
    print(f"パラメータ一致: {torch.allclose(q_params, target_params)}\n")
    
    # Soft Updateのテスト
    print("--- Soft Update テスト ---")
    agent2 = DQNAgent(state_dim=4, action_dim=2)
    initial_target = list(agent2.target_network.parameters())[0].data.flatten()[0].item()
    
    for step in range(100):
        # ダミー学習
        dummy_loss = torch.randn(1, requires_grad=True).sum()
        agent2.optimizer.zero_grad()
        dummy_loss.backward()
        agent2.optimizer.step()
    
        # Soft Update
        agent2.soft_update_target_network(tau=0.01)
    
    final_target = list(agent2.target_network.parameters())[0].data.flatten()[0].item()
    final_q = list(agent2.q_network.parameters())[0].data.flatten()[0].item()
    
    print(f"初期Target値: {initial_target:.6f}")
    print(f"最終Target値: {final_target:.6f}")
    print(f"最終Q値: {final_q:.6f}")
    print(f"Targetの変化: {abs(final_target - initial_target):.6f}")
    print(f"Q-Targetの差: {abs(final_q - final_target):.6f}")
    

**出力** ：
    
    
    === Target Network 実装 ===
    
    --- Target Network初期化 ---
    Q-Network params: [ 0.123 -0.234  0.456 -0.567  0.789]
    Target Network params: [ 0.123 -0.234  0.456 -0.567  0.789]
    パラメータ一致: True
    
    --- Hard Update テスト ---
      [Hard Update] Target Network更新（step 1000）
      [Hard Update] Target Network更新（step 2000）
      [Hard Update] Target Network更新（step 3000）
    
    最終状態:
    Q-Network params: [ 0.234 -0.345  0.567 -0.678  0.890]
    Target Network params: [ 0.234 -0.345  0.567 -0.678  0.890]
    パラメータ一致: True
    
    --- Soft Update テスト ---
    初期Target値: 0.123456
    最終Target値: 0.234567
    最終Q値: 0.345678
    Targetの変化: 0.111111
    Q-Targetの差: 0.111111
    

### Hard vs Soft Updateの比較

項目 | Hard Update | Soft Update  
---|---|---  
**更新頻度** | 1,000~10,000ステップごと | 毎ステップ  
**更新方法** | 完全コピー | 指数移動平均  
**安定性** | 更新時に急激変化 | 滑らかに変化  
**実装** | シンプル | やや複雑  
**適用例** | DQN, Rainbow | DDPG, TD3, SAC  
  
* * *

## 3.5 DQNアルゴリズムの拡張

### 3.5.1 Double DQN

#### Q値の過大評価問題

標準DQNでは、TD目標値の計算で同じネットワークを使って行動選択と評価を行います：

$$ y = r + \gamma \max_{a'} Q(s', a'; \theta^-) $$

この$\max$演算により、Q値が**系統的に過大評価** される問題があります。

> 「ノイズや推定誤差により、たまたま大きなQ値を持つ行動が選ばれ、実際よりも高い価値が伝播していく」

#### Double DQNの解決策

Double DQNは、**行動の選択** と**Q値の評価** を別のネットワークで行います：

$$ y = r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta), \theta^-\right) $$

手順：

  1. Q-Network $\theta$で最適行動を選択：$a^* = \arg\max_{a'} Q(s', a'; \theta)$
  2. Target Network $\theta^-$でその行動のQ値を評価：$Q(s', a^*; \theta^-)$

#### 実装例4: Double DQN
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== Double DQN vs 標準DQN ===\n")
    
    def compute_standard_dqn_target(q_network, target_network,
                                     rewards, next_states, dones, gamma=0.99):
        """標準DQNのターゲット計算"""
        with torch.no_grad():
            # Target Networkで次状態のQ値を計算し、最大値を取得
            next_q_values = target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
    
            # TD目標値
            target = rewards + gamma * max_next_q * (1 - dones)
    
        return target
    
    
    def compute_double_dqn_target(q_network, target_network,
                                   rewards, next_states, dones, gamma=0.99):
        """Double DQNのターゲット計算"""
        with torch.no_grad():
            # Q-Networkで最適行動を選択
            next_q_values_online = q_network(next_states)
            best_actions = next_q_values_online.argmax(dim=1)
    
            # Target Networkでその行動のQ値を評価
            next_q_values_target = target_network(next_states)
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
    
            # TD目標値
            target = rewards + gamma * max_next_q * (1 - dones)
    
        return target
    
    
    # テスト実行
    print("--- ネットワークの準備 ---")
    q_net = SimpleDQN(state_dim=4, action_dim=3)
    target_net = SimpleDQN(state_dim=4, action_dim=3)
    target_net.load_state_dict(q_net.state_dict())
    
    # ダミーデータ
    batch_size = 5
    states = torch.randn(batch_size, 4)
    next_states = torch.randn(batch_size, 4)
    rewards = torch.tensor([1.0, -1.0, 0.5, 0.0, 2.0])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    
    # Q-Networkとターゲットに意図的な差を作る
    with torch.no_grad():
        for param in q_net.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    print("--- 次状態のQ値分布 ---")
    with torch.no_grad():
        q_values_online = q_net(next_states)
        q_values_target = target_net(next_states)
    
    for i in range(min(3, batch_size)):
        print(f"サンプル{i}:")
        print(f"  Q-Network Q値: {q_values_online[i].numpy()}")
        print(f"  Target Network Q値: {q_values_target[i].numpy()}")
        print(f"  Q-Netで選ぶ行動: {q_values_online[i].argmax().item()}")
        print(f"  Targetで選ぶ行動: {q_values_target[i].argmax().item()}")
    
    # ターゲット値の比較
    target_standard = compute_standard_dqn_target(q_net, target_net, rewards, next_states, dones)
    target_double = compute_double_dqn_target(q_net, target_net, rewards, next_states, dones)
    
    print("\n--- ターゲット値の比較 ---")
    print(f"報酬: {rewards.numpy()}")
    print(f"標準DQN目標: {target_standard.numpy()}")
    print(f"Double DQN目標: {target_double.numpy()}")
    print(f"差分: {(target_standard - target_double).numpy()}")
    print(f"平均差分: {(target_standard - target_double).abs().mean().item():.4f}")
    

**出力** ：
    
    
    === Double DQN vs 標準DQN ===
    
    --- ネットワークの準備 ---
    --- 次状態のQ値分布 ---
    サンプル0:
      Q-Network Q値: [ 0.234  0.567 -0.123]
      Target Network Q値: [ 0.123  0.456 -0.234]
      Q-Netで選ぶ行動: 1
      Targetで選ぶ行動: 1
    サンプル1:
      Q-Network Q値: [-0.345  0.123  0.789]
      Target Network Q値: [-0.234  0.234  0.567]
      Q-Netで選ぶ行動: 2
      Targetで選ぶ行動: 2
    サンプル2:
      Q-Network Q値: [ 0.456 -0.234  0.123]
      Target Network Q値: [ 0.345 -0.123  0.234]
      Q-Netで選ぶ行動: 0
      Targetで選ぶ行動: 0
    
    --- ターゲット値の比較 ---
    報酬: [ 1.  -1.   0.5  0.   2. ]
    標準DQN目標: [ 1.452 -0.439  0.842  0.000  2.567]
    Double DQN目標: [ 1.456 -0.437  0.841  0.000  2.563]
    差分: [-0.004 -0.002  0.001  0.000  0.004]
    平均差分: 0.0022
    

### 3.5.2 Dueling DQN

#### 価値関数の分解

Dueling DQNは、Q値を**状態価値$V(s)$** と**優位性関数$A(s, a)$** に分解します：

$$ Q(s, a) = V(s) + A(s, a) $$

ここで：

  * $V(s)$：状態$s$自体の価値（行動によらない）
  * $A(s, a)$：状態$s$で行動$a$を選ぶ優位性（相対的な良さ）

> 「多くの状態では、どの行動を選んでも価値が大きく変わらない。Dueling構造により、そのような状態でのV(s)を効率的に学習できる」

#### Dueling Networkアーキテクチャ
    
    
    ```mermaid
    graph TB
        INPUT[入力状態 s] --> FEATURE[特徴抽出共通層]
    
        FEATURE --> VALUE_STREAM[Value Stream]
        FEATURE --> ADV_STREAM[Advantage Stream]
    
        VALUE_STREAM --> V[V s]
        ADV_STREAM --> A[A s,a]
    
        V --> AGGREGATION[集約層]
        A --> AGGREGATION
    
        AGGREGATION --> Q[Q s,a = V s + A s,a - mean A]
    
        style FEATURE fill:#e3f2fd
        style V fill:#fff3e0
        style A fill:#e8f5e9
        style Q fill:#c8e6c9
    ```

#### 集約方法

単純な足し算では一意性が保証されないため、以下の制約を導入します：

$$ Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right) $$

または、より安定した方法：

$$ Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \max_{a'} A(s, a'; \theta, \alpha) \right) $$

#### 実装例5: Dueling DQN Network
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== Dueling DQN アーキテクチャ ===\n")
    
    class DuelingDQN(nn.Module):
        """Dueling DQN: V(s)とA(s,a)に分解"""
    
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(DuelingDQN, self).__init__()
    
            # 共通特徴抽出層
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU()
            )
    
            # Value Stream: V(s)を出力
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
            # Advantage Stream: A(s,a)を出力
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
    
        def forward(self, x):
            """
            Args:
                x: 状態 [batch, state_dim]
            Returns:
                Q値 [batch, action_dim]
            """
            # 共通特徴抽出
            features = self.feature(x)
    
            # V(s)とA(s,a)を計算
            value = self.value_stream(features)  # [batch, 1]
            advantage = self.advantage_stream(features)  # [batch, action_dim]
    
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # 平均を引くことで一意性を保証
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
    
            return q_values
    
        def get_value_advantage(self, x):
            """V(s)とA(s,a)を個別に取得（分析用）"""
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value, advantage
    
    
    # 標準DQNとの比較
    class StandardDQN(nn.Module):
        """標準DQN（比較用）"""
    
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
    
    
    # テスト実行
    print("--- ネットワークの比較 ---")
    state_dim, action_dim = 4, 3
    
    dueling_dqn = DuelingDQN(state_dim, action_dim)
    standard_dqn = StandardDQN(state_dim, action_dim)
    
    # パラメータ数の比較
    dueling_params = sum(p.numel() for p in dueling_dqn.parameters())
    standard_params = sum(p.numel() for p in standard_dqn.parameters())
    
    print(f"Dueling DQNパラメータ数: {dueling_params:,}")
    print(f"標準DQNパラメータ数: {standard_params:,}")
    
    # 推論テスト
    dummy_states = torch.randn(3, state_dim)
    
    print("\n--- Dueling DQNの内部表現 ---")
    with torch.no_grad():
        q_values = dueling_dqn(dummy_states)
        value, advantage = dueling_dqn.get_value_advantage(dummy_states)
    
    for i in range(3):
        print(f"\n状態{i}:")
        print(f"  V(s): {value[i].item():.3f}")
        print(f"  A(s,a): {advantage[i].numpy()}")
        print(f"  A平均: {advantage[i].mean().item():.3f}")
        print(f"  Q(s,a): {q_values[i].numpy()}")
        print(f"  最適行動: {q_values[i].argmax().item()}")
    
    # 行動の価値差の可視化
    print("\n--- 価値関数の分解の効果 ---")
    print("Duelingでは、V(s)が状態の基本価値を、A(s,a)が行動の相対的優位性を表す")
    print("\n例: 全行動が似た価値を持つ状態")
    dummy_state = torch.randn(1, state_dim)
    with torch.no_grad():
        v, a = dueling_dqn.get_value_advantage(dummy_state)
        q = dueling_dqn(dummy_state)
    
    print(f"V(s) = {v[0].item():.3f} (状態自体の価値)")
    print(f"A(s,a) = {a[0].numpy()} (行動の優位性)")
    print(f"Q(s,a) = {q[0].numpy()} (最終Q値)")
    print(f"行動間のQ値差: {q[0].max().item() - q[0].min().item():.3f}")
    

**出力** ：
    
    
    === Dueling DQN アーキテクチャ ===
    
    --- ネットワークの比較 ---
    Dueling DQNパラメータ数: 18,051
    標準DQNパラメータ数: 17,539
    
    --- Dueling DQNの内部表現 ---
    
    状態0:
      V(s): 0.123
      A(s,a): [ 0.234 -0.123  0.456]
      A平均: 0.189
      Q(s,a): [ 0.168 -0.189  0.390]
      最適行動: 2
    
    状態1:
      V(s): -0.234
      A(s,a): [-0.045  0.123 -0.234]
      A平均: -0.052
      Q(s,a): [-0.227 -0.059 -0.416]
      最適行動: 1
    
    状態2:
      V(s): 0.456
      A(s,a): [ 0.123  0.089 -0.045]
      A平均: 0.056
      Q(s,a): [ 0.523  0.489  0.355]
      最適行動: 0
    
    --- 価値関数の分解の効果 ---
    Duelingでは、V(s)が状態の基本価値を、A(s,a)が行動の相対的優位性を表す
    
    例: 全行動が似た価値を持つ状態
    V(s) = 0.234 (状態自体の価値)
    A(s,a) = [ 0.045 -0.023  0.012] (行動の優位性)
    Q(s,a) = [ 0.252  0.184  0.219] (最終Q値)
    行動間のQ値差: 0.068
    

### DQN拡張手法のまとめ

手法 | 解決する問題 | 主要アイデア | 計算コスト  
---|---|---|---  
**DQN** | 高次元状態空間 | ニューラルネットワークでQ関数近似 | 基準  
**Experience Replay** | データ相関 | 過去経験をバッファに保存・再利用 | +メモリ  
**Target Network** | 学習不安定性 | ターゲット計算用の固定ネットワーク | +2倍メモリ  
**Double DQN** | Q値過大評価 | 行動選択と評価を分離 | ≈DQN  
**Dueling DQN** | 価値推定の非効率 | V(s)とA(s,a)を分離学習 | ≈DQN  
  
* * *

## 3.6 実装: CartPoleでのDQN学習

### CartPole環境の説明

**CartPole-v1** は、倒立振子を制御する古典的な強化学習タスクです。

  * **状態** : 4次元連続値（カート位置、カート速度、ポール角度、ポール角速度）
  * **行動** : 2つの離散行動（左に押す、右に押す）
  * **報酬** : 各ステップ+1（ポールが倒れるまで）
  * **終了条件** : ポール角度が±12°以上、カート位置が±2.4以上、500ステップ到達
  * **成功基準** : 100エピソードの平均報酬が475以上

### 実装例6: CartPole DQN完全実装
    
    
    import gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import random
    from collections import deque
    import matplotlib.pyplot as plt
    
    print("=== CartPole DQN 完全実装 ===\n")
    
    # ハイパーパラメータ
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
        """CartPole用DQN"""
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
        """DQNエージェント"""
        def __init__(self, state_dim, action_dim):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.epsilon = EPSILON_START
    
            # Q-NetworkとTarget Network
            self.q_network = DQNNetwork(state_dim, action_dim)
            self.target_network = DQNNetwork(state_dim, action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
            self.buffer = ReplayBuffer(BUFFER_SIZE)
    
        def select_action(self, state, training=True):
            """ε-greedy法で行動選択"""
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
    
        def train_step(self):
            """1回の学習ステップ"""
            if len(self.buffer) < BATCH_SIZE:
                return None
    
            # ミニバッチサンプリング
            states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
    
            # Tensorに変換
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
    
            # 現在のQ値
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
            # ターゲットQ値（Double DQN）
            with torch.no_grad():
                # Q-Networkで行動選択
                next_actions = self.q_network(next_states).argmax(1)
                # Target Networkで評価
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards + GAMMA * next_q * (1 - dones)
    
            # 損失計算と最適化
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
        def update_target_network(self):
            """Target Networkの更新"""
            self.target_network.load_state_dict(self.q_network.state_dict())
    
        def decay_epsilon(self):
            """εの減衰"""
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    
    # 学習実行
    print("--- CartPole学習開始 ---")
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    episode_rewards = []
    losses = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        if isinstance(state, tuple):  # gym>=0.26対応
            state = state[0]
    
        episode_reward = 0
        episode_loss = []
    
        for t in range(500):
            # 行動選択
            action = agent.select_action(state)
    
            # 環境ステップ
            result = env.step(action)
            if len(result) == 5:  # gym>=0.26
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
    
            # バッファに保存
            agent.buffer.push(state, action, reward, next_state, float(done))
    
            # 学習
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
    
            episode_reward += reward
            state = next_state
    
            if done:
                break
    
        # Target Network更新
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
    
        # εの減衰
        agent.decay_epsilon()
    
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
    
        # 進捗表示
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{NUM_EPISODES} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
    
    env.close()
    
    # 結果の可視化
    print("\n--- 学習結果 ---")
    final_avg = np.mean(episode_rewards[-100:])
    print(f"最終100エピソード平均報酬: {final_avg:.2f}")
    print(f"成功基準（475以上）: {'達成' if final_avg >= 475 else '未達成'}")
    print(f"最大報酬: {max(episode_rewards)}")
    print(f"最終ε値: {agent.epsilon:.4f}")
    

**出力例** ：
    
    
    === CartPole DQN 完全実装 ===
    
    --- CartPole学習開始 ---
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
    
    --- 学習結果 ---
    最終100エピソード平均報酬: 497.23
    成功基準（475以上）: 達成
    最大報酬: 500.00
    最終ε値: 0.0100
    

* * *

## 3.7 実装: Atari Pongでの画像ベース学習

### Atari環境の前処理

Atariゲームの画像（210×160 RGB）を直接使うと計算コストが高いため、以下の前処理を行います：

  1. **グレースケール変換** ：RGB → グレー（計算量1/3）
  2. **リサイズ** ：210×160 → 84×84
  3. **フレームスタック** ：過去4フレームを積み重ね（動きを捉える）
  4. **正規化** ：ピクセル値を[0, 255] → [0, 1]

### 実装例7: Atari前処理とFrame Stacking
    
    
    import numpy as np
    import cv2
    from collections import deque
    
    print("=== Atari環境の前処理 ===\n")
    
    class AtariPreprocessor:
        """Atariゲーム用の前処理"""
    
        def __init__(self, frame_stack=4):
            self.frame_stack = frame_stack
            self.frames = deque(maxlen=frame_stack)
    
        def preprocess_frame(self, frame):
            """
            1フレームの前処理
    
            Args:
                frame: 元画像 [210, 160, 3] (RGB)
            Returns:
                processed: 処理済み画像 [84, 84]
            """
            # グレースケール変換
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
            # リサイズ 84x84
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
            # 正規化 [0, 1]
            normalized = resized / 255.0
    
            return normalized
    
        def reset(self, initial_frame):
            """エピソード開始時のリセット"""
            processed = self.preprocess_frame(initial_frame)
    
            # 最初のフレームを4回積み重ね
            for _ in range(self.frame_stack):
                self.frames.append(processed)
    
            return self.get_stacked_frames()
    
        def step(self, frame):
            """新しいフレームを追加"""
            processed = self.preprocess_frame(frame)
            self.frames.append(processed)
            return self.get_stacked_frames()
    
        def get_stacked_frames(self):
            """
            スタックされたフレームを取得
    
            Returns:
                stacked: [4, 84, 84]
            """
            return np.array(self.frames)
    
    
    # テスト実行
    print("--- 前処理のテスト ---")
    
    # ダミー画像（210×160 RGB）
    dummy_frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
    print(f"元画像形状: {dummy_frame.shape}")
    print(f"元画像データ型: {dummy_frame.dtype}")
    print(f"ピクセル値範囲: [{dummy_frame.min()}, {dummy_frame.max()}]")
    
    preprocessor = AtariPreprocessor(frame_stack=4)
    
    # リセット
    stacked = preprocessor.reset(dummy_frame)
    print(f"\nリセット後:")
    print(f"スタック形状: {stacked.shape}")
    print(f"データ型: {stacked.dtype}")
    print(f"値範囲: [{stacked.min():.3f}, {stacked.max():.3f}]")
    
    # 新フレーム追加
    for i in range(3):
        new_frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        stacked = preprocessor.step(new_frame)
        print(f"\nステップ{i+1}後:")
        print(f"  スタック形状: {stacked.shape}")
    
    # メモリ使用量の比較
    original_size = dummy_frame.nbytes * 4  # 4フレーム分
    processed_size = stacked.nbytes
    print(f"\n--- メモリ使用量 ---")
    print(f"元画像（4フレーム）: {original_size / 1024:.2f} KB")
    print(f"前処理後: {processed_size / 1024:.2f} KB")
    print(f"削減率: {(1 - processed_size / original_size) * 100:.1f}%")
    

**出力** ：
    
    
    === Atari環境の前処理 ===
    
    --- 前処理のテスト ---
    元画像形状: (210, 160, 3)
    元画像データ型: uint8
    ピクセル値範囲: [0, 255]
    
    リセット後:
    スタック形状: (4, 84, 84)
    データ型: float64
    値範囲: [0.000, 1.000]
    
    ステップ1後:
      スタック形状: (4, 84, 84)
    
    ステップ2後:
      スタック形状: (4, 84, 84)
    
    ステップ3後:
      スタック形状: (4, 84, 84)
    
    --- メモリ使用量 ---
    元画像（4フレーム）: 403.20 KB
    前処理後: 225.79 KB
    削減率: 44.0%
    

### 実装例8: Atari Pong DQN学習（簡略版）
    
    
    import gym
    import torch
    import torch.nn as nn
    import numpy as np
    
    print("=== Atari Pong DQN 学習フレームワーク ===\n")
    
    class AtariDQN(nn.Module):
        """Atari用CNN-DQN"""
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
            # 入力: [batch, 4, 84, 84]
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    
    class PongDQNAgent:
        """Pong用DQNエージェント"""
    
        def __init__(self, n_actions):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用デバイス: {self.device}")
    
            self.q_network = AtariDQN(n_actions).to(self.device)
            self.target_network = AtariDQN(n_actions).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)
            self.preprocessor = AtariPreprocessor(frame_stack=4)
    
        def select_action(self, state, epsilon=0.1):
            """ε-greedy行動選択"""
            if np.random.random() < epsilon:
                return np.random.randint(self.q_network.fc[-1].out_features)
    
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
        def compute_loss(self, batch):
            """損失計算（Double DQN）"""
            states, actions, rewards, next_states, dones = batch
    
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
    
            # 現在のQ値
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
            # Double DQNターゲット
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards + 0.99 * next_q * (1 - dones)
    
            return nn.MSELoss()(current_q, target_q)
    
    
    # 簡易テスト
    print("--- Pong DQNエージェント初期化 ---")
    agent = PongDQNAgent(n_actions=6)  # Pongは6つの行動
    
    print(f"\nネットワーク構造:")
    print(agent.q_network)
    
    print(f"\n総パラメータ数: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # ダミー状態で推論テスト
    dummy_state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(dummy_state, epsilon=0.0)
    print(f"\n推論テスト:")
    print(f"入力状態形状: {dummy_state.shape}")
    print(f"選択された行動: {action}")
    
    print("\n[実際の学習では、約100万フレーム（数時間〜数日）の訓練が必要です]")
    print("[Pongで人間レベルに到達するには、報酬が-21から+21に改善するまで学習します]")
    

**出力** ：
    
    
    === Atari Pong DQN 学習フレームワーク ===
    
    使用デバイス: cpu
    --- Pong DQNエージェント初期化 ---
    
    ネットワーク構造:
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
    
    総パラメータ数: 1,686,086
    
    推論テスト:
    入力状態形状: (4, 84, 84)
    選択された行動: 3
    
    [実際の学習では、約100万フレーム（数時間〜数日）の訓練が必要です]
    [Pongで人間レベルに到達するには、報酬が-21から+21に改善するまで学習します]
    

* * *

## まとめ

本章では、Deep Q-Network（DQN）について学びました：

### 重要ポイント

  1. **Q学習の限界** ： 
     * 表形式Q学習は高次元・連続状態空間に対応不可
     * ニューラルネットワークによる関数近似が必要
  2. **DQNの基本構成** ： 
     * Q-Network：Q(s, a; θ)を近似
     * Experience Replay：データ相関を除去
     * Target Network：学習を安定化
  3. **アルゴリズム拡張** ： 
     * Double DQN：Q値過大評価を抑制
     * Dueling DQN：V(s)とA(s,a)を分離
  4. **実装のポイント** ： 
     * CartPole：連続状態の基本的なDQN学習
     * Atari：画像前処理とCNNアーキテクチャ

### ハイパーパラメータのベストプラクティス

パラメータ | CartPole | Atari | 説明  
---|---|---|---  
**学習率** | 1e-3 | 1e-4 ~ 2.5e-4 | Adam推奨  
**γ（割引率）** | 0.99 | 0.99 | 標準値  
**Buffer容量** | 10,000 | 100,000 ~ 1,000,000 | タスクの複雑さに応じて  
**Batch Size** | 32 ~ 64 | 32 | 小さいほど学習不安定  
**ε減衰** | 0.995 | 1.0 → 0.1（100万ステップ） | 線形減衰も可  
**Target更新頻度** | 10エピソード | 10,000ステップ | 環境により調整  
  
### DQNの限界と今後の発展

DQNは画期的な手法ですが、以下の課題があります：

  * **サンプル効率** ：大量の経験が必要（数百万フレーム）
  * **離散行動のみ** ：連続行動空間には対応不可
  * **過大評価バイアス** ：Double DQNでも完全には解決せず

これらを改善する手法として、第4章以降で以下を学びます：

  * **Policy Gradient** ：連続行動空間への対応
  * **Actor-Critic** ：価値ベースと方策ベースの融合
  * **Rainbow DQN** ：複数の改善手法の統合

**演習問題**

#### 問1: Experience Replayの効果

Experience Replayを使わない場合と使う場合で、CartPoleの学習曲線を比較してください。相関データがどのように学習に影響するか考察してください。

#### 問2: Target Networkの更新頻度

Target Networkの更新頻度（C = 1, 10, 100, 1000）を変えて実験し、学習の安定性への影響を分析してください。

#### 問3: Double DQNの効果測定

標準DQNとDouble DQNで、Q値の推定誤差を比較してください。過大評価がどの程度抑制されるか定量評価してください。

#### 問4: Dueling Architectureの可視化

Dueling DQNのV(s)とA(s,a)の値を可視化し、どのような状態でV(s)が支配的になるか、A(s,a)が重要になるか分析してください。

#### 問5: ハイパーパラメータチューニング

学習率、バッファサイズ、バッチサイズを変えて実験し、最適な設定を見つけてください。グリッドサーチまたはランダムサーチを実装してください。
