---
title: AI寺子屋
chapter_title: AI寺子屋
subtitle: AIとマテリアルサイエンスの学習コンテンツ
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/ai-agent-process/chapter-4.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Ai Agent Process](<../../PI/ai-agent-process/index.html>)›Chapter 4

AIエージェントによる自律プロセス運転 シリーズ

# Chapter 4: マルチエージェント協調制御

[← Chapter 3: モデルフリー制御](<chapter-3.html>) [Chapter 5: 実プラントへのデプロイと安全性 →](<chapter-5.html>) [シリーズ目次](<index.html>)

## Chapter 4の概要

複雑なプロセスプラントでは、複数の反応器や蒸留塔が相互に影響し合いながら運転されます。 このような分散システムでは、単一のAIエージェントではなく、複数のエージェントが協調して制御を行うことで、 より効率的で柔軟な運転が可能になります。 

本章では、マルチエージェント強化学習（MARL: Multi-Agent Reinforcement Learning）の基礎から、 実際のプロセス制御への応用までを7つの実装例とともに解説します。 

### 本章で学ぶこと

  * **CTDE（Centralized Training with Decentralized Execution）** ：学習時は中央集権、実行時は分散
  * **Independent Q-Learning** ：各エージェントが独立して学習
  * **QMIX** ：価値関数の分解と混合によるクレジット割当
  * **エージェント間通信** ：メッセージパッシングによる協調
  * **協調タスク** ：複数反応器の同期制御
  * **競争タスク** ：限られたリソースの配分
  * **混合タスク** ：協調と競争の両面を持つ現実的なシナリオ

## マルチエージェント強化学習の基礎

### 定式化

マルチエージェント系は、部分観測マルコフゲーム（Partially Observable Stochastic Game）として定式化されます： 

**マルコフゲーム：**

\\[ \mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \\{\mathcal{A}^i\\}_{i \in \mathcal{N}}, \mathcal{T}, \\{R^i\\}_{i \in \mathcal{N}}, \\{\mathcal{O}^i\\}_{i \in \mathcal{N}}, \gamma \rangle \\] 

  * \\(\mathcal{N} = \\{1, 2, \ldots, n\\}\\)：エージェント集合
  * \\(\mathcal{S}\\)：グローバル状態空間
  * \\(\mathcal{A}^i\\)：エージェント\\(i\\)の行動空間
  * \\(\mathcal{T}: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \Delta(\mathcal{S})\\)：状態遷移関数
  * \\(R^i: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \mathbb{R}\\)：エージェント\\(i\\)の報酬関数
  * \\(\mathcal{O}^i\\)：エージェント\\(i\\)の観測空間

### 協調・競争・混合の分類
    
    
    ```mermaid
    graph TD
                        A[マルチエージェントタスク] --> B[完全協調]
                        A --> C[完全競争]
                        A --> D[混合]
                        B --> E[共通報酬R¹=R²=...=Rⁿ]
                        C --> F[ゼロサムΣᵢ Rⁱ = 0]
                        D --> G[一般ゲーム協調+競争]
    ```

プロセス制御では、以下のような状況が考えられます： 

  * **協調タスク** ：複数の反応器を協調させて全体の生産性を最大化
  * **競争タスク** ：限られたユーティリティ（蒸気、冷却水）を各プラントが取り合う
  * **混合タスク** ：各プラントが自身の生産目標を達成しつつ、全体の安定性も維持

1 CTDE（Centralized Training with Decentralized Execution）フレームワーク 

CTDEは、学習時には全エージェントの情報を使って中央集権的に学習し、 実行時には各エージェントが自身の観測のみで分散的に行動する枠組みです。 これにより、学習の効率性と実行時のスケーラビリティを両立します。 
    
    
    ```mermaid
    graph LR
                            subgraph Training[学習時（中央集権）]
                                S[グローバル状態] --> C[中央Critic]
                                O1[観測1] --> A1[Actor1]
                                O2[観測2] --> A2[Actor2]
                                O3[観測3] --> A3[Actor3]
                                C --> A1
                                C --> A2
                                C --> A3
                            end
                            subgraph Execution[実行時（分散）]
                                O1'[観測1] --> A1'[Actor1]
                                O2' [観測2] --> A2'[Actor2]
                                O3' [観測3] --> A3'[Actor3]
                            end
    ```
    
    
    # CTDE基本フレームワーク実装
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
    class Actor(nn.Module):
        """各エージェントのActor（分散実行可能）"""
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, action_dim), nn.Tanh()
            )
    
        def forward(self, obs):
            return self.net(obs)
    
    class CentralizedCritic(nn.Module):
        """中央Critic（学習時のみ使用）"""
        def __init__(self, state_dim, n_agents, action_dim):
            super().__init__()
            total_action_dim = n_agents * action_dim
            self.net = nn.Sequential(
                nn.Linear(state_dim + total_action_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 1)
            )
    
        def forward(self, state, actions):
            # actions: [n_agents, action_dim] -> flatten
            x = torch.cat([state, actions.flatten()], dim=-1)
            return self.net(x)
    
    class CTDEAgent:
        """CTDE学習エージェント"""
        def __init__(self, n_agents, obs_dim, action_dim, state_dim):
            self.n_agents = n_agents
            self.actors = [Actor(obs_dim, action_dim) for _ in range(n_agents)]
            self.critic = CentralizedCritic(state_dim, n_agents, action_dim)
    
            self.actor_opts = [optim.Adam(a.parameters(), lr=3e-4) for a in self.actors]
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
    
        def select_actions(self, observations):
            """実行時：各エージェントが独立に行動選択"""
            actions = []
            for i, obs in enumerate(observations):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs)
                    action = self.actors[i](obs_t).numpy()
                actions.append(action)
            return np.array(actions)
    
        def train_step(self, batch):
            """学習時：中央Criticを使った更新"""
            states, obs, actions, rewards, next_states, next_obs, dones = batch
    
            # Critic更新（TD誤差）
            with torch.no_grad():
                next_actions = torch.stack([
                    self.actors[i](torch.FloatTensor(next_obs[i]))
                    for i in range(self.n_agents)
                ])
                target_q = rewards + 0.99 * self.critic(
                    torch.FloatTensor(next_states), next_actions
                ) * (1 - dones)
    
            current_q = self.critic(torch.FloatTensor(states), torch.FloatTensor(actions))
            critic_loss = nn.MSELoss()(current_q, target_q)
    
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
    
            # Actor更新（ポリシー勾配）
            for i in range(self.n_agents):
                new_actions = [
                    self.actors[j](torch.FloatTensor(obs[j])) if j == i
                    else torch.FloatTensor(actions[j])
                    for j in range(self.n_agents)
                ]
                new_actions = torch.stack(new_actions)
    
                actor_loss = -self.critic(torch.FloatTensor(states), new_actions).mean()
    
                self.actor_opts[i].zero_grad()
                actor_loss.backward()
                self.actor_opts[i].step()
    
    # 使用例：3つのCSTRの協調制御
    n_agents = 3
    agent = CTDEAgent(n_agents=n_agents, obs_dim=4, action_dim=2, state_dim=12)
    
    # 実行時は分散（各エージェントは自身の観測のみ使用）
    observations = [np.random.randn(4) for _ in range(n_agents)]
    actions = agent.select_actions(observations)  # 分散実行
    print(f"分散実行での行動: {actions.shape}")  # (3, 2)

2 Independent Q-Learning（IQL）によるマルチエージェント学習 

最もシンプルなマルチエージェント学習手法は、各エージェントが独立してQ学習を行うものです。 他のエージェントは環境の一部とみなされ、非定常性の問題がありますが、実装が容易で実用的なケースも多いです。 
    
    
    # Independent Q-Learning実装
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    import random
    
    class QNetwork(nn.Module):
        """各エージェント独立のQ関数"""
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            )
    
        def forward(self, obs):
            return self.net(obs)
    
    class IQLAgent:
        """Independent Q-Learningエージェント"""
        def __init__(self, obs_dim, action_dim, agent_id):
            self.agent_id = agent_id
            self.action_dim = action_dim
            self.q_net = QNetwork(obs_dim, action_dim)
            self.target_net = QNetwork(obs_dim, action_dim)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
            self.memory = deque(maxlen=10000)
            self.epsilon = 1.0
    
        def select_action(self, obs):
            """ε-greedy行動選択"""
            if random.random() < self.epsilon:
                return np.random.randint(self.action_dim)
            else:
                with torch.no_grad():
                    q_values = self.q_net(torch.FloatTensor(obs))
                    return q_values.argmax().item()
    
        def store_transition(self, obs, action, reward, next_obs, done):
            self.memory.append((obs, action, reward, next_obs, done))
    
        def train_step(self, batch_size=32):
            if len(self.memory) < batch_size:
                return 0.0
    
            batch = random.sample(self.memory, batch_size)
            obs, actions, rewards, next_obs, dones = zip(*batch)
    
            obs_t = torch.FloatTensor(obs)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_obs_t = torch.FloatTensor(next_obs)
            dones_t = torch.FloatTensor(dones)
    
            # 現在のQ値
            q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze()
    
            # ターゲットQ値
            with torch.no_grad():
                next_q_values = self.target_net(next_obs_t).max(1)[0]
                target_q = rewards_t + 0.99 * next_q_values * (1 - dones_t)
    
            # 更新
            loss = nn.MSELoss()(q_values, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            # ε減衰
            self.epsilon = max(0.01, self.epsilon * 0.995)
    
            return loss.item()
    
        def update_target(self):
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    # マルチエージェント環境シミュレーション
    class MultiCSTREnv:
        """3つのCSTRが直列接続された環境"""
        def __init__(self):
            self.n_agents = 3
            self.reset()
    
        def reset(self):
            # 各反応器の状態 [温度, 濃度]
            self.states = np.array([[350.0, 0.5], [340.0, 0.3], [330.0, 0.1]])
            return self.states
    
        def step(self, actions):
            # actions: [0=冷却, 1=加熱, 2=維持] for each agent
            rewards = []
            for i in range(self.n_agents):
                T, C = self.states[i]
    
                # 行動による温度変化
                if actions[i] == 0:  # 冷却
                    T -= 5
                elif actions[i] == 1:  # 加熱
                    T += 5
    
                # 反応進行（簡易モデル）
                k = 0.1 * np.exp((T - 350) / 10)
                C = C * (1 - k * 0.1)
    
                # 次の反応器への流入
                if i < self.n_agents - 1:
                    self.states[i+1, 1] += C * 0.3
    
                self.states[i] = [T, C]
    
                # 報酬：目標温度との差と生産性
                temp_penalty = -abs(T - 350)
                production = k * C
                rewards.append(temp_penalty * 0.1 + production * 10)
    
            done = False
            return self.states.copy(), np.array(rewards), [done]*self.n_agents
    
    # 学習ループ
    env = MultiCSTREnv()
    agents = [IQLAgent(obs_dim=2, action_dim=3, agent_id=i) for i in range(3)]
    
    for episode in range(500):
        obs = env.reset()
        episode_rewards = [0] * 3
    
        for step in range(100):
            actions = [agents[i].select_action(obs[i]) for i in range(3)]
            next_obs, rewards, dones = env.step(actions)
    
            # 各エージェントが独立に学習
            for i in range(3):
                agents[i].store_transition(obs[i], actions[i], rewards[i],
                                           next_obs[i], dones[i])
                agents[i].train_step()
                episode_rewards[i] += rewards[i]
    
            obs = next_obs
    
        # 定期的にターゲット更新
        if episode % 10 == 0:
            for agent in agents:
                agent.update_target()
    
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Rewards: {sum(episode_rewards):.2f}")

3 QMIX：価値分解ネットワークによるクレジット割当 

QMIXは、各エージェントの個別Q値を混合ネットワークで統合し、 単調性制約（Individual-Global-Max: IGM）を保ちながらクレジット割当を実現します。 

**QMIX価値分解：**

\\[ Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) = f_{mix}(Q_1(\tau^1, u^1), \ldots, Q_n(\tau^n, u^n); s) \\] 

単調性制約：

\\[ \frac{\partial Q_{tot}}{\partial Q_i} \geq 0, \quad \forall i \\] 
    
    
    # QMIX実装
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class AgentQNetwork(nn.Module):
        """各エージェントのQ関数"""
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            )
    
        def forward(self, obs):
            return self.net(obs)
    
    class QMixerNetwork(nn.Module):
        """単調性を保証する混合ネットワーク"""
        def __init__(self, n_agents, state_dim):
            super().__init__()
            self.n_agents = n_agents
    
            # ハイパーネットワーク（状態から重みを生成）
            self.hyper_w1 = nn.Linear(state_dim, n_agents * 32)
            self.hyper_b1 = nn.Linear(state_dim, 32)
            self.hyper_w2 = nn.Linear(state_dim, 32)
            self.hyper_b2 = nn.Sequential(
                nn.Linear(state_dim, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )
    
        def forward(self, agent_qs, state):
            """
            agent_qs: [batch, n_agents]
            state: [batch, state_dim]
            """
            batch_size = agent_qs.size(0)
            agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
    
            # 第1層の重み（絶対値で単調性保証）
            w1 = torch.abs(self.hyper_w1(state))
            w1 = w1.view(batch_size, self.n_agents, 32)
            b1 = self.hyper_b1(state).view(batch_size, 1, 32)
    
            # 第1層の出力
            hidden = torch.bmm(agent_qs, w1) + b1
            hidden = torch.relu(hidden)
    
            # 第2層の重み（絶対値で単調性保証）
            w2 = torch.abs(self.hyper_w2(state))
            w2 = w2.view(batch_size, 32, 1)
            b2 = self.hyper_b2(state).view(batch_size, 1, 1)
    
            # 最終出力
            q_tot = torch.bmm(hidden, w2) + b2
            return q_tot.view(batch_size)
    
    class QMIX:
        """QMIX学習アルゴリズム"""
        def __init__(self, n_agents, obs_dim, action_dim, state_dim):
            self.n_agents = n_agents
            self.agent_networks = [AgentQNetwork(obs_dim, action_dim)
                                   for _ in range(n_agents)]
            self.mixer = QMixerNetwork(n_agents, state_dim)
            self.target_networks = [AgentQNetwork(obs_dim, action_dim)
                                    for _ in range(n_agents)]
            self.target_mixer = QMixerNetwork(n_agents, state_dim)
    
            # ターゲット初期化
            for i in range(n_agents):
                self.target_networks[i].load_state_dict(
                    self.agent_networks[i].state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
    
            # オプティマイザ
            params = list(self.mixer.parameters())
            for net in self.agent_networks:
                params += list(net.parameters())
            self.optimizer = optim.Adam(params, lr=5e-4)
    
        def select_actions(self, observations, epsilon=0.05):
            """各エージェントの行動選択"""
            actions = []
            for i, obs in enumerate(observations):
                if torch.rand(1).item() < epsilon:
                    actions.append(torch.randint(0, 5, (1,)).item())
                else:
                    with torch.no_grad():
                        q_vals = self.agent_networks[i](torch.FloatTensor(obs))
                        actions.append(q_vals.argmax().item())
            return actions
    
        def train_step(self, batch):
            """QMIX更新"""
            states, obs_list, actions, rewards, next_states, next_obs_list, dones = batch
    
            # 現在のQ値（各エージェント）
            agent_qs = []
            for i in range(self.n_agents):
                q_vals = self.agent_networks[i](torch.FloatTensor(obs_list[i]))
                q = q_vals.gather(1, torch.LongTensor(actions[:, i]).unsqueeze(1))
                agent_qs.append(q)
            agent_qs = torch.cat(agent_qs, dim=1)
    
            # 混合してQ_tot
            q_tot = self.mixer(agent_qs, torch.FloatTensor(states))
    
            # ターゲットQ値
            with torch.no_grad():
                target_agent_qs = []
                for i in range(self.n_agents):
                    target_q = self.target_networks[i](
                        torch.FloatTensor(next_obs_list[i])).max(1)[0]
                    target_agent_qs.append(target_q.unsqueeze(1))
                target_agent_qs = torch.cat(target_agent_qs, dim=1)
                target_q_tot = self.target_mixer(target_agent_qs,
                                                 torch.FloatTensor(next_states))
                target = rewards + 0.99 * target_q_tot * (1 - dones)
    
            # 損失
            loss = nn.MSELoss()(q_tot, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)
            self.optimizer.step()
    
            return loss.item()
    
    # 使用例
    qmix = QMIX(n_agents=3, obs_dim=4, action_dim=5, state_dim=12)
    observations = [torch.randn(4) for _ in range(3)]
    actions = qmix.select_actions(observations)
    print(f"QMIX actions: {actions}")

4 エージェント間通信プロトコル（メッセージパッシング） 

エージェント間で情報を交換することで、より効果的な協調が可能になります。 CommNetやTarMACなどの手法では、注意機構を用いたメッセージパッシングを実装します。 
    
    
    ```mermaid
    graph LR
                            A1[Agent 1] -->|msg₁| C[CommunicationChannel]
                            A2[Agent 2] -->|msg₂| C
                            A3[Agent 3] -->|msg₃| C
                            C -->|aggregated| A1
                            C -->|aggregated| A2
                            C -->|aggregated| A3
    ```
    
    
    # メッセージパッシング機構の実装
    import torch
    import torch.nn as nn
    
    class AttentionCommModule(nn.Module):
        """注意機構ベースの通信モジュール"""
        def __init__(self, hidden_dim, n_agents):
            super().__init__()
            self.n_agents = n_agents
            self.hidden_dim = hidden_dim
    
            # メッセージ生成
            self.msg_encoder = nn.Linear(hidden_dim, hidden_dim)
    
            # 注意機構
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
    
        def forward(self, hidden_states):
            """
            hidden_states: [n_agents, hidden_dim]
            returns: [n_agents, hidden_dim] (通信後の隠れ状態)
            """
            # メッセージ生成
            messages = self.msg_encoder(hidden_states)  # [n_agents, hidden_dim]
    
            # 注意スコア計算
            Q = self.query(hidden_states)  # [n_agents, hidden_dim]
            K = self.key(messages)  # [n_agents, hidden_dim]
            V = self.value(messages)  # [n_agents, hidden_dim]
    
            # スケールドドット積注意
            attn_scores = torch.matmul(Q, K.T) / (self.hidden_dim ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
    
            # メッセージ集約
            aggregated = torch.matmul(attn_weights, V)
    
            return hidden_states + aggregated  # 残差接続
    
    class CommunicativeAgent(nn.Module):
        """通信可能なエージェント"""
        def __init__(self, obs_dim, action_dim, hidden_dim, n_agents):
            super().__init__()
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU()
            )
            self.comm_module = AttentionCommModule(hidden_dim, n_agents)
            self.policy = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
    
        def forward(self, observations, agent_idx):
            """
            observations: [n_agents, obs_dim]
            agent_idx: このエージェントのインデックス
            """
            # 各エージェントの観測をエンコード
            hidden_states = self.obs_encoder(observations)
    
            # 通信フェーズ
            comm_hidden = self.comm_module(hidden_states)
    
            # 自分の隠れ状態で行動選択
            my_hidden = comm_hidden[agent_idx]
            action_logits = self.policy(my_hidden)
    
            return action_logits, comm_hidden
    
    # 複数ラウンドの通信
    class MultiRoundCommAgent(nn.Module):
        """複数ラウンドの通信を行うエージェント"""
        def __init__(self, obs_dim, action_dim, hidden_dim, n_agents, n_comm_rounds=2):
            super().__init__()
            self.n_comm_rounds = n_comm_rounds
    
            self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
            self.comm_modules = nn.ModuleList([
                AttentionCommModule(hidden_dim, n_agents)
                for _ in range(n_comm_rounds)
            ])
            self.policy = nn.Linear(hidden_dim, action_dim)
    
        def forward(self, observations):
            """
            observations: [n_agents, obs_dim]
            returns: [n_agents, action_dim]
            """
            hidden = torch.relu(self.obs_encoder(observations))
    
            # 複数ラウンドの通信
            for comm_module in self.comm_modules:
                hidden = comm_module(hidden)
    
            # 各エージェントが行動選択
            actions = self.policy(hidden)
            return actions
    
    # 使用例：3反応器の協調制御
    n_agents = 3
    obs_dim = 4  # [温度, 濃度, 流量, 圧力]
    action_dim = 5  # 離散行動
    hidden_dim = 64
    
    agent = MultiRoundCommAgent(obs_dim, action_dim, hidden_dim, n_agents, n_comm_rounds=2)
    
    # シミュレーション
    observations = torch.randn(n_agents, obs_dim)
    actions = agent(observations)
    print(f"通信後の行動: {actions.shape}")  # [3, 5]
    
    # 個別エージェントでの使用
    single_agent = CommunicativeAgent(obs_dim, action_dim, hidden_dim, n_agents)
    action_logits, comm_hidden = single_agent(observations, agent_idx=0)
    print(f"Agent 0 の行動: {torch.softmax(action_logits, dim=-1)}")
    print(f"通信後の隠れ状態: {comm_hidden.shape}")  # [3, 64]

5 協調タスク：3つのCSTRの同期制御 

3つの連続攪拌槽反応器（CSTR）を直列に接続し、全体の生産性を最大化しながら 各反応器の温度を適切に制御する協調タスクを実装します。 
    
    
    # 協調タスク：3つのCSTRの同期制御
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class CooperativeCSTREnv:
        """3つのCSTRが直列接続された協調環境"""
        def __init__(self):
            self.n_agents = 3
            self.dt = 0.1  # 時間刻み [min]
            self.reset()
    
        def reset(self):
            # 各CSTR: [温度T(K), 濃度CA(mol/L), 流量F(L/min)]
            self.states = np.array([
                [350.0, 2.0, 100.0],  # CSTR1
                [340.0, 1.5, 100.0],  # CSTR2
                [330.0, 1.0, 100.0]   # CSTR3
            ])
            self.time = 0
            return self._get_observations()
    
        def _get_observations(self):
            """各エージェントの観測（局所情報+隣接情報）"""
            obs = []
            for i in range(self.n_agents):
                local = self.states[i].copy()
                # 前段の情報（協調のため）
                prev = self.states[i-1] if i > 0 else np.zeros(3)
                # 後段の情報
                next_ = self.states[i+1] if i < self.n_agents-1 else np.zeros(3)
                obs.append(np.concatenate([local, prev, next_]))
            return obs
    
        def step(self, actions):
            """
            actions: [n_agents, 2] = [[Q1, Tin1], [Q2, Tin2], [Q3, Tin3]]
            Q: 冷却流量 [L/min] (0-50)
            Tin: 入口温度 [K] (300-400)
            """
            rewards = []
    
            for i in range(self.n_agents):
                T, CA, F = self.states[i]
                Q = actions[i][0] * 50  # 正規化解除
                Tin = actions[i][1] * 100 + 300
    
                # 反応速度定数（Arrhenius式）
                Ea = 50000  # [J/mol]
                R = 8.314   # [J/(mol·K)]
                k = 1e10 * np.exp(-Ea / (R * T))
    
                # CSTRモデル
                V = 1000  # 反応器体積 [L]
                rho = 1000  # 密度 [g/L]
                Cp = 4.18   # 比熱 [J/(g·K)]
                dHr = -50000  # 反応熱 [J/mol]
    
                # 入口濃度（前段からの流入）
                CA_in = self.states[i-1, 1] if i > 0 else 2.5
    
                # 物質収支
                dCA = (F / V) * (CA_in - CA) - k * CA
                CA_new = CA + dCA * self.dt
    
                # エネルギー収支
                Q_reaction = -dHr * k * CA * V
                Q_cooling = Q * rho * Cp * (T - Tin)
                dT = (Q_reaction - Q_cooling) / (V * rho * Cp)
                T_new = T + dT * self.dt
    
                self.states[i] = [T_new, max(0, CA_new), F]
    
                # 協調報酬：全体の生産性と安定性
                production = k * CA  # 反応速度
                temp_penalty = -abs(T_new - 350) ** 2  # 目標温度からの偏差
                flow_continuity = -abs(F - 100) ** 2 if i > 0 else 0
    
                reward = production * 100 + temp_penalty * 0.1 + flow_continuity * 0.01
                rewards.append(reward)
    
            # 共通報酬（協調タスク）
            total_production = sum([self.states[i, 1] for i in range(self.n_agents)])
            common_reward = total_production * 10
            rewards = [r + common_reward for r in rewards]
    
            self.time += self.dt
            done = self.time >= 10  # 10分間のエピソード
    
            return self._get_observations(), np.array(rewards), [done]*self.n_agents
    
    # QMIX学習（簡略版）
    class SimpleQMIX:
        def __init__(self, n_agents, obs_dim, action_dim):
            self.n_agents = n_agents
            self.q_nets = [nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            ) for _ in range(n_agents)]
            self.mixer = nn.Sequential(
                nn.Linear(n_agents, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )
            params = list(self.mixer.parameters())
            for net in self.q_nets:
                params += list(net.parameters())
            self.optimizer = optim.Adam(params, lr=1e-3)
    
        def select_actions(self, observations):
            actions = []
            for i, obs in enumerate(observations):
                with torch.no_grad():
                    q_vals = self.q_nets[i](torch.FloatTensor(obs))
                    # 連続行動を離散化（簡略化）
                    action = torch.rand(2)  # [Q正規化, Tin正規化]
                    actions.append(action.numpy())
            return np.array(actions)
    
    # 学習実行
    env = CooperativeCSTREnv()
    agent = SimpleQMIX(n_agents=3, obs_dim=9, action_dim=10)
    
    for episode in range(100):
        obs = env.reset()
        episode_reward = 0
    
        for step in range(100):
            actions = agent.select_actions(obs)
            next_obs, rewards, dones = env.step(actions)
            episode_reward += sum(rewards)
            obs = next_obs
    
            if dones[0]:
                break
    
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")
            print(f"  Final Temps: {[f'{s[0]:.1f}K' for s in env.states]}")
            print(f"  Final Concs: {[f'{s[1]:.3f}mol/L' for s in env.states]}")

6 競争タスク：限られたユーティリティの配分 

複数のプラントが限られた蒸気や冷却水といったユーティリティを取り合う競争シナリオです。 各エージェントは自身の生産を最大化しつつ、資源制約の下で調整する必要があります。 
    
    
    # 競争タスク：ユーティリティ配分問題
    import numpy as np
    import torch
    import torch.nn as nn
    
    class CompetitiveUtilityEnv:
        """限られたユーティリティを複数プラントで競争"""
        def __init__(self, n_plants=3):
            self.n_plants = n_plants
            self.total_steam = 500.0  # 総蒸気供給量 [kg/h]
            self.total_cooling = 1000.0  # 総冷却水 [L/min]
            self.reset()
    
        def reset(self):
            # 各プラント: [生産速度, 温度, 蒸気使用量, 冷却水使用量]
            self.states = np.array([
                [50.0, 350.0, 150.0, 300.0] for _ in range(self.n_plants)
            ])
            return self._get_observations()
    
        def _get_observations(self):
            """各プラントの観測"""
            obs = []
            for i in range(self.n_plants):
                # 自身の状態 + 資源の残量情報
                steam_used = sum(self.states[:, 2])
                cooling_used = sum(self.states[:, 3])
                steam_avail = max(0, self.total_steam - steam_used)
                cooling_avail = max(0, self.total_cooling - cooling_used)
    
                obs_i = np.concatenate([
                    self.states[i],
                    [steam_avail, cooling_avail]
                ])
                obs.append(obs_i)
            return obs
    
        def step(self, actions):
            """
            actions: [n_plants, 2] = [[steam_request, cooling_request], ...]
            各値は0-1で正規化
            """
            # 要求量を実数値に変換
            steam_requests = actions[:, 0] * 200  # 0-200 kg/h
            cooling_requests = actions[:, 1] * 400  # 0-400 L/min
    
            # 資源配分（比例配分）
            total_steam_req = sum(steam_requests)
            total_cooling_req = sum(cooling_requests)
    
            if total_steam_req > self.total_steam:
                steam_allocated = steam_requests * (self.total_steam / total_steam_req)
            else:
                steam_allocated = steam_requests
    
            if total_cooling_req > self.total_cooling:
                cooling_allocated = cooling_requests * (self.total_cooling / total_cooling_req)
            else:
                cooling_allocated = cooling_requests
    
            rewards = []
            for i in range(self.n_plants):
                # 生産速度は資源に依存
                steam_factor = steam_allocated[i] / 200
                cooling_factor = cooling_allocated[i] / 400
                production = 100 * steam_factor * cooling_factor
    
                # 温度管理（冷却不足でペナルティ）
                temp_change = (steam_allocated[i] * 0.5 - cooling_allocated[i] * 0.3)
                temp_new = self.states[i, 1] + temp_change
                temp_penalty = -abs(temp_new - 350) if temp_new > 380 else 0
    
                self.states[i] = [production, temp_new,
                                steam_allocated[i], cooling_allocated[i]]
    
                # 報酬：生産性 - リソース不足ペナルティ
                shortage_penalty = 0
                if steam_allocated[i] < steam_requests[i]:
                    shortage_penalty += (steam_requests[i] - steam_allocated[i]) * 0.5
                if cooling_allocated[i] < cooling_requests[i]:
                    shortage_penalty += (cooling_requests[i] - cooling_allocated[i]) * 0.3
    
                reward = production - shortage_penalty + temp_penalty
                rewards.append(reward)
    
            done = False
            return self._get_observations(), np.array(rewards), [done]*self.n_plants
    
    # Nash Q-Learning（競争タスク用）
    class NashQLearningAgent:
        """Nash均衡を学習するエージェント"""
        def __init__(self, obs_dim, action_dim, agent_id):
            self.agent_id = agent_id
            self.q_net = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
            self.epsilon = 0.3
    
        def select_action(self, obs):
            """ε-greedy with Nash equilibrium consideration"""
            if np.random.rand() < self.epsilon:
                return np.random.rand(2)
            else:
                with torch.no_grad():
                    # 連続行動の近似（簡略化）
                    return torch.sigmoid(torch.randn(2)).numpy()
    
        def update_policy(self, obs, action, reward, next_obs):
            """Q学習更新（簡略版）"""
            # 実装は省略（競争環境での学習アルゴリズム）
            pass
    
    # シミュレーション
    env = CompetitiveUtilityEnv(n_plants=3)
    agents = [NashQLearningAgent(obs_dim=6, action_dim=2, agent_id=i)
              for i in range(3)]
    
    for episode in range(50):
        obs = env.reset()
        episode_rewards = [0] * 3
    
        for step in range(50):
            actions = np.array([agents[i].select_action(obs[i]) for i in range(3)])
            next_obs, rewards, dones = env.step(actions)
    
            for i in range(3):
                episode_rewards[i] += rewards[i]
    
            obs = next_obs
    
        if episode % 10 == 0:
            print(f"\nEpisode {episode}")
            print(f"  Individual Rewards: {[f'{r:.1f}' for r in episode_rewards]}")
            print(f"  Productions: {[f'{s[0]:.1f}' for s in env.states]}")
            print(f"  Steam Usage: {[f'{s[2]:.1f}' for s in env.states]} / {env.total_steam}")
            print(f"  Cooling Usage: {[f'{s[3]:.1f}' for s in env.states]} / {env.total_cooling}")

7 混合タスク：協調と競争が共存する生産システム 

現実のプラントでは、協調と競争の両面が存在します。 各プラントは自身の生産目標を達成しつつ（競争）、全体の安定性や効率も考慮する（協調）必要があります。 
    
    
    # 混合タスク：協調と競争が共存する生産システム
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class MixedCoopCompEnv:
        """協調と競争が混在する複雑な環境"""
        def __init__(self, n_plants=3):
            self.n_plants = n_plants
            self.total_energy = 1000.0  # 総エネルギー [kW]
            self.production_targets = [100, 120, 80]  # 各プラントの目標生産量
            self.reset()
    
        def reset(self):
            # 各プラント: [生産量, 温度, エネルギー使用, 品質]
            self.states = np.array([
                [50.0, 350.0, 300.0, 0.9] for _ in range(self.n_plants)
            ])
            self.time = 0
            return self._get_observations()
    
        def _get_observations(self):
            obs = []
            for i in range(self.n_plants):
                # 自身の状態 + 目標 + 他プラントの生産量（協調のため）
                others_production = [self.states[j, 0] for j in range(self.n_plants) if j != i]
                total_energy_used = sum(self.states[:, 2])
    
                obs_i = np.concatenate([
                    self.states[i],
                    [self.production_targets[i]],
                    others_production,
                    [total_energy_used, self.total_energy]
                ])
                obs.append(obs_i)
            return obs
    
        def step(self, actions):
            """
            actions: [n_plants, 3] = [[energy_req, temp_setpoint, quality_target], ...]
            """
            # エネルギー配分（競争要素）
            energy_requests = actions[:, 0] * 500
            total_energy_req = sum(energy_requests)
    
            if total_energy_req > self.total_energy:
                # 不足時は優先度ベースで配分（目標達成度が低いプラントを優先）
                priorities = [
                    max(0, self.production_targets[i] - self.states[i, 0])
                    for i in range(self.n_plants)
                ]
                total_priority = sum(priorities) + 1e-6
                energy_allocated = [
                    self.total_energy * (priorities[i] / total_priority)
                    for i in range(self.n_plants)
                ]
            else:
                energy_allocated = energy_requests
    
            rewards = []
            for i in range(self.n_plants):
                temp_setpoint = actions[i, 1] * 100 + 300  # 300-400K
                quality_target = actions[i, 2]  # 0-1
    
                # 生産モデル
                energy_factor = energy_allocated[i] / 500
                temp_factor = 1.0 - abs(temp_setpoint - 350) / 100
                production = self.production_targets[i] * energy_factor * temp_factor
    
                # 品質モデル
                quality = 0.5 + 0.5 * quality_target * temp_factor
    
                # 温度更新
                temp = self.states[i, 1] + (temp_setpoint - self.states[i, 1]) * 0.3
    
                self.states[i] = [production, temp, energy_allocated[i], quality]
    
                # 報酬設計（混合）
                # 1. 個別目標達成（競争要素）
                target_achievement = -abs(production - self.production_targets[i])
    
                # 2. 品質ペナルティ
                quality_reward = quality * 10
    
                # 3. エネルギー効率（協調要素）
                energy_efficiency = production / (energy_allocated[i] + 1)
    
                reward = target_achievement + quality_reward + energy_efficiency * 5
                rewards.append(reward)
    
            # 協調ボーナス：全体の安定性
            total_production = sum(self.states[:, 0])
            stability = -np.std(self.states[:, 1])  # 温度の標準偏差
            cooperation_bonus = (total_production / sum(self.production_targets)) * 50 + stability
    
            # 最終報酬 = 個別報酬 + 協調ボーナス
            rewards = [r + cooperation_bonus * 0.3 for r in rewards]
    
            self.time += 1
            done = self.time >= 100
    
            return self._get_observations(), np.array(rewards), [done]*self.n_plants
    
    # COMA (Counterfactual Multi-Agent Policy Gradient)
    class COMAAgent:
        """混合タスク用のCOMAエージェント"""
        def __init__(self, n_agents, obs_dim, action_dim, state_dim):
            self.n_agents = n_agents
    
            # 各エージェントのActor
            self.actors = [nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, action_dim), nn.Tanh()
            ) for _ in range(n_agents)]
    
            # 中央Critic（反事実ベースライン）
            self.critic = nn.Sequential(
                nn.Linear(state_dim + n_agents * action_dim, 128), nn.ReLU(),
                nn.Linear(128, n_agents)  # 各エージェントのQ値
            )
    
            self.actor_opts = [optim.Adam(a.parameters(), lr=3e-4) for a in self.actors]
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
    
        def select_actions(self, observations):
            actions = []
            for i, obs in enumerate(observations):
                with torch.no_grad():
                    action = self.actors[i](torch.FloatTensor(obs)).numpy()
                actions.append(action)
            return np.array(actions)
    
    # 学習ループ
    env = MixedCoopCompEnv(n_plants=3)
    agent = COMAAgent(n_agents=3, obs_dim=9, action_dim=3, state_dim=12)
    
    for episode in range(100):
        obs = env.reset()
        episode_rewards = [0] * 3
    
        for step in range(100):
            actions = agent.select_actions(obs)
            next_obs, rewards, dones = env.step(actions)
    
            for i in range(3):
                episode_rewards[i] += rewards[i]
    
            obs = next_obs
    
            if dones[0]:
                break
    
        if episode % 20 == 0:
            print(f"\nEpisode {episode}")
            print(f"  Individual Rewards: {[f'{r:.1f}' for r in episode_rewards]}")
            print(f"  Productions: {[f'{s[0]:.1f}' for s in env.states]}")
            print(f"  Targets: {env.production_targets}")
            print(f"  Qualities: {[f'{s[3]:.2f}' for s in env.states]}")
            print(f"  Total Energy Used: {sum(env.states[:, 2]):.1f} / {env.total_energy}")

## Chapter 4 まとめ

### 学んだこと

  * **CTDE** ：学習時は中央集権、実行時は分散の効率的なフレームワーク
  * **Independent Q-Learning** ：最もシンプルだが非定常性の問題あり
  * **QMIX** ：価値分解と単調性制約によるクレジット割当
  * **通信機構** ：注意機構ベースのメッセージパッシングで協調強化
  * **協調タスク** ：共通報酬で全体最適を目指す
  * **競争タスク** ：限られた資源の配分問題
  * **混合タスク** ：現実のプラント運転に近い複雑なシナリオ

### アルゴリズム比較

手法 | 適用タスク | 学習効率 | 実装難易度  
---|---|---|---  
IQL | 単純な協調 | 低 | 低  
QMIX | 協調タスク | 高 | 中  
CTDE | 混合タスク | 高 | 中  
CommNet | 複雑な協調 | 高 | 高  
  
### プロセス制御への示唆

  * 複数の反応器や蒸留塔の協調制御にはQMIXやCTDEが有効
  * ユーティリティ配分のような競争タスクにはNash Q-Learningを検討
  * 通信遅延がある場合は、観測にバッファ情報を含める
  * 実プラントでは混合タスクが一般的で、報酬設計が重要

### 次のChapter

**[Chapter 5: 実プラントへのデプロイと安全性](<chapter-5.html>)**

実際のプラントに強化学習エージェントをデプロイする際の安全性確保、sim-to-real転移、不確実性の定量化、人間によるオーバーライド機構などを学びます。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
