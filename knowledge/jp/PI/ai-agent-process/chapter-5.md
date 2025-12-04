---
title: AI寺子屋
chapter_title: AI寺子屋
subtitle: AIとマテリアルサイエンスの学習コンテンツ
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/ai-agent-process/chapter-5.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Ai Agent Process](<../../PI/ai-agent-process/index.html>)›Chapter 5

AIエージェントによる自律プロセス運転 シリーズ

# Chapter 5: 実プラントへのデプロイと安全性

← Chapter 4: マルチエージェント協調制御（準備中） [シリーズ目次](<index.html>)

## Chapter 5の概要

強化学習エージェントをシミュレーション環境で学習させるだけでは不十分です。 実際のプラントにデプロイする際には、**安全性の確保** 、**シミュレーションと現実のギャップの克服** 、 **不確実性への対処** など、多くの課題があります。 

本章では、実プラントへの適用を見据えた7つの重要技術を実装例とともに解説します。 化学プラントは高温・高圧・危険物を扱うため、AIによる自律制御には特に慎重なアプローチが必要です。 

#### 安全性に関する重要な注意

本章で扱う技術は、実プラントへの適用を前提としています。 実際のデプロイには、プロセス安全の専門知識、規制遵守、十分な検証・テストが必要です。 本コードは教育目的であり、実プラントでの使用には適切な安全評価が不可欠です。 

### 本章で学ぶこと

  * **Sim-to-Real転移** ：ドメインランダマイゼーションによるロバスト性向上
  * **安全な探索** ：行動制約による危険領域の回避
  * **保守的Q学習（CQL）** ：過大評価を防ぐオフライン学習
  * **人間オーバーライド** ：緊急時の人間介入機構
  * **不確実性定量化** ：ベイズNNやアンサンブルによる信頼区間推定
  * **性能監視とドリフト検出** ：継続的なモニタリング
  * **統合デプロイメントフレームワーク** ：全要素を組み合わせた実装

## 実プラントデプロイの課題

### Sim-to-Real Gap（シミュレーションと現実のギャップ）
    
    
    ```mermaid
    graph LR
                        A[シミュレーション環境] -->|理想的なモデル| B[完璧な制御]
                        C[実プラント] -->|モデル誤差外乱センサーノイズ| D[性能劣化]
                        B -.->|Sim-to-Real Gap| D
                        E[ドメインランダマイゼーション] --> F[ロバストな方策]
                        F --> C
    ```

### 安全性の階層

レイヤー | 機能 | 実装  
---|---|---  
1\. 行動制約 | 危険な行動の禁止 | ハードリミット、セーフティフィルタ  
2\. 不確実性考慮 | 信頼区間の評価 | ベイズNN、アンサンブル  
3\. 性能監視 | 異常検知 | ドリフト検出、KPIモニタリング  
4\. 人間介入 | 緊急停止 | オーバーライド機構  
  
1 ドメインランダマイゼーションによるSim-to-Real転移 

シミュレーション環境のパラメータをランダム化して学習することで、 実環境の不確実性に対してロバストな方策を獲得します。 

**ドメインランダマイゼーション：**

\\[ \theta \sim p(\Theta), \quad \pi^* = \arg\max_\pi \mathbb{E}_{\theta \sim p(\Theta)} [J(\pi; \theta)] \\] 

ここで、\\(\theta\\)は環境パラメータ、\\(p(\Theta)\\)はパラメータ分布
    
    
    # ドメインランダマイゼーション実装
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class RandomizedCSTREnv:
        """パラメータをランダム化したCSTR環境"""
        def __init__(self, randomize=True):
            self.randomize = randomize
            self.reset()
    
        def _sample_parameters(self):
            """物理パラメータのランダムサンプリング"""
            if self.randomize:
                # 活性化エネルギー（±20%の変動）
                self.Ea = np.random.uniform(40000, 60000)
    
                # 反応熱（±15%の変動）
                self.dHr = np.random.uniform(-57500, -42500)
    
                # 熱伝達係数（±25%の変動）
                self.U = np.random.uniform(300, 500)
    
                # 体積（製造ばらつき±5%）
                self.V = np.random.uniform(950, 1050)
    
                # センサーノイズ標準偏差
                self.temp_noise = np.random.uniform(0.1, 1.0)
                self.conc_noise = np.random.uniform(0.01, 0.05)
    
                # 制御遅延（通信遅延+バルブ応答）
                self.control_delay = np.random.randint(1, 4)
            else:
                # 公称値
                self.Ea = 50000
                self.dHr = -50000
                self.U = 400
                self.V = 1000
                self.temp_noise = 0.5
                self.conc_noise = 0.02
                self.control_delay = 2
    
        def reset(self):
            self._sample_parameters()
            self.state = np.array([350.0, 2.0])  # [温度, 濃度]
            self.action_buffer = [0.5] * self.control_delay
            return self._get_observation()
    
        def _get_observation(self):
            """ノイズを含む観測"""
            T, CA = self.state
            T_obs = T + np.random.normal(0, self.temp_noise)
            CA_obs = CA + np.random.normal(0, self.conc_noise)
            return np.array([T_obs, CA_obs])
    
        def step(self, action):
            """制御遅延を考慮したステップ"""
            # 遅延のある行動を適用
            self.action_buffer.append(action)
            actual_action = self.action_buffer.pop(0)
    
            T, CA = self.state
    
            # 反応速度（ランダム化されたパラメータ）
            R = 8.314
            k = 1e10 * np.exp(-self.Ea / (R * T))
    
            # CSTR dynamics
            dt = 0.1
            F = 100  # 流量 [L/min]
            CA_in = 2.5
            Tin = 350
            rho = 1000
            Cp = 4.18
    
            # 冷却量（行動）
            Q_cool = actual_action * 10000  # 0-10000 W
    
            # 物質収支
            dCA = (F / self.V) * (CA_in - CA) - k * CA
            CA_new = CA + dCA * dt
    
            # エネルギー収支
            Q_rxn = -self.dHr * k * CA * self.V
            Q_jacket = self.U * 10 * (T - Tin) + Q_cool
            dT = (Q_rxn - Q_jacket) / (self.V * rho * Cp)
            T_new = T + dT * dt
    
            self.state = np.array([T_new, max(0, CA_new)])
    
            # 報酬
            temp_penalty = -abs(T_new - 350) ** 2 * 0.01
            production = k * CA * 10
            reward = temp_penalty + production
    
            done = T_new > 400 or T_new < 300  # 安全範囲外
            return self._get_observation(), reward, done
    
    # ロバストなSAC学習
    class RobustSACAgent:
        """ドメインランダマイゼーションを使うSACエージェント"""
        def __init__(self, obs_dim, action_dim):
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, action_dim), nn.Tanh()
            )
            self.optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
    
        def select_action(self, obs):
            with torch.no_grad():
                return self.actor(torch.FloatTensor(obs)).numpy()
    
    # 学習：ランダム化環境で訓練
    train_env = RandomizedCSTREnv(randomize=True)
    agent = RobustSACAgent(obs_dim=2, action_dim=1)
    
    print("ランダム化環境での学習...")
    for episode in range(500):
        obs = train_env.reset()
        episode_reward = 0
    
        for step in range(100):
            action = agent.select_action(obs)
            next_obs, reward, done = train_env.step(action[0])
            episode_reward += reward
            obs = next_obs
    
            if done:
                break
    
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
            print(f"  Env params: Ea={train_env.Ea:.0f}, V={train_env.V:.0f}, "
                  f"delay={train_env.control_delay}")
    
    # 評価：公称環境（実プラント想定）でテスト
    test_env = RandomizedCSTREnv(randomize=False)
    print("\n公称環境でのテスト...")
    
    obs = test_env.reset()
    test_reward = 0
    temps = []
    
    for step in range(100):
        action = agent.select_action(obs)
        obs, reward, done = test_env.step(action[0])
        test_reward += reward
        temps.append(obs[0])
    
        if done:
            break
    
    print(f"Test Reward: {test_reward:.2f}")
    print(f"Temp Mean: {np.mean(temps):.2f}K, Std: {np.std(temps):.2f}K")

2 安全な探索：行動制約によるセーフティフィルタ 

強化学習エージェントが危険な行動を取らないよう、物理的・安全的制約を課します。 Control Barrier Functions (CBF)やセーフティレイヤーを実装します。 
    
    
    ```mermaid
    graph LR
                            A[RL方策π] -->|危険な行動?| B[セーフティフィルタ]
                            B -->|安全な行動| C[実行]
                            B -->|制約違反| D[安全な代替行動]
                            D --> C
    ```
    
    
    # 安全な探索：セーフティフィルタ実装
    import numpy as np
    import torch
    import torch.nn as nn
    
    class SafetyConstraints:
        """CSTRの安全制約"""
        def __init__(self):
            # 温度制約
            self.T_min = 310.0  # [K]
            self.T_max = 390.0  # [K]
            self.T_target = 350.0
    
            # 濃度制約
            self.CA_min = 0.1  # [mol/L]
            self.CA_max = 3.0
    
            # 制御入力制約
            self.u_min = -100.0  # 最大冷却 [kW]
            self.u_max = 50.0    # 最大加熱 [kW]
    
            # 変化率制約
            self.du_max = 20.0  # [kW/step]
    
        def is_safe_state(self, state):
            """状態が安全かチェック"""
            T, CA = state
            return (self.T_min <= T <= self.T_max and
                    self.CA_min <= CA <= self.CA_max)
    
        def is_safe_action(self, state, action, prev_action):
            """行動が安全かチェック"""
            # 制御入力範囲
            if not (self.u_min <= action <= self.u_max):
                return False
    
            # 変化率制約
            if abs(action - prev_action) > self.du_max:
                return False
    
            return True
    
        def project_to_safe(self, action, prev_action):
            """行動を安全領域に射影"""
            # 範囲制約
            action = np.clip(action, self.u_min, self.u_max)
    
            # 変化率制約
            delta = action - prev_action
            if abs(delta) > self.du_max:
                action = prev_action + np.sign(delta) * self.du_max
    
            return action
    
    class ControlBarrierFunction:
        """Control Barrier Function (CBF)による安全保証"""
        def __init__(self, safety_constraints):
            self.constraints = safety_constraints
            self.alpha = 0.5  # クラスK関数のゲイン
    
        def barrier_function(self, state):
            """バリア関数 h(x) >= 0 が安全領域"""
            T, CA = state
    
            # 温度バリア（距離関数）
            h_T_min = T - self.constraints.T_min
            h_T_max = self.constraints.T_max - T
    
            # 濃度バリア
            h_CA_min = CA - self.constraints.CA_min
            h_CA_max = self.constraints.CA_max - CA
    
            # 最小値（最も厳しい制約）
            return min(h_T_min, h_T_max, h_CA_min, h_CA_max)
    
        def safe_action(self, state, desired_action, env_model):
            """CBF制約を満たす安全な行動を計算"""
            h = self.barrier_function(state)
    
            # 安全な領域なら何もしない
            if h > 10.0:
                return desired_action
    
            # 境界近くでは制約を課す
            # 簡易実装：予測される次状態でバリア条件をチェック
            next_state_pred = env_model.predict(state, desired_action)
            h_next = self.barrier_function(next_state_pred)
    
            # CBF条件: h_next >= -alpha * h
            if h_next >= -self.alpha * h:
                return desired_action
            else:
                # 安全側に修正（保守的な行動）
                T, CA = state
                if T > self.constraints.T_target:
                    # 冷却強化
                    return max(desired_action, 0)
                else:
                    # 加熱抑制
                    return min(desired_action, 0)
    
    class SimpleCSTRModel:
        """CSTRの簡易予測モデル"""
        def predict(self, state, action, dt=0.1):
            T, CA = state
            k = 1e10 * np.exp(-50000 / (8.314 * T))
    
            # 簡易dynamics
            dCA = -k * CA * dt
            dT = (action * 1000 - 400 * (T - 350)) / 4180 * dt
    
            return np.array([T + dT, CA + dCA])
    
    # 使用例
    safety = SafetyConstraints()
    cbf = ControlBarrierFunction(safety)
    model = SimpleCSTRModel()
    
    # RLエージェントの行動にセーフティフィルタを適用
    class SafeRLAgent:
        def __init__(self, base_agent, safety_filter):
            self.base_agent = base_agent
            self.safety_filter = safety_filter
            self.prev_action = 0.0
    
        def select_safe_action(self, state):
            # ベースエージェントの行動
            desired_action = self.base_agent.select_action(state)[0] * 100
    
            # セーフティフィルタ適用
            safe_action = self.safety_filter.project_to_safe(
                desired_action, self.prev_action)
    
            # CBF制約
            safe_action = cbf.safe_action(state, safe_action, model)
    
            self.prev_action = safe_action
            return safe_action
    
    # シミュレーション
    from example1 import RobustSACAgent, RandomizedCSTREnv
    
    base_agent = RobustSACAgent(obs_dim=2, action_dim=1)
    safe_agent = SafeRLAgent(base_agent, safety)
    env = RandomizedCSTREnv(randomize=False)
    
    print("安全制約付き実行...")
    state = env.reset()
    unsafe_count = 0
    
    for step in range(200):
        action = safe_agent.select_safe_action(state)
        next_state, reward, done = env.step(action / 100)
    
        if not safety.is_safe_state(state):
            unsafe_count += 1
            print(f"Step {step}: UNSAFE STATE! T={state[0]:.1f}K, CA={state[1]:.3f}")
    
        state = next_state
    
        if done:
            break
    
    print(f"\nUnsafe states encountered: {unsafe_count} / {step+1}")
    print(f"Safety rate: {(1 - unsafe_count/(step+1))*100:.1f}%")

3 Conservative Q-Learning（CQL）：保守的なオフライン学習 

実プラントでは探索が危険なため、過去のデータからオフラインで学習します。 CQLは分布外行動のQ値を過小評価し、安全な方策を学習します。 

**CQL目的関数：**

\\[ \min_Q \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim \mu(a|s)} [Q(s,a)] \right] + \mathcal{L}_{TD}(Q) \\] 

第1項：分布外行動のQ値を下げる、第2項：データ内行動のQ値を保つ
    
    
    # Conservative Q-Learning (CQL) 実装
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from collections import deque
    import random
    
    class CQLQNetwork(nn.Module):
        """CQL用のQ関数"""
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1)
            )
    
        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            return self.net(x)
    
    class CQLAgent:
        """Conservative Q-Learning エージェント"""
        def __init__(self, state_dim, action_dim, alpha=1.0):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.alpha = alpha  # CQL正則化係数
    
            self.q_net = CQLQNetwork(state_dim, action_dim)
            self.target_q = CQLQNetwork(state_dim, action_dim)
            self.target_q.load_state_dict(self.q_net.state_dict())
    
            self.policy = nn.Sequential(
                nn.Linear(state_dim, 256), nn.ReLU(),
                nn.Linear(256, action_dim), nn.Tanh()
            )
    
            self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
    
        def select_action(self, state, deterministic=False):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy(state_t)
                if not deterministic:
                    action += torch.randn_like(action) * 0.1
                return action.squeeze().numpy()
    
        def train_step(self, batch):
            """CQL更新"""
            states, actions, rewards, next_states, dones = batch
    
            states_t = torch.FloatTensor(states)
            actions_t = torch.FloatTensor(actions)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones).unsqueeze(1)
    
            # Q関数の更新
            # 1. Bellman誤差
            with torch.no_grad():
                next_actions = self.policy(next_states_t)
                target_q = rewards_t + 0.99 * self.target_q(
                    next_states_t, next_actions) * (1 - dones_t)
    
            current_q = self.q_net(states_t, actions_t)
            bellman_loss = nn.MSELoss()(current_q, target_q)
    
            # 2. CQL正則化項
            # ランダム行動のQ値を計算
            num_random = 10
            random_actions = torch.FloatTensor(
                np.random.uniform(-1, 1, (states_t.shape[0], num_random, self.action_dim)))
    
            random_q = []
            for i in range(num_random):
                q = self.q_net(states_t, random_actions[:, i, :])
                random_q.append(q)
            random_q = torch.cat(random_q, dim=1)
    
            # ポリシーによる行動のQ値
            policy_actions = self.policy(states_t)
            policy_q = self.q_net(states_t, policy_actions)
    
            # CQL項：ランダム行動のQ値を大きく、データ内行動を小さく評価
            cql_loss = (torch.logsumexp(random_q, dim=1).mean() -
                        policy_q.mean())
    
            # 合計損失
            q_loss = bellman_loss + self.alpha * cql_loss
    
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
    
            # 3. ポリシー更新（Q値最大化）
            policy_loss = -self.q_net(states_t, self.policy(states_t)).mean()
    
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
    
            # ターゲット更新
            for target_param, param in zip(self.target_q.parameters(),
                                            self.q_net.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
    
            return q_loss.item(), cql_loss.item(), policy_loss.item()
    
    # オフラインデータセット生成（過去の運転データ）
    def generate_offline_data(n_trajectories=100):
        from example1 import RandomizedCSTREnv
    
        env = RandomizedCSTREnv(randomize=True)
        dataset = []
    
        for _ in range(n_trajectories):
            state = env.reset()
            for _ in range(50):
                # ランダムポリシー + ノイズ（現実的なデータ）
                action = np.random.uniform(-1, 1, 1)
                next_state, reward, done = env.step(action[0])
                dataset.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    break
    
        return dataset
    
    # CQL学習
    print("オフラインデータ生成...")
    offline_data = generate_offline_data(n_trajectories=200)
    print(f"Dataset size: {len(offline_data)}")
    
    agent = CQLAgent(state_dim=2, action_dim=1, alpha=1.0)
    
    print("\nCQL学習中...")
    batch_size = 256
    for epoch in range(100):
        # ミニバッチサンプリング
        batch = random.sample(offline_data, min(batch_size, len(offline_data)))
        states, actions, rewards, next_states, dones = zip(*batch)
    
        batch_tuple = (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
        q_loss, cql_loss, p_loss = agent.train_step(batch_tuple)
    
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Q_loss={q_loss:.3f}, CQL_loss={cql_loss:.3f}, "
                  f"Policy_loss={p_loss:.3f}")
    
    # テスト
    print("\nCQL方策のテスト...")
    from example1 import RandomizedCSTREnv
    test_env = RandomizedCSTREnv(randomize=False)
    
    state = test_env.reset()
    total_reward = 0
    
    for step in range(100):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done = test_env.step(action[0])
        total_reward += reward
        state = next_state
        if done:
            break
    
    print(f"Test Reward: {total_reward:.2f}")

4 Human-in-the-Loop：人間オーバーライド機構 

AIエージェントが予期しない行動をとる場合に備え、オペレーターが介入できる機構が必要です。 介入判断、スムーズな移行、履歴記録を実装します。 
    
    
    ```mermaid
    graph TD
                            A[AI制御] -->|異常検知| B{人間判断}
                            B -->|OK| A
                            B -->|介入| C[手動制御]
                            C -->|安定化| D{復帰条件}
                            D -->|満たす| A
                            D -->|満たさない| C
    ```
    
    
    # Human-in-the-Loop システム実装
    import numpy as np
    import time
    from datetime import datetime
    from enum import Enum
    
    class ControlMode(Enum):
        """制御モード"""
        AI_CONTROL = "AI"
        HUMAN_OVERRIDE = "Human"
        TRANSITION = "Transition"
    
    class HumanOverrideSystem:
        """人間介入システム"""
        def __init__(self):
            self.mode = ControlMode.AI_CONTROL
            self.intervention_history = []
            self.confidence_threshold = 0.7
    
        def check_intervention_needed(self, state, ai_action, confidence):
            """介入が必要かチェック"""
            T, CA = state
    
            # トリガー条件
            triggers = {
                'high_temp': T > 380,
                'low_temp': T < 320,
                'low_confidence': confidence < self.confidence_threshold,
                'extreme_action': abs(ai_action) > 0.9,
                'unstable_state': CA < 0.2 or CA > 2.8
            }
    
            if any(triggers.values()):
                reason = [k for k, v in triggers.items() if v]
                return True, reason
            return False, []
    
        def request_human_action(self, state, ai_suggestion):
            """オペレーターに行動を問い合わせ（実際はGUI/CLI）"""
            T, CA = state
            print(f"\n{'='*60}")
            print(f"人間介入要求！")
            print(f"現在の状態: T={T:.2f}K, CA={CA:.3f}mol/L")
            print(f"AI提案行動: {ai_suggestion:.3f}")
            print(f"{'='*60}")
    
            # 簡略化：ルールベースでシミュレート
            # 実際はオペレーターの入力を待つ
            if T > 380:
                human_action = -0.8  # 強冷却
                override = True
            elif T < 320:
                human_action = 0.6   # 加熱
                override = True
            else:
                human_action = ai_suggestion
                override = False
    
            return human_action, override
    
        def smooth_transition(self, from_action, to_action, alpha=0.3):
            """スムーズな制御移行"""
            return alpha * to_action + (1 - alpha) * from_action
    
        def log_intervention(self, timestamp, state, ai_action, human_action, reason):
            """介入履歴記録"""
            log_entry = {
                'timestamp': timestamp,
                'state': state.copy(),
                'ai_action': ai_action,
                'human_action': human_action,
                'reason': reason
            }
            self.intervention_history.append(log_entry)
    
        def generate_report(self):
            """介入レポート生成"""
            if not self.intervention_history:
                return "No interventions recorded."
    
            report = "\n" + "="*60 + "\n"
            report += "Human Intervention Report\n"
            report += "="*60 + "\n"
            report += f"Total interventions: {len(self.intervention_history)}\n\n"
    
            for i, entry in enumerate(self.intervention_history):
                report += f"Intervention {i+1}:\n"
                report += f"  Time: {entry['timestamp']}\n"
                report += f"  State: T={entry['state'][0]:.2f}K, CA={entry['state'][1]:.3f}\n"
                report += f"  AI action: {entry['ai_action']:.3f}\n"
                report += f"  Human action: {entry['human_action']:.3f}\n"
                report += f"  Reason: {', '.join(entry['reason'])}\n\n"
    
            return report
    
    class HITLController:
        """Human-in-the-Loop制御システム"""
        def __init__(self, ai_agent, override_system):
            self.ai_agent = ai_agent
            self.override_system = override_system
            self.prev_action = 0.0
    
        def select_action(self, state, confidence=1.0):
            """人間介入を考慮した行動選択"""
            # AI提案
            ai_action = self.ai_agent.select_action(state)[0]
    
            # 介入判定
            need_intervention, reasons = self.override_system.check_intervention_needed(
                state, ai_action, confidence)
    
            if need_intervention:
                # 人間に問い合わせ
                human_action, overridden = self.override_system.request_human_action(
                    state, ai_action)
    
                if overridden:
                    # 介入記録
                    self.override_system.log_intervention(
                        datetime.now(), state, ai_action, human_action, reasons)
                    self.override_system.mode = ControlMode.HUMAN_OVERRIDE
                    final_action = human_action
                else:
                    final_action = ai_action
            else:
                final_action = ai_action
                self.override_system.mode = ControlMode.AI_CONTROL
    
            # スムーズ遷移
            final_action = self.override_system.smooth_transition(
                self.prev_action, final_action)
    
            self.prev_action = final_action
            return final_action
    
    # 使用例
    from example1 import RobustSACAgent, RandomizedCSTREnv
    
    print("Human-in-the-Loop システム起動...")
    
    ai_agent = RobustSACAgent(obs_dim=2, action_dim=1)
    override_system = HumanOverrideSystem()
    controller = HITLController(ai_agent, override_system)
    
    env = RandomizedCSTREnv(randomize=True)
    state = env.reset()
    
    # 過酷な条件でテスト
    for step in range(50):
        # 信頼度をシミュレート（不確実性が高い状況）
        confidence = np.random.uniform(0.5, 1.0)
    
        action = controller.select_action(state, confidence)
        next_state, reward, done = env.step(action)
    
        print(f"Step {step}: Mode={override_system.mode.value}, "
              f"T={state[0]:.1f}K, Action={action:.3f}")
    
        state = next_state
    
        if done:
            print("エピソード終了（異常状態）")
            break
    
    # レポート生成
    print(override_system.generate_report())

5 不確実性定量化：ベイズNNとアンサンブル手法 

AIの予測がどれだけ確からしいかを定量化することで、不確実性が高い状況では保守的な行動を取ります。 ベイズニューラルネットワークやアンサンブル手法で信頼区間を推定します。 
    
    
    # 不確実性定量化：アンサンブルとベイズ近似
    import torch
    import torch.nn as nn
    import numpy as np
    
    class EnsembleQNetwork:
        """Q関数のアンサンブル"""
        def __init__(self, state_dim, action_dim, n_models=5):
            self.n_models = n_models
            self.models = [
                nn.Sequential(
                    nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
                    nn.Linear(128, 128), nn.ReLU(),
                    nn.Linear(128, 1)
                ) for _ in range(n_models)
            ]
            self.optimizers = [
                torch.optim.Adam(m.parameters(), lr=1e-3) for m in self.models
            ]
    
        def predict_with_uncertainty(self, state, action):
            """予測平均と不確実性（標準偏差）を返す"""
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.FloatTensor(action).unsqueeze(0)
            x = torch.cat([state_t, action_t], dim=-1)
    
            predictions = []
            for model in self.models:
                with torch.no_grad():
                    pred = model(x).item()
                predictions.append(pred)
    
            mean = np.mean(predictions)
            std = np.std(predictions)
    
            return mean, std
    
        def train_step(self, batch):
            """全モデルを異なるブートストラップサンプルで学習"""
            states, actions, targets = batch
    
            losses = []
            for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
                # ブートストラップサンプリング
                indices = np.random.choice(len(states), len(states), replace=True)
                s = torch.FloatTensor(states[indices])
                a = torch.FloatTensor(actions[indices])
                t = torch.FloatTensor(targets[indices])
    
                x = torch.cat([s, a], dim=-1)
                pred = model(x).squeeze()
    
                loss = nn.MSELoss()(pred, t)
                opt.zero_grad()
                loss.backward()
                opt.step()
    
                losses.append(loss.item())
    
            return np.mean(losses)
    
    class MCDropoutQNetwork(nn.Module):
        """Monte Carlo Dropout（ベイズ近似）"""
        def __init__(self, state_dim, action_dim, dropout_rate=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
    
        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            return self.net(x)
    
        def predict_with_uncertainty(self, state, action, n_samples=20):
            """MC Dropoutで不確実性推定"""
            self.train()  # Dropoutを有効化
    
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.FloatTensor(action).unsqueeze(0)
    
            predictions = []
            for _ in range(n_samples):
                pred = self.forward(state_t, action_t).item()
                predictions.append(pred)
    
            mean = np.mean(predictions)
            std = np.std(predictions)
    
            return mean, std
    
    class UncertaintyAwareAgent:
        """不確実性を考慮したエージェント"""
        def __init__(self, state_dim, action_dim, method='ensemble'):
            self.method = method
            if method == 'ensemble':
                self.q_network = EnsembleQNetwork(state_dim, action_dim, n_models=5)
            else:
                self.q_network = MCDropoutQNetwork(state_dim, action_dim)
    
            self.policy = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, action_dim), nn.Tanh()
            )
            self.uncertainty_threshold = 0.5  # 不確実性の閾値
    
        def select_action(self, state):
            """不確実性を考慮した行動選択"""
            with torch.no_grad():
                nominal_action = self.policy(torch.FloatTensor(state)).numpy()
    
            # 複数候補の不確実性評価
            action_candidates = [
                nominal_action,
                nominal_action * 0.5,  # 保守的
                np.zeros_like(nominal_action)  # 現状維持
            ]
    
            best_action = nominal_action
            min_uncertainty = float('inf')
    
            for action in action_candidates:
                if self.method == 'ensemble':
                    q_mean, q_std = self.q_network.predict_with_uncertainty(state, action)
                else:
                    q_mean, q_std = self.q_network.predict_with_uncertainty(state, action)
    
                # 不確実性が低く、Q値が高い行動を選択
                if q_std < self.uncertainty_threshold and q_std < min_uncertainty:
                    best_action = action
                    min_uncertainty = q_std
    
            return best_action, min_uncertainty
    
    # デモンストレーション
    print("不確実性定量化デモ\n")
    
    # アンサンブル手法
    ensemble_agent = UncertaintyAwareAgent(state_dim=2, action_dim=1, method='ensemble')
    
    test_states = [
        np.array([350.0, 1.5]),  # 正常
        np.array([385.0, 0.8]),  # 高温
        np.array([310.0, 2.5])   # 低温
    ]
    
    print("Ensemble Method:")
    for i, state in enumerate(test_states):
        action, uncertainty = ensemble_agent.select_action(state)
        print(f"State {i+1}: T={state[0]}K, CA={state[1]}")
        print(f"  Action: {action[0]:.3f}, Uncertainty: {uncertainty:.3f}")
    
        if uncertainty > ensemble_agent.uncertainty_threshold:
            print(f"  WARNING: High uncertainty! Consider human oversight.")
        print()
    
    # MC Dropout手法
    mc_agent = UncertaintyAwareAgent(state_dim=2, action_dim=1, method='mcdropout')
    
    print("\nMC Dropout Method:")
    for i, state in enumerate(test_states):
        action, uncertainty = mc_agent.select_action(state)
        print(f"State {i+1}: T={state[0]}K, CA={state[1]}")
        print(f"  Action: {action[0]:.3f}, Uncertainty: {uncertainty:.3f}")
    
        if uncertainty > mc_agent.uncertainty_threshold:
            print(f"  WARNING: High uncertainty!")
        print()

6 性能監視とドリフト検出 

デプロイ後も継続的に性能を監視し、プラントの経年劣化やモデルドリフトを検出します。 KPIモニタリング、統計的検定、異常検知を実装します。 
    
    
    # 性能監視とドリフト検出システム
    import numpy as np
    from collections import deque
    from scipy import stats
    
    class PerformanceMonitor:
        """性能監視システム"""
        def __init__(self, window_size=100):
            self.window_size = window_size
    
            # KPI履歴
            self.rewards = deque(maxlen=window_size)
            self.temperatures = deque(maxlen=window_size)
            self.concentrations = deque(maxlen=window_size)
            self.actions = deque(maxlen=window_size)
    
            # ベースライン統計
            self.baseline_reward_mean = None
            self.baseline_reward_std = None
    
            # 異常カウンター
            self.anomaly_count = 0
            self.total_steps = 0
    
        def update(self, state, action, reward):
            """KPI更新"""
            T, CA = state
            self.rewards.append(reward)
            self.temperatures.append(T)
            self.concentrations.append(CA)
            self.actions.append(action)
            self.total_steps += 1
    
        def set_baseline(self):
            """ベースライン性能を設定"""
            if len(self.rewards) >= self.window_size:
                self.baseline_reward_mean = np.mean(self.rewards)
                self.baseline_reward_std = np.std(self.rewards)
                print(f"Baseline set: μ={self.baseline_reward_mean:.2f}, "
                      f"σ={self.baseline_reward_std:.2f}")
    
        def detect_drift(self, alpha=0.05):
            """統計的ドリフト検出（t検定）"""
            if self.baseline_reward_mean is None or len(self.rewards) < 50:
                return False, None
    
            current_mean = np.mean(list(self.rewards)[-50:])
    
            # t検定
            t_stat, p_value = stats.ttest_1samp(
                list(self.rewards)[-50:],
                self.baseline_reward_mean
            )
    
            drift_detected = p_value < alpha and current_mean < self.baseline_reward_mean
    
            return drift_detected, {
                't_stat': t_stat,
                'p_value': p_value,
                'current_mean': current_mean,
                'baseline_mean': self.baseline_reward_mean
            }
    
        def detect_anomaly(self, state, action):
            """異常検出（3σルール）"""
            if len(self.rewards) < 30:
                return False
    
            T, CA = state
    
            # 温度異常
            temp_mean = np.mean(self.temperatures)
            temp_std = np.std(self.temperatures)
            temp_anomaly = abs(T - temp_mean) > 3 * temp_std
    
            # 濃度異常
            conc_mean = np.mean(self.concentrations)
            conc_std = np.std(self.concentrations)
            conc_anomaly = abs(CA - conc_mean) > 3 * conc_std
    
            # 行動異常
            action_mean = np.mean(self.actions)
            action_std = np.std(self.actions)
            action_anomaly = abs(action - action_mean) > 3 * action_std
    
            is_anomaly = temp_anomaly or conc_anomaly or action_anomaly
    
            if is_anomaly:
                self.anomaly_count += 1
    
            return is_anomaly
    
        def generate_report(self):
            """監視レポート生成"""
            if len(self.rewards) == 0:
                return "No data collected."
    
            report = "\n" + "="*60 + "\n"
            report += "Performance Monitoring Report\n"
            report += "="*60 + "\n\n"
    
            report += f"Total steps: {self.total_steps}\n"
            report += f"Anomalies detected: {self.anomaly_count} "
            report += f"({self.anomaly_count/self.total_steps*100:.2f}%)\n\n"
    
            report += "KPI Statistics (last {} steps):\n".format(len(self.rewards))
            report += f"  Reward: μ={np.mean(self.rewards):.2f}, "
            report += f"σ={np.std(self.rewards):.2f}\n"
            report += f"  Temperature: μ={np.mean(self.temperatures):.2f}K, "
            report += f"σ={np.std(self.temperatures):.2f}K\n"
            report += f"  Concentration: μ={np.mean(self.concentrations):.3f}, "
            report += f"σ={np.std(self.concentrations):.3f}\n"
            report += f"  Action: μ={np.mean(self.actions):.3f}, "
            report += f"σ={np.std(self.actions):.3f}\n\n"
    
            # ドリフト検出
            drift, drift_info = self.detect_drift()
            if drift:
                report += "WARNING: Performance drift detected!\n"
                report += f"  Current mean reward: {drift_info['current_mean']:.2f}\n"
                report += f"  Baseline mean reward: {drift_info['baseline_mean']:.2f}\n"
                report += f"  p-value: {drift_info['p_value']:.4f}\n"
            else:
                report += "No significant drift detected.\n"
    
            report += "="*60 + "\n"
            return report
    
    class DriftDetector:
        """より高度なドリフト検出（ADWIN）"""
        def __init__(self, delta=0.002):
            self.delta = delta
            self.window = deque()
            self.drift_detected = False
    
        def add_element(self, value):
            """データ点を追加してドリフトチェック"""
            self.window.append(value)
    
            # 簡易版ADWIN: ウィンドウを2分割して平均を比較
            if len(self.window) > 50:
                mid = len(self.window) // 2
                window1 = list(self.window)[:mid]
                window2 = list(self.window)[mid:]
    
                # Welchのt検定（等分散を仮定しない）
                t_stat, p_value = stats.ttest_ind(window1, window2, equal_var=False)
    
                if p_value < self.delta:
                    self.drift_detected = True
                    self.window.clear()  # ドリフト検出後リセット
                    return True
    
            return False
    
    # デモンストレーション
    from example1 import RandomizedCSTREnv, RobustSACAgent
    
    print("性能監視システム起動...\n")
    
    env = RandomizedCSTREnv(randomize=False)
    agent = RobustSACAgent(obs_dim=2, action_dim=1)
    monitor = PerformanceMonitor(window_size=100)
    drift_detector = DriftDetector()
    
    # フェーズ1: 正常運転（ベースライン設定）
    print("Phase 1: ベースライン設定中...")
    state = env.reset()
    
    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action[0])
    
        monitor.update(state, action[0], reward)
    
        state = next_state if not done else env.reset()
    
    monitor.set_baseline()
    
    # フェーズ2: 劣化シミュレーション
    print("\nPhase 2: プラント劣化をシミュレート...")
    
    for step in range(200):
        action = agent.select_action(state)
    
        # 時間経過で性能劣化をシミュレート
        degradation = step / 200 * 0.3
        next_state, reward, done = env.step(action[0])
        reward -= degradation * 10  # 性能低下
    
        monitor.update(state, action[0], reward)
    
        # 異常検出
        if monitor.detect_anomaly(state, action[0]):
            print(f"Step {step+100}: Anomaly detected!")
    
        # ドリフト検出
        if drift_detector.add_element(reward):
            print(f"Step {step+100}: DRIFT DETECTED by ADWIN!")
    
        state = next_state if not done else env.reset()
    
        # 定期的なドリフトチェック
        if step % 50 == 0:
            drift, info = monitor.detect_drift()
            if drift:
                print(f"\nStep {step+100}: Statistical drift detected!")
                print(f"  Current: {info['current_mean']:.2f}, "
                      f"Baseline: {info['baseline_mean']:.2f}")
    
    # 最終レポート
    print(monitor.generate_report())

7 統合デプロイメントフレームワーク 

これまでの全要素（ドメインランダマイゼーション、安全制約、人間オーバーライド、 不確実性定量化、性能監視）を統合した実践的なデプロイメントシステムです。 
    
    
    ```mermaid
    graph TD
                            A[センサーデータ] --> B[前処理]
                            B --> C[不確実性推定]
                            C --> D{不確実性高?}
                            D -->|Yes| E[保守的行動]
                            D -->|No| F[AI方策]
                            F --> G[安全フィルタ]
                            E --> G
                            G --> H{安全?}
                            H -->|No| I[人間オーバーライド]
                            H -->|Yes| J[アクチュエータ]
                            I --> J
                            J --> K[性能監視]
                            K --> L{ドリフト?}
                            L -->|Yes| M[アラート+再学習]
                            L -->|No| A
    ```
    
    
    # 統合デプロイメントフレームワーク
    import numpy as np
    import torch
    from datetime import datetime
    
    class IntegratedDeploymentSystem:
        """全機能統合デプロイメントシステム"""
        def __init__(self, agent, env):
            # コンポーネント
            self.agent = agent
            self.env = env
    
            # 例2: 安全制約
            from example2 import SafetyConstraints, ControlBarrierFunction, SimpleCSTRModel
            self.safety = SafetyConstraints()
            self.cbf = ControlBarrierFunction(self.safety)
            self.model = SimpleCSTRModel()
    
            # 例4: 人間オーバーライド
            from example4 import HumanOverrideSystem, ControlMode
            self.override_system = HumanOverrideSystem()
    
            # 例5: 不確実性定量化
            from example5 import EnsembleQNetwork
            self.uncertainty_estimator = EnsembleQNetwork(
                state_dim=2, action_dim=1, n_models=5)
    
            # 例6: 性能監視
            from example6 import PerformanceMonitor, DriftDetector
            self.monitor = PerformanceMonitor(window_size=100)
            self.drift_detector = DriftDetector()
    
            # システム状態
            self.prev_action = 0.0
            self.running = True
            self.emergency_stop = False
    
        def preprocess_observation(self, raw_obs):
            """センサーデータの前処理"""
            # 外れ値除去（簡易）
            T, CA = raw_obs
            T = np.clip(T, 250, 450)
            CA = np.clip(CA, 0, 5)
            return np.array([T, CA])
    
        def estimate_uncertainty(self, state, action):
            """不確実性推定"""
            q_mean, q_std = self.uncertainty_estimator.predict_with_uncertainty(
                state, action)
            return q_std
    
        def apply_safety_filter(self, state, action):
            """安全フィルタ適用"""
            # 制約射影
            safe_action = self.safety.project_to_safe(action, self.prev_action)
    
            # CBF制約
            safe_action = self.cbf.safe_action(state, safe_action, self.model)
    
            return safe_action
    
        def check_human_override(self, state, action, uncertainty):
            """人間介入チェック"""
            need_intervention, reasons = self.override_system.check_intervention_needed(
                state, action, confidence=1.0 - uncertainty)
    
            if need_intervention:
                human_action, overridden = self.override_system.request_human_action(
                    state, action)
    
                if overridden:
                    self.override_system.log_intervention(
                        datetime.now(), state, action, human_action, reasons)
                    return human_action, True
    
            return action, False
    
        def monitor_performance(self, state, action, reward):
            """性能監視とドリフト検出"""
            self.monitor.update(state, action, reward)
    
            # 異常検出
            if self.monitor.detect_anomaly(state, action):
                print(f"  [MONITOR] Anomaly detected at step {self.monitor.total_steps}")
    
            # ドリフト検出
            if self.drift_detector.add_element(reward):
                print(f"  [MONITOR] Performance drift detected!")
                return True  # 再学習トリガー
    
            # 定期的な統計的ドリフトチェック
            if self.monitor.total_steps % 100 == 0:
                drift, info = self.monitor.detect_drift()
                if drift:
                    print(f"  [MONITOR] Statistical drift: "
                          f"current={info['current_mean']:.2f}, "
                          f"baseline={info['baseline_mean']:.2f}")
                    return True
    
            return False
    
        def control_loop(self, n_steps=500):
            """メイン制御ループ"""
            print("統合デプロイメントシステム起動\n")
            print("="*60)
    
            state = self.env.reset()
            state = self.preprocess_observation(state)
    
            # ベースライン設定
            for step in range(100):
                action = self.agent.select_action(state)[0]
                next_state, reward, done = self.env.step(action)
                next_state = self.preprocess_observation(next_state)
    
                self.monitor.update(state, action, reward)
                state = next_state if not done else self.env.reset()
    
            self.monitor.set_baseline()
            print("ベースライン設定完了\n")
    
            # メインループ
            for step in range(n_steps):
                if self.emergency_stop:
                    print("緊急停止！")
                    break
    
                # 1. AI方策
                raw_action = self.agent.select_action(state)[0]
    
                # 2. 不確実性推定
                uncertainty = self.estimate_uncertainty(state, np.array([raw_action]))
    
                # 高不確実性時は保守的に
                if uncertainty > 0.5:
                    raw_action *= 0.5
                    print(f"  [UNCERTAINTY] High uncertainty ({uncertainty:.3f}), "
                          f"conservative action")
    
                # 3. 安全フィルタ
                safe_action = self.apply_safety_filter(state, raw_action)
    
                # 4. 人間オーバーライド
                final_action, overridden = self.check_human_override(
                    state, safe_action, uncertainty)
    
                # 5. 実行
                next_state, reward, done = self.env.step(final_action)
                next_state = self.preprocess_observation(next_state)
    
                # 6. 性能監視
                need_retraining = self.monitor_performance(state, final_action, reward)
    
                if need_retraining:
                    print(f"  [SYSTEM] 再学習が推奨されます")
    
                # 定期レポート
                if step % 100 == 0:
                    print(f"\nStep {step}:")
                    print(f"  State: T={state[0]:.2f}K, CA={state[1]:.3f}")
                    print(f"  Action: {final_action:.3f}, Uncertainty: {uncertainty:.3f}")
                    print(f"  Mode: {self.override_system.mode.value}")
                    print(f"  Anomalies: {self.monitor.anomaly_count}")
    
                self.prev_action = final_action
                state = next_state if not done else self.env.reset()
    
            # 最終レポート
            print("\n" + "="*60)
            print("運転終了")
            print("="*60)
            print(self.monitor.generate_report())
            print(self.override_system.generate_report())
    
    # 実行
    from example1 import RobustSACAgent, RandomizedCSTREnv
    
    print("統合デプロイメントシステムのデモンストレーション\n")
    
    env = RandomizedCSTREnv(randomize=True)
    agent = RobustSACAgent(obs_dim=2, action_dim=1)
    
    system = IntegratedDeploymentSystem(agent, env)
    system.control_loop(n_steps=300)

## Chapter 5 まとめ

### 学んだこと

  * **Sim-to-Real転移** ：ドメインランダマイゼーションでロバスト性を獲得
  * **安全な探索** ：セーフティフィルタとCBFで危険回避
  * **保守的Q学習** ：オフラインデータから安全に学習
  * **人間オーバーライド** ：緊急時の介入機構
  * **不確実性定量化** ：アンサンブルやMCDropoutで信頼度評価
  * **性能監視** ：ドリフト検出と異常検知
  * **統合フレームワーク** ：全要素を組み合わせた実用システム

### デプロイメント成熟度モデル

レベル | 説明 | 必要技術  
---|---|---  
L1: 実験室 | シミュレーションのみ | 基本RL  
L2: テスト | パイロットプラント | Sim-to-real、安全制約  
L3: 監視付き | 実プラント（人間監視） | オーバーライド、性能監視  
L4: 自律 | 完全自律運転 | 全機能統合、継続学習  
  
### 実プラント適用のチェックリスト

  * ✓ 十分なシミュレーション検証（1000+エピソード）
  * ✓ パラメータランダマイゼーションの妥当性確認
  * ✓ 安全制約の網羅的定義
  * ✓ 緊急停止プロトコルの実装とテスト
  * ✓ オペレーター訓練とマニュアル整備
  * ✓ 性能監視ダッシュボード構築
  * ✓ 定期的な性能評価と再学習計画
  * ✓ 規制当局への報告体制

#### 最終注意事項

強化学習の実プラント適用は、まだ発展途上の技術です。 特に化学プラントのような安全クリティカルな環境では、段階的なアプローチが不可欠です： 

  1. シミュレーションでの徹底検証
  2. パイロットプラントでの実証
  3. 人間監視下での限定運用
  4. 段階的な自律レベル向上

常に人間のエキスパートと協働し、AIを道具として活用する姿勢が重要です。 

### シリーズ完了

**おめでとうございます！**

「AIエージェントによる自律プロセス運転」シリーズの全5章を修了しました。 強化学習の基礎から実プラントデプロイまで、幅広い知識を習得されました。 

[シリーズ目次に戻る](<index.html>)

さらなる学習には、以下のシリーズもご活用ください： 

  * プロセス監視・制御入門（準備中）
  * プロセス最適化入門（準備中）

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
