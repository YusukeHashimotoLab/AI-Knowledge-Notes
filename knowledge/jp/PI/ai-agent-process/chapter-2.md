---
title: 第2章：プロセス環境のモデリング
chapter_title: 第2章：プロセス環境のモデリング
subtitle: OpenAI Gymで強化学習可能な化学プロセス環境を構築
---

## 2.1 状態空間の定義

強化学習において、状態空間（State Space）は環境の現在の状況を表す変数の集合です。化学プロセスでは、温度、圧力、濃度、流量などの連続変数が状態を構成します。

**💡 状態空間の設計原則**

  * **マルコフ性** : 現在の状態に未来の挙動を決定する情報が含まれている
  * **観測可能性** : センサーで実際に測定可能な変数を選択
  * **正規化** : 各変数を同じスケール（例: 0-1）に変換
  * **次元削減** : 冗長な変数を除去し学習効率を向上

### Example 1: 状態空間の構成と正規化

CSTR（連続攪拌槽反応器）の状態空間を定義し、正規化を実装します。
    
    
    import numpy as np
    from typing import Dict, Tuple
    import gym
    from gym import spaces
    
    # ===================================
    # Example 1: 状態空間の定義と正規化
    # ===================================
    
    class StateSpace:
        """化学プロセスの状態空間定義"""
    
        def __init__(self):
            # 物理変数の範囲（最小値、最大値）
            self.bounds = {
                'temperature': (300.0, 400.0),      # K
                'pressure': (1.0, 10.0),            # bar
                'concentration': (0.0, 2.0),        # mol/L
                'flow_rate': (0.5, 5.0),            # L/min
                'level': (0.0, 100.0)               # %
            }
    
        def get_state_vector(self, physical_state: Dict) -> np.ndarray:
            """物理変数から状態ベクトルを構成"""
            state = np.array([
                physical_state['temperature'],
                physical_state['pressure'],
                physical_state['concentration'],
                physical_state['flow_rate'],
                physical_state['level']
            ])
            return state
    
        def normalize(self, state: np.ndarray) -> np.ndarray:
            """状態を[0, 1]範囲に正規化"""
            normalized = np.zeros_like(state)
            for i, var_name in enumerate(self.bounds.keys()):
                min_val, max_val = self.bounds[var_name]
                normalized[i] = (state[i] - min_val) / (max_val - min_val)
                normalized[i] = np.clip(normalized[i], 0, 1)
            return normalized
    
        def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
            """正規化された状態を物理値に戻す"""
            state = np.zeros_like(normalized_state)
            for i, var_name in enumerate(self.bounds.keys()):
                min_val, max_val = self.bounds[var_name]
                state[i] = normalized_state[i] * (max_val - min_val) + min_val
            return state
    
        def get_gym_space(self) -> spaces.Box:
            """OpenAI Gym用の状態空間を取得"""
            low = np.array([bounds[0] for bounds in self.bounds.values()])
            high = np.array([bounds[1] for bounds in self.bounds.values()])
            return spaces.Box(low=low, high=high, dtype=np.float32)
    
    
    # ===== 使用例 =====
    print("=== Example 1: 状態空間の定義と正規化 ===\n")
    
    state_space = StateSpace()
    
    # サンプル状態
    physical_state = {
        'temperature': 350.0,
        'pressure': 5.5,
        'concentration': 1.2,
        'flow_rate': 2.5,
        'level': 75.0
    }
    
    # 状態ベクトルの構成
    state_vector = state_space.get_state_vector(physical_state)
    print("物理状態ベクトル:")
    print(state_vector)
    
    # 正規化
    normalized = state_space.normalize(state_vector)
    print("\n正規化された状態（0-1範囲）:")
    print(normalized)
    
    # 逆正規化で確認
    denormalized = state_space.denormalize(normalized)
    print("\n逆正規化された状態（元の物理値）:")
    print(denormalized)
    
    # Gym空間の定義
    gym_space = state_space.get_gym_space()
    print(f"\nOpenAI Gym状態空間:")
    print(f"  Low: {gym_space.low}")
    print(f"  High: {gym_space.high}")
    print(f"  Shape: {gym_space.shape}")
    
    # ランダムサンプリング
    random_state = gym_space.sample()
    print(f"\nランダムサンプル: {random_state}")
    

**出力例:**  
物理状態ベクトル:  
[350. 5.5 1.2 2.5 75. ]  
  
正規化された状態（0-1範囲）:  
[0.5 0.5 0.6 0.44 0.75]  
  
OpenAI Gym状態空間:  
Shape: (5,) 

## 2.2 行動空間の設計

行動空間（Action Space）は、エージェントが実行できる操作の集合です。離散行動（バルブ開閉）と連続行動（流量調整）があります。

### Example 2: 離散・連続・混合行動空間の実装
    
    
    import gym
    from gym import spaces
    import numpy as np
    
    # ===================================
    # Example 2: 行動空間の設計
    # ===================================
    
    class ActionSpaceDesign:
        """行動空間の設計パターン"""
    
        @staticmethod
        def discrete_action_space() -> spaces.Discrete:
            """離散行動空間（例: バルブ操作）
    
            Actions:
                0: バルブ全閉
                1: バルブ25%開
                2: バルブ50%開
                3: バルブ75%開
                4: バルブ全開
            """
            return spaces.Discrete(5)
    
        @staticmethod
        def continuous_action_space() -> spaces.Box:
            """連続行動空間（例: 流量制御）
    
            Actions:
                [0]: ヒーター出力 (0-10 kW)
                [1]: 冷却水流量 (0-5 L/min)
            """
            return spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([10.0, 5.0]),
                dtype=np.float32
            )
    
        @staticmethod
        def mixed_action_space() -> spaces.Dict:
            """混合行動空間（離散+連続）
    
            Actions:
                'mode': 運転モード選択 (0: 待機, 1: 運転, 2: 停止)
                'heating': ヒーター出力 (0-10 kW)
                'flow': 流量 (0-5 L/min)
            """
            return spaces.Dict({
                'mode': spaces.Discrete(3),
                'heating': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                'flow': spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32)
            })
    
        @staticmethod
        def apply_safety_constraints(action: np.ndarray, state: np.ndarray) -> np.ndarray:
            """安全制約の適用
    
            Args:
                action: 元の行動
                state: 現在の状態 [temp, pressure, ...]
    
            Returns:
                制約後の行動
            """
            safe_action = action.copy()
    
            # 制約1: 高温時はヒーター出力を制限
            if state[0] > 380:  # 温度が380K以上
                safe_action[0] = min(safe_action[0], 2.0)  # ヒーター最大2kW
    
            # 制約2: 高圧時は流量を制限
            if len(state) > 1 and state[1] > 8:  # 圧力が8bar以上
                safe_action[1] = min(safe_action[1], 1.0)  # 流量最大1L/min
    
            # 制約3: 物理的な限界
            safe_action = np.clip(safe_action, [0.0, 0.0], [10.0, 5.0])
    
            return safe_action
    
    
    # ===== 使用例 =====
    print("\n=== Example 2: 行動空間の設計 ===\n")
    
    designer = ActionSpaceDesign()
    
    # 1. 離散行動空間
    discrete_space = designer.discrete_action_space()
    print("離散行動空間:")
    print(f"  アクション数: {discrete_space.n}")
    print(f"  サンプル: {discrete_space.sample()}")
    
    # 2. 連続行動空間
    continuous_space = designer.continuous_action_space()
    print("\n連続行動空間:")
    print(f"  Low: {continuous_space.low}")
    print(f"  High: {continuous_space.high}")
    print(f"  サンプル: {continuous_space.sample()}")
    
    # 3. 混合行動空間
    mixed_space = designer.mixed_action_space()
    print("\n混合行動空間:")
    sample_mixed = mixed_space.sample()
    print(f"  Mode: {sample_mixed['mode']}")
    print(f"  Heating: {sample_mixed['heating']}")
    print(f"  Flow: {sample_mixed['flow']}")
    
    # 4. 安全制約の適用
    print("\n安全制約の適用:")
    unsafe_action = np.array([8.0, 4.0])  # ヒーター8kW, 流量4L/min
    high_temp_state = np.array([385.0, 5.0])  # 高温状態
    
    safe_action = designer.apply_safety_constraints(unsafe_action, high_temp_state)
    print(f"  元の行動: {unsafe_action}")
    print(f"  制約後: {safe_action}")
    print(f"  理由: 温度{high_temp_state[0]:.0f}K > 380K → ヒーター出力を2kW以下に制限")
    

**出力例:**  
離散行動空間:  
アクション数: 5  
サンプル: 2  
  
連続行動空間:  
サンプル: [6.23 2.84]  
  
安全制約の適用:  
元の行動: [8. 4.]  
制約後: [2. 4.] 

## 2.3 報酬関数の基礎設計

報酬関数（Reward Function）は、エージェントの行動の良し悪しを数値化します。化学プロセスでは、設定値追従、エネルギー効率、安全性などを考慮した多目的報酬関数を設計します。

### Example 3: 多目的報酬関数の実装
    
    
    import numpy as np
    from typing import Dict
    
    # ===================================
    # Example 3: 多目的報酬関数
    # ===================================
    
    class RewardFunction:
        """化学プロセス用の報酬関数"""
    
        def __init__(self, weights: Dict[str, float] = None):
            # 各目的の重み（デフォルト値）
            self.weights = weights or {
                'setpoint_tracking': 1.0,    # 設定値追従
                'energy': 0.3,                # エネルギー効率
                'safety': 2.0,                # 安全性
                'stability': 0.5              # 安定性
            }
    
        def compute_reward(self, state: np.ndarray, action: np.ndarray,
                          target_temp: float = 350.0) -> Tuple[float, Dict[str, float]]:
            """総合報酬を計算
    
            Args:
                state: [temperature, pressure, concentration, ...]
                action: [heating_power, flow_rate]
                target_temp: 目標温度
    
            Returns:
                total_reward: 総合報酬
                components: 各成分の報酬の詳細
            """
            temp, pressure = state[0], state[1]
            heating, flow = action[0], action[1] if len(action) > 1 else 0
    
            # 1. 設定値追従報酬（温度）
            temp_error = abs(temp - target_temp)
            r_tracking = -temp_error / 10.0  # -10〜0の範囲
    
            # 2. エネルギー効率報酬
            energy_cost = heating * 0.1 + flow * 0.05  # エネルギーコスト
            r_energy = -energy_cost
    
            # 3. 安全性報酬（ペナルティ）
            r_safety = 0.0
            if temp > 380:  # 高温警告
                r_safety = -10.0 * (temp - 380)
            if temp > 400:  # 危険領域
                r_safety = -100.0
            if pressure > 9:  # 高圧警告
                r_safety += -5.0 * (pressure - 9)
    
            # 4. 安定性報酬（変動の少なさ）
            # 注: 実際は前ステップとの差分を使用
            r_stability = 0.0  # 簡略化のため省略
    
            # 重み付け総和
            components = {
                'tracking': r_tracking * self.weights['setpoint_tracking'],
                'energy': r_energy * self.weights['energy'],
                'safety': r_safety * self.weights['safety'],
                'stability': r_stability * self.weights['stability']
            }
    
            total_reward = sum(components.values())
    
            return total_reward, components
    
        def reward_shaping(self, raw_reward: float, progress: float) -> float:
            """報酬シェーピング（学習初期の探索促進）
    
            Args:
                raw_reward: 元の報酬
                progress: 学習進捗（0-1）
    
            Returns:
                shaped_reward: シェーピング後の報酬
            """
            # 初期はペナルティを緩和
            penalty_scale = 0.3 + 0.7 * progress
            if raw_reward < 0:
                return raw_reward * penalty_scale
            else:
                return raw_reward
    
    
    # ===== 使用例 =====
    print("\n=== Example 3: 多目的報酬関数 ===\n")
    
    reward_func = RewardFunction()
    
    # シナリオ1: 最適状態
    state_optimal = np.array([350.0, 5.0, 1.0])
    action_optimal = np.array([5.0, 2.0])
    
    reward, components = reward_func.compute_reward(state_optimal, action_optimal)
    print("シナリオ1: 最適状態")
    print(f"  状態: T={state_optimal[0]}K, P={state_optimal[1]}bar")
    print(f"  行動: Heating={action_optimal[0]}kW, Flow={action_optimal[1]}L/min")
    print(f"  総合報酬: {reward:.3f}")
    for key, val in components.items():
        print(f"    {key}: {val:.3f}")
    
    # シナリオ2: 高温危険状態
    state_danger = np.array([390.0, 5.0, 1.0])
    action_danger = np.array([8.0, 2.0])
    
    reward, components = reward_func.compute_reward(state_danger, action_danger)
    print("\nシナリオ2: 高温危険状態")
    print(f"  状態: T={state_danger[0]}K, P={state_danger[1]}bar")
    print(f"  総合報酬: {reward:.3f}")
    for key, val in components.items():
        print(f"    {key}: {val:.3f}")
    
    # シナリオ3: エネルギー過剰使用
    state_normal = np.array([345.0, 5.0, 1.0])
    action_waste = np.array([10.0, 5.0])
    
    reward, components = reward_func.compute_reward(state_normal, action_waste)
    print("\nシナリオ3: エネルギー過剰使用")
    print(f"  状態: T={state_normal[0]}K, P={state_normal[1]}bar")
    print(f"  行動: Heating={action_waste[0]}kW, Flow={action_waste[1]}L/min")
    print(f"  総合報酬: {reward:.3f}")
    for key, val in components.items():
        print(f"    {key}: {val:.3f}")
    

**出力例:**  
シナリオ1: 最適状態  
総合報酬: -0.250  
tracking: 0.000  
energy: -0.250  
safety: 0.000  
  
シナリオ2: 高温危険状態  
総合報酬: -204.550  
tracking: -4.000  
energy: -0.550  
safety: -200.000 

**⚠️ 報酬関数設計の注意点**

  * **スケールの統一** : 各成分の報酬スケールを合わせる
  * **スパース報酬の回避** : 適度な中間報酬を与える
  * **報酬ハッキング防止** : 意図しない挙動を生まないか検証

## 学習目標の確認

### 基本理解

  * ✅ 状態空間と行動空間の定義方法を理解している
  * ✅ 報酬関数の設計原則を知っている
  * ✅ OpenAI Gym環境の構造を理解している

### 実践スキル

  * ✅ 状態の正規化・逆正規化を実装できる
  * ✅ 離散・連続・混合行動空間を設計できる
  * ✅ 多目的報酬関数を実装できる
  * ✅ 安全制約を組み込める

### 応用力

  * ✅ CSTR環境をGym準拠で実装できる
  * ✅ 蒸留塔環境をモデル化できる
  * ✅ マルチユニットプロセスを統合できる

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
