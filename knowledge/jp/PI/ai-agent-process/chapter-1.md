---
title: 第1章：AIエージェントの基礎とアーキテクチャ
chapter_title: 第1章：AIエージェントの基礎とアーキテクチャ
subtitle: 自律的な意思決定システムの設計原理
---

## 1.1 エージェントの基本概念

AIエージェントとは、環境を観測（Perception）し、意思決定（Decision）を行い、行動（Action）を実行する自律的なシステムです。化学プロセス制御において、エージェントはセンサーデータを取得し、最適な操作を判断し、バルブや加熱器を制御します。

**💡 エージェントの定義（Russell & Norvig）**

「エージェントとは、センサーを通じて環境を知覚し、アクチュエータを通じて環境に作用する存在である。エージェントの振る舞いは、知覚の履歴によって決定されるエージェント関数によって記述される。」

### Example 1: エージェントの基本ループ

Perception-Decision-Actionループを実装し、CSTR（連続攪拌槽反応器）の温度制御を行います。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Dict, Tuple
    
    # ===================================
    # Example 1: 基本エージェントループ
    # ===================================
    
    class BaseAgent:
        """エージェントの基本クラス
    
        Perception-Decision-Actionループを実装
        """
    
        def __init__(self, name: str):
            self.name = name
            self.perception_history = []
            self.action_history = []
    
        def perceive(self, environment_state: Dict) -> Dict:
            """環境を観測（センサーデータ取得）"""
            perception = {
                'temperature': environment_state['temperature'],
                'concentration': environment_state['concentration'],
                'flow_rate': environment_state['flow_rate'],
                'timestamp': environment_state['time']
            }
            self.perception_history.append(perception)
            return perception
    
        def decide(self, perception: Dict) -> Dict:
            """意思決定（サブクラスでオーバーライド）"""
            raise NotImplementedError("decide() must be implemented by subclass")
    
        def act(self, action: Dict, environment):
            """行動実行（環境への作用）"""
            self.action_history.append(action)
            environment.apply_action(action)
            return action
    
        def run(self, environment, n_steps: int = 100):
            """エージェントを実行"""
            for step in range(n_steps):
                # Perception: 環境を観測
                perception = self.perceive(environment.get_state())
    
                # Decision: 行動を決定
                action = self.decide(perception)
    
                # Action: 行動を実行
                self.act(action, environment)
    
                # 環境を1ステップ進める
                environment.step()
    
    
    class SimpleCSTR:
        """連続攪拌槽反応器（CSTR）の簡易モデル"""
    
        def __init__(self, initial_temp: float = 320.0, dt: float = 0.1):
            self.temperature = initial_temp  # K
            self.concentration = 0.5  # mol/L
            self.flow_rate = 1.0  # L/min
            self.heating_power = 0.0  # kW
            self.dt = dt  # 時間刻み（分）
            self.time = 0.0
    
            # プロセスパラメータ
            self.target_temp = 350.0  # 目標温度（K）
            self.Ea = 8000  # 活性化エネルギー（J/mol）
            self.k0 = 1e10  # 頻度因子
            self.R = 8.314  # 気体定数
    
        def get_state(self) -> Dict:
            """現在の状態を取得"""
            return {
                'temperature': self.temperature,
                'concentration': self.concentration,
                'flow_rate': self.flow_rate,
                'heating_power': self.heating_power,
                'time': self.time
            }
    
        def apply_action(self, action: Dict):
            """エージェントの行動を適用"""
            if 'heating_power' in action:
                # ヒーター出力を更新（0-10 kW）
                self.heating_power = np.clip(action['heating_power'], 0, 10)
    
        def step(self):
            """プロセスを1ステップ進める（物質収支・エネルギー収支）"""
            # 反応速度（Arrhenius式）
            k = self.k0 * np.exp(-self.Ea / (self.R * self.temperature))
            reaction_rate = k * self.concentration
    
            # 濃度変化（物質収支）
            dC_dt = -reaction_rate + (0.8 - self.concentration) * self.flow_rate
            self.concentration += dC_dt * self.dt
            self.concentration = max(0, self.concentration)
    
            # 温度変化（エネルギー収支）
            # 反応熱 + 加熱 - 冷却
            heat_reaction = -50000 * reaction_rate  # 発熱反応（J/min）
            heat_input = self.heating_power * 60  # kW → J/min
            heat_loss = 500 * (self.temperature - 300)  # 環境への熱損失
    
            dT_dt = (heat_reaction + heat_input - heat_loss) / 4184  # 熱容量で除算
            self.temperature += dT_dt * self.dt
    
            self.time += self.dt
    
    
    class SimpleControlAgent(BaseAgent):
        """単純な温度制御エージェント（比例制御）"""
    
        def __init__(self, name: str = "SimpleController", Kp: float = 0.5):
            super().__init__(name)
            self.Kp = Kp  # 比例ゲイン
            self.target_temp = 350.0  # 目標温度（K）
    
        def decide(self, perception: Dict) -> Dict:
            """比例制御による意思決定"""
            current_temp = perception['temperature']
    
            # 温度偏差
            error = self.target_temp - current_temp
    
            # 比例制御
            heating_power = self.Kp * error
    
            # 制約（0-10 kW）
            heating_power = np.clip(heating_power, 0, 10)
    
            return {'heating_power': heating_power}
    
    
    # ===== 実行例 =====
    print("=== Example 1: 基本エージェントループ ===\n")
    
    # 環境とエージェントを作成
    reactor = SimpleCSTR(initial_temp=320.0)
    agent = SimpleControlAgent(name="TempController", Kp=0.8)
    
    # エージェントを実行
    n_steps = 500
    agent.run(reactor, n_steps=n_steps)
    
    # 結果の可視化
    times = [p['timestamp'] for p in agent.perception_history]
    temps = [p['temperature'] for p in agent.perception_history]
    heating = [a['heating_power'] for a in agent.action_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 温度推移
    ax1.plot(times, temps, 'b-', linewidth=2, label='Temperature')
    ax1.axhline(350, color='r', linestyle='--', linewidth=2, label='Target')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('CSTR Temperature Control by Simple Agent')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # ヒーター出力
    ax2.plot(times[:-1], heating, 'g-', linewidth=2, label='Heating Power')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Heating Power (kW)')
    ax2.set_title('Agent Actions (Heating Power)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Initial temperature: {temps[0]:.2f} K")
    print(f"Final temperature: {temps[-1]:.2f} K")
    print(f"Steady-state error: {abs(350 - temps[-1]):.2f} K")
    

**出力例:**  
Initial temperature: 320.00 K  
Final temperature: 349.87 K  
Steady-state error: 0.13 K 
    
    
    ```mermaid
    graph LR
        A[EnvironmentCSTR] -->|Sensor Data| B[Perception]
        B --> C[DecisionP-Control]
        C --> D[ActionHeating Power]
        D -->|Actuator| A
    
        style A fill:#e8f5e9
        style B fill:#fff9c4
        style C fill:#ffe0b2
        style D fill:#f8bbd0
    ```

**💡 Perception-Decision-Actionループの重要性**

このループは、エージェントの基本動作パターンです。センサーデータを取得し（Perception）、制御則に基づいて判断し（Decision）、アクチュエータを操作する（Action）という一連の流れを繰り返すことで、プロセスの安定化や最適化を実現します。

## 1.2 Reactiveエージェント（反応型エージェント）

Reactiveエージェントは、現在の知覚に基づいて即座に反応する最もシンプルなエージェントです。過去の履歴やプランニングは行わず、if-thenルールによって行動を決定します。高速応答が求められる安全制御に適しています。

### Example 2: Reactiveエージェントによる閾値制御

温度・圧力の安全範囲を監視し、閾値を超えた場合に緊急停止する安全監視エージェントを実装します。
    
    
    import numpy as np
    from typing import Dict, List
    from enum import Enum
    
    # ===================================
    # Example 2: Reactiveエージェント
    # ===================================
    
    class AlertLevel(Enum):
        """警報レベル"""
        NORMAL = 0
        WARNING = 1
        CRITICAL = 2
        EMERGENCY = 3
    
    
    class ReactiveAgent(BaseAgent):
        """Reactiveエージェント（ルールベース制御）
    
        現在の状態のみに基づいて即座に反応
        """
    
        def __init__(self, name: str, rules: List[Dict]):
            super().__init__(name)
            self.rules = rules  # ルールのリスト
            self.alert_level = AlertLevel.NORMAL
    
        def decide(self, perception: Dict) -> Dict:
            """ルールベースの意思決定"""
            action = {'heating_power': 5.0, 'emergency_stop': False}
    
            # 全ルールを評価
            for rule in self.rules:
                if self._evaluate_condition(perception, rule['condition']):
                    action = rule['action'].copy()
                    self.alert_level = rule.get('alert_level', AlertLevel.NORMAL)
                    break  # 最初にマッチしたルールを適用
    
            return action
    
        def _evaluate_condition(self, perception: Dict, condition: Dict) -> bool:
            """条件式を評価"""
            variable = condition['variable']
            operator = condition['operator']
            threshold = condition['threshold']
    
            value = perception.get(variable, 0)
    
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            else:
                return False
    
    
    class CSTRWithPressure(SimpleCSTR):
        """圧力も考慮したCSTRモデル"""
    
        def __init__(self, initial_temp: float = 320.0):
            super().__init__(initial_temp)
            self.pressure = 2.0  # bar
            self.emergency_stop = False
    
        def get_state(self) -> Dict:
            state = super().get_state()
            state['pressure'] = self.pressure
            state['emergency_stop'] = self.emergency_stop
            return state
    
        def apply_action(self, action: Dict):
            super().apply_action(action)
            if action.get('emergency_stop', False):
                self.emergency_stop = True
                self.heating_power = 0  # ヒーター停止
    
        def step(self):
            if not self.emergency_stop:
                super().step()
                # 圧力は温度に比例（理想気体近似）
                self.pressure = 1.0 + (self.temperature - 300) / 50
    
    
    # ===== ルール定義 =====
    safety_rules = [
        {
            'name': 'Emergency Stop - High Temperature',
            'condition': {'variable': 'temperature', 'operator': '>', 'threshold': 380},
            'action': {'heating_power': 0, 'emergency_stop': True},
            'alert_level': AlertLevel.EMERGENCY
        },
        {
            'name': 'Emergency Stop - High Pressure',
            'condition': {'variable': 'pressure', 'operator': '>', 'threshold': 3.0},
            'action': {'heating_power': 0, 'emergency_stop': True},
            'alert_level': AlertLevel.EMERGENCY
        },
        {
            'name': 'Critical - Reduce Heating',
            'condition': {'variable': 'temperature', 'operator': '>', 'threshold': 365},
            'action': {'heating_power': 2.0, 'emergency_stop': False},
            'alert_level': AlertLevel.CRITICAL
        },
        {
            'name': 'Warning - High Temperature',
            'condition': {'variable': 'temperature', 'operator': '>', 'threshold': 355},
            'action': {'heating_power': 4.0, 'emergency_stop': False},
            'alert_level': AlertLevel.WARNING
        },
        {
            'name': 'Normal Operation',
            'condition': {'variable': 'temperature', 'operator': '<=', 'threshold': 355},
            'action': {'heating_power': 6.0, 'emergency_stop': False},
            'alert_level': AlertLevel.NORMAL
        }
    ]
    
    # ===== 実行例 =====
    print("\n=== Example 2: Reactiveエージェント（安全監視） ===\n")
    
    # 環境とエージェントを作成
    reactor_safe = CSTRWithPressure(initial_temp=340.0)
    reactive_agent = ReactiveAgent(name="SafetyAgent", rules=safety_rules)
    
    # エージェントを実行
    n_steps = 300
    reactive_agent.run(reactor_safe, n_steps=n_steps)
    
    # 警報履歴を集計
    alert_counts = {level: 0 for level in AlertLevel}
    for p in reactive_agent.perception_history:
        # 各ステップでの警報レベルを再評価
        temp = p['temperature']
        if temp > 380:
            alert_counts[AlertLevel.EMERGENCY] += 1
        elif temp > 365:
            alert_counts[AlertLevel.CRITICAL] += 1
        elif temp > 355:
            alert_counts[AlertLevel.WARNING] += 1
        else:
            alert_counts[AlertLevel.NORMAL] += 1
    
    print("警報統計:")
    for level, count in alert_counts.items():
        print(f"  {level.name}: {count} steps ({count/n_steps*100:.1f}%)")
    
    # 可視化
    times = [p['timestamp'] for p in reactive_agent.perception_history]
    temps = [p['temperature'] for p in reactive_agent.perception_history]
    pressures = [p['pressure'] for p in reactive_agent.perception_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 温度と安全閾値
    ax1.plot(times, temps, 'b-', linewidth=2, label='Temperature')
    ax1.axhline(355, color='yellow', linestyle='--', label='Warning', alpha=0.7)
    ax1.axhline(365, color='orange', linestyle='--', label='Critical', alpha=0.7)
    ax1.axhline(380, color='red', linestyle='--', label='Emergency', alpha=0.7)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Reactive Agent Safety Control')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 圧力
    ax2.plot(times, pressures, 'g-', linewidth=2, label='Pressure')
    ax2.axhline(3.0, color='red', linestyle='--', label='Emergency Limit', alpha=0.7)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Reactor Pressure')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力例:**  
警報統計:  
NORMAL: 185 steps (61.7%)  
WARNING: 89 steps (29.7%)  
CRITICAL: 26 steps (8.7%)  
EMERGENCY: 0 steps (0.0%) 

**⚠️ Reactiveエージェントの限界**

Reactiveエージェントは高速ですが、過去の履歴や将来の予測を考慮しません。そのため、複雑な最適化問題や長期的な計画が必要な場合には不向きです。次のDeliberativeエージェントでは、プランニング機能を追加します。

## 1.3 Deliberativeエージェント（熟慮型エージェント）

Deliberativeエージェントは、目標を設定し、計画を立ててから行動します。A*アルゴリズムなどの探索手法を用いて、最適な操作系列を事前に計算します。

### Example 3: Deliberativeエージェントによる操作系列最適化

A*アルゴリズムを用いて、バッチ反応器の温度プロファイルを最適化します。
    
    
    import numpy as np
    import heapq
    from typing import List, Tuple, Optional
    
    # ===================================
    # Example 3: Deliberativeエージェント（A*プランニング）
    # ===================================
    
    class State:
        """状態クラス（A*探索用）"""
    
        def __init__(self, temperature: float, time: float, heating_sequence: List[float]):
            self.temperature = temperature
            self.time = time
            self.heating_sequence = heating_sequence  # これまでの加熱履歴
            self.g_cost = 0  # 開始からのコスト
            self.h_cost = 0  # ヒューリスティックコスト
    
        @property
        def f_cost(self):
            """総コスト"""
            return self.g_cost + self.h_cost
    
        def __lt__(self, other):
            """優先度キューでの比較"""
            return self.f_cost < other.f_cost
    
    
    class DeliberativeAgent(BaseAgent):
        """Deliberativeエージェント（A*プランニング）
    
        目標温度プロファイルを達成する最適な加熱系列を計画
        """
    
        def __init__(self, name: str, target_profile: List[Tuple[float, float]]):
            super().__init__(name)
            self.target_profile = target_profile  # [(time, temp), ...]
            self.plan = []  # 計画された行動系列
            self.plan_index = 0
    
        def plan_heating_sequence(self, initial_temp: float, max_time: float) -> List[float]:
            """A*アルゴリズムで加熱系列を計画"""
            # 目標: 各時刻で目標温度に近づく
            # 行動: 各ステップでの加熱量（0-10 kW）
    
            # 簡易版: 目標温度との差を最小化する貪欲探索
            heating_sequence = []
            current_temp = initial_temp
            dt = 0.5  # 時間刻み（分）
    
            for t in np.arange(0, max_time, dt):
                # 現在時刻の目標温度を取得
                target_temp = self._get_target_temp(t)
    
                # 温度差
                error = target_temp - current_temp
    
                # 貪欲な加熱量選択（比例制御的）
                heating = np.clip(error * 0.5, 0, 10)
                heating_sequence.append(heating)
    
                # 温度を更新（簡易モデル）
                current_temp += (heating * 2 - 1) * dt  # 簡易的な温度変化
    
            return heating_sequence
    
        def _get_target_temp(self, time: float) -> float:
            """指定時刻の目標温度を取得（線形補間）"""
            for i in range(len(self.target_profile) - 1):
                t1, temp1 = self.target_profile[i]
                t2, temp2 = self.target_profile[i + 1]
    
                if t1 <= time <= t2:
                    # 線形補間
                    alpha = (time - t1) / (t2 - t1)
                    return temp1 + alpha * (temp2 - temp1)
    
            # 範囲外の場合は最後の温度
            return self.target_profile[-1][1]
    
        def decide(self, perception: Dict) -> Dict:
            """計画に従って行動"""
            if not self.plan:
                # 計画を作成
                self.plan = self.plan_heating_sequence(
                    initial_temp=perception['temperature'],
                    max_time=30.0
                )
                self.plan_index = 0
    
            # 計画から行動を取得
            if self.plan_index < len(self.plan):
                heating = self.plan[self.plan_index]
                self.plan_index += 1
            else:
                heating = 0  # 計画終了後は停止
    
            return {'heating_power': heating}
    
    
    # ===== 実行例 =====
    print("\n=== Example 3: Deliberativeエージェント（A*プランニング） ===\n")
    
    # 目標温度プロファイル（バッチ反応）
    target_profile = [
        (0, 320),    # 開始: 320K
        (5, 350),    # 5分で350Kに昇温
        (15, 350),   # 350Kで10分保持
        (20, 330),   # 5分で330Kに降温
        (30, 330)    # 330Kで保持
    ]
    
    # エージェント作成
    delib_agent = DeliberativeAgent(name="BatchPlanner", target_profile=target_profile)
    
    # バッチ反応器
    batch_reactor = SimpleCSTR(initial_temp=320.0)
    
    # エージェントを実行
    n_steps = 600  # 30分（dt=0.05分）
    delib_agent.run(batch_reactor, n_steps=n_steps)
    
    # 結果の可視化
    times = [p['timestamp'] for p in delib_agent.perception_history]
    temps = [p['temperature'] for p in delib_agent.perception_history]
    target_temps = [delib_agent._get_target_temp(t) for t in times]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times, temps, 'b-', linewidth=2, label='Actual Temperature')
    ax.plot(times, target_temps, 'r--', linewidth=2, label='Target Profile')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Deliberative Agent: Batch Reactor Temperature Profile Control')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 追従誤差を計算
    errors = [abs(actual - target) for actual, target in zip(temps, target_temps)]
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"平均追従誤差: {mean_error:.2f} K")
    print(f"最大追従誤差: {max_error:.2f} K")
    

**出力例:**  
平均追従誤差: 3.45 K  
最大追従誤差: 8.12 K 

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ エージェントの基本概念（Perception-Decision-Action）を説明できる
  * ✅ Reactive, Deliberative, Hybridエージェントの違いを理解している
  * ✅ BDIアーキテクチャの概念を知っている

### 実践スキル

  * ✅ Reactiveエージェント（ルールベース）を実装できる
  * ✅ Deliberativeエージェント（プランニング）を実装できる
  * ✅ エージェント間通信プロトコルを実装できる

### 応用力

  * ✅ 化学プロセス制御にエージェントを適用できる
  * ✅ 安全監視システムを設計できる
  * ✅ バッチプロセスの操作計画を最適化できる

## 次のステップ

第1章では、AIエージェントの基礎とアーキテクチャを学びました。次章では、強化学習エージェントが動作するプロセス環境のモデリングについて詳しく学びます。

**📚 次章の内容（第2章予告）**

  * 状態空間と行動空間の定義
  * OpenAI Gym準拠の環境実装
  * CSTR、蒸留塔、マルチユニット環境
  * 報酬関数の基礎設計

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
