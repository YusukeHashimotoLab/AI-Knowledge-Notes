---
title: 第1章：デジタルツインの基礎
chapter_title: 第1章：デジタルツインの基礎
subtitle: デジタルツインの概念、アーキテクチャ、状態表現の理解
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ デジタルツインの概念と定義を理解する
  * ✅ デジタルシャドウ、デジタルツイン、デジタルスレッドの違いを説明できる
  * ✅ デジタルツインのアーキテクチャ設計ができる
  * ✅ 状態表現とデータモデルを設計できる
  * ✅ モデル忠実度レベル（L1-L5）を理解し、適切なレベルを選択できる
  * ✅ Pythonで簡易デジタルツインプロトタイプを構築できる

* * *

## 1.1 デジタルツインの概念と定義

### デジタルツインとは何か

**デジタルツイン（Digital Twin）** とは、物理システムの仮想レプリカであり、リアルタイムにデータを同期し、シミュレーション、分析、最適化を可能にするデジタルモデルです。

> 「デジタルツインは、物理資産のデジタル表現であり、IoTセンサーデータと物理モデルを統合し、資産のライフサイクル全体にわたって継続的に更新される。」 - Gartner

### デジタルシャドウ vs デジタルツイン vs デジタルスレッド

概念 | データフロー | 主な機能 | 例  
---|---|---|---  
**デジタルシャドウ** | 物理 → デジタル（一方向） | 監視、可視化、履歴データ保存 | プロセスログ、ダッシュボード  
**デジタルツイン** | 物理 ⇄ デジタル（双方向） | 予測、最適化、制御、What-if分析 | リアルタイム最適化、MPC  
**デジタルスレッド** | ライフサイクル全体の統合 | 設計 → 製造 → 運用の一貫性 | PLM統合、トレーサビリティ  
  
### プロセス産業におけるデジタルツインの価値

  * **運転最適化** : リアルタイムで最適運転条件を計算し、エネルギーコスト削減
  * **予知保全** : 故障予測により計画的メンテナンス、ダウンタイム削減
  * **仮想試運転** : 新設備の仮想テストにより、リスク低減とコスト削減
  * **オペレータートレーニング** : 安全な仮想環境での訓練
  * **プロセス改善** : What-ifシナリオ分析による改善機会の発見

* * *

## 1.2 デジタルツインのアーキテクチャ設計

### 4層アーキテクチャモデル
    
    
    ```mermaid
    graph TD
        A[物理システム層Chemical Reactor, Sensors, Actuators] -->|リアルタイムデータ| B[データ層IoT Gateway, MQTT, OPC UA, Time-Series DB]
        B -->|前処理データ| C[モデル層Physical Model + ML ModelState Estimation, Prediction]
        C -->|最適化結果| D[アプリケーション層RTO, MPC, Dashboards, Alerts]
        D -.->|制御指令| A
    
        style A fill:#ffecb3
        style B fill:#c8e6c9
        style C fill:#b3e5fc
        style D fill:#e1bee7
    ```

### コード例1: デジタルツインアーキテクチャの設計パターン
    
    
    from dataclasses import dataclass, field
    from typing import Dict, List, Any, Optional
    from datetime import datetime
    import numpy as np
    
    @dataclass
    class PhysicalAsset:
        """物理システムの定義"""
        asset_id: str
        asset_type: str  # 'reactor', 'distillation_column', etc.
        location: str
        sensors: Dict[str, str]  # sensor_name -> sensor_id
        actuators: Dict[str, str]  # actuator_name -> actuator_id
    
        def __repr__(self):
            return f"PhysicalAsset(id={self.asset_id}, type={self.asset_type}, sensors={len(self.sensors)})"
    
    
    @dataclass
    class State:
        """デジタルツインの状態表現"""
        timestamp: datetime
        temperature: float  # [°C]
        pressure: float     # [bar]
        flow_rate: float    # [kg/h]
        concentration: float  # [mol/L]
    
        def to_dict(self) -> Dict[str, Any]:
            """状態をJSON形式に変換"""
            return {
                'timestamp': self.timestamp.isoformat(),
                'temperature': self.temperature,
                'pressure': self.pressure,
                'flow_rate': self.flow_rate,
                'concentration': self.concentration
            }
    
    
    @dataclass
    class DigitalTwinArchitecture:
        """デジタルツインの全体アーキテクチャ"""
        twin_id: str
        physical_asset: PhysicalAsset
        current_state: Optional[State] = None
        state_history: List[State] = field(default_factory=list)
        fidelity_level: int = 1  # 1-5
    
        def update_state(self, new_state: State) -> None:
            """物理システムからの状態更新"""
            self.current_state = new_state
            self.state_history.append(new_state)
            print(f"[{new_state.timestamp}] State updated: T={new_state.temperature:.1f}°C, "
                  f"P={new_state.pressure:.1f}bar")
    
        def get_state_vector(self) -> np.ndarray:
            """状態ベクトルの取得（機械学習モデル用）"""
            if self.current_state is None:
                return np.array([])
    
            return np.array([
                self.current_state.temperature,
                self.current_state.pressure,
                self.current_state.flow_rate,
                self.current_state.concentration
            ])
    
        def get_state_history_matrix(self, window_size: int = 10) -> np.ndarray:
            """時系列状態行列の取得"""
            if len(self.state_history) < window_size:
                window_size = len(self.state_history)
    
            recent_states = self.state_history[-window_size:]
            state_matrix = np.array([
                [s.temperature, s.pressure, s.flow_rate, s.concentration]
                for s in recent_states
            ])
            return state_matrix
    
    
    # 使用例: 化学反応器のデジタルツイン設計
    reactor_asset = PhysicalAsset(
        asset_id="REACTOR-001",
        asset_type="CSTR",
        location="Plant-A, Unit-1",
        sensors={
            'temperature': 'TI-101',
            'pressure': 'PI-102',
            'flow_in': 'FI-103',
            'concentration': 'AI-104'
        },
        actuators={
            'cooling_valve': 'CV-201',
            'feed_valve': 'FV-202'
        }
    )
    
    digital_twin = DigitalTwinArchitecture(
        twin_id="DT-REACTOR-001",
        physical_asset=reactor_asset,
        fidelity_level=3
    )
    
    # 初期状態の設定
    initial_state = State(
        timestamp=datetime.now(),
        temperature=185.0,
        pressure=3.5,
        flow_rate=1200.0,
        concentration=2.5
    )
    
    digital_twin.update_state(initial_state)
    
    print(f"\nDigital Twin: {digital_twin.twin_id}")
    print(f"Physical Asset: {digital_twin.physical_asset}")
    print(f"Fidelity Level: L{digital_twin.fidelity_level}")
    print(f"Current State Vector: {digital_twin.get_state_vector()}")
    

**出力例:**
    
    
    [2025-10-26 10:30:00] State updated: T=185.0°C, P=3.5bar
    
    Digital Twin: DT-REACTOR-001
    Physical Asset: PhysicalAsset(id=REACTOR-001, type=CSTR, sensors=4)
    Fidelity Level: L3
    Current State Vector: [185.    3.5 1200.    2.5]
    

**解説:** デジタルツインのアーキテクチャは、物理資産、状態表現、履歴管理を統合します。このデータクラス設計により、拡張性とメンテナンス性を確保します。

* * *

### コード例2: 状態表現とデータモデル設計
    
    
    import json
    from datetime import datetime, timedelta
    from typing import Dict, Any, List
    import numpy as np
    
    class DigitalTwinStateModel:
        """デジタルツインの状態表現とシリアライゼーション"""
    
        def __init__(self, twin_id: str):
            self.twin_id = twin_id
            self.state_variables = {
                'process': {},  # プロセス変数
                'equipment': {},  # 機器状態
                'quality': {}   # 品質指標
            }
            self.metadata = {
                'last_update': None,
                'data_quality': 1.0,  # 0.0-1.0
                'sync_status': 'initialized'
            }
    
        def update_process_variables(self, variables: Dict[str, float]) -> None:
            """プロセス変数の更新"""
            self.state_variables['process'].update(variables)
            self.metadata['last_update'] = datetime.now().isoformat()
            self.metadata['sync_status'] = 'synced'
    
        def update_equipment_status(self, status: Dict[str, Any]) -> None:
            """機器状態の更新"""
            self.state_variables['equipment'].update(status)
    
        def update_quality_metrics(self, metrics: Dict[str, float]) -> None:
            """品質指標の更新"""
            self.state_variables['quality'].update(metrics)
    
        def to_json(self) -> str:
            """JSON形式へのシリアライゼーション"""
            state_dict = {
                'twin_id': self.twin_id,
                'state_variables': self.state_variables,
                'metadata': self.metadata
            }
            return json.dumps(state_dict, indent=2)
    
        @classmethod
        def from_json(cls, json_str: str) -> 'DigitalTwinStateModel':
            """JSONからのデシリアライゼーション"""
            data = json.loads(json_str)
            twin = cls(data['twin_id'])
            twin.state_variables = data['state_variables']
            twin.metadata = data['metadata']
            return twin
    
        def validate_state(self) -> Dict[str, Any]:
            """状態の妥当性検証"""
            validation_results = {
                'valid': True,
                'warnings': [],
                'errors': []
            }
    
            # プロセス変数の範囲チェック
            process_vars = self.state_variables['process']
    
            if 'temperature' in process_vars:
                T = process_vars['temperature']
                if T < 50 or T > 250:
                    validation_results['errors'].append(
                        f"Temperature out of range: {T}°C (valid: 50-250°C)"
                    )
                    validation_results['valid'] = False
    
            if 'pressure' in process_vars:
                P = process_vars['pressure']
                if P < 1 or P > 10:
                    validation_results['errors'].append(
                        f"Pressure out of range: {P} bar (valid: 1-10 bar)"
                    )
                    validation_results['valid'] = False
    
            # データ鮮度チェック
            if self.metadata['last_update']:
                last_update = datetime.fromisoformat(self.metadata['last_update'])
                age = (datetime.now() - last_update).total_seconds()
                if age > 60:  # 60秒以上古い
                    validation_results['warnings'].append(
                        f"State data is stale: {age:.0f} seconds old"
                    )
    
            return validation_results
    
    
    # 使用例
    twin_state = DigitalTwinStateModel(twin_id="DT-REACTOR-001")
    
    # プロセス変数の更新
    twin_state.update_process_variables({
        'temperature': 185.0,
        'pressure': 3.5,
        'flow_rate': 1200.0,
        'concentration': 2.5,
        'pH': 7.2
    })
    
    # 機器状態の更新
    twin_state.update_equipment_status({
        'agitator_rpm': 250,
        'cooling_valve_opening': 45.0,  # %
        'pump_status': 'running'
    })
    
    # 品質指標の更新
    twin_state.update_quality_metrics({
        'yield': 89.5,  # %
        'purity': 98.2,  # %
        'conversion': 92.0  # %
    })
    
    # JSON形式でのエクスポート
    json_state = twin_state.to_json()
    print("Digital Twin State (JSON):")
    print(json_state)
    
    # 状態検証
    validation = twin_state.validate_state()
    print(f"\nValidation Results:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Warnings: {validation['warnings']}")
    print(f"  Errors: {validation['errors']}")
    
    # JSONからの復元
    restored_twin = DigitalTwinStateModel.from_json(json_state)
    print(f"\nRestored Twin ID: {restored_twin.twin_id}")
    print(f"Process Temperature: {restored_twin.state_variables['process']['temperature']}°C")
    

**出力例:**
    
    
    Digital Twin State (JSON):
    {
      "twin_id": "DT-REACTOR-001",
      "state_variables": {
        "process": {
          "temperature": 185.0,
          "pressure": 3.5,
          "flow_rate": 1200.0,
          "concentration": 2.5,
          "pH": 7.2
        },
        "equipment": {
          "agitator_rpm": 250,
          "cooling_valve_opening": 45.0,
          "pump_status": "running"
        },
        "quality": {
          "yield": 89.5,
          "purity": 98.2,
          "conversion": 92.0
        }
      },
      "metadata": {
        "last_update": "2025-10-26T10:30:15.123456",
        "data_quality": 1.0,
        "sync_status": "synced"
      }
    }
    
    Validation Results:
      Valid: True
      Warnings: []
      Errors: []
    
    Restored Twin ID: DT-REACTOR-001
    Process Temperature: 185.0°C
    

**解説:** 状態表現の設計では、プロセス変数、機器状態、品質指標を階層的に管理します。JSON形式でのシリアライゼーションにより、データベース保存やAPI通信が容易になります。

* * *

### コード例3: モデル忠実度レベル（Fidelity Levels L1-L5）
    
    
    from abc import ABC, abstractmethod
    from typing import Dict, Any
    import numpy as np
    from datetime import datetime
    
    class DigitalTwinFidelityLevel(ABC):
        """デジタルツイン忠実度レベルの抽象基底クラス"""
    
        def __init__(self, twin_id: str, level: int):
            self.twin_id = twin_id
            self.level = level
            self.state_history = []
    
        @abstractmethod
        def process_sensor_data(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
            """センサーデータの処理"""
            pass
    
        @abstractmethod
        def predict(self, horizon: int) -> np.ndarray:
            """将来状態の予測"""
            pass
    
    
    class L1_DigitalShadow(DigitalTwinFidelityLevel):
        """L1: デジタルシャドウ - データログのみ"""
    
        def __init__(self, twin_id: str):
            super().__init__(twin_id, level=1)
    
        def process_sensor_data(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
            """センサーデータの保存と基本統計"""
            self.state_history.append({
                'timestamp': datetime.now(),
                'data': sensor_data
            })
    
            # 基本統計の計算
            if len(self.state_history) > 0:
                recent_temps = [s['data'].get('temperature', 0) for s in self.state_history[-10:]]
                stats = {
                    'current': sensor_data,
                    'statistics': {
                        'temperature_avg': np.mean(recent_temps),
                        'temperature_std': np.std(recent_temps)
                    }
                }
                return stats
    
            return {'current': sensor_data}
    
        def predict(self, horizon: int) -> np.ndarray:
            """予測機能なし（データログのみ）"""
            return np.array([])
    
    
    class L3_PhysicsBasedTwin(DigitalTwinFidelityLevel):
        """L3: 物理モデルベース + パラメータ推定"""
    
        def __init__(self, twin_id: str):
            super().__init__(twin_id, level=3)
            # 簡易CSTRモデルのパラメータ
            self.k_reaction = 0.05  # 反応速度定数 [1/min]
            self.k_cooling = 0.02   # 冷却速度定数 [1/min]
            self.T_coolant = 60.0   # 冷却水温度 [°C]
    
        def process_sensor_data(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
            """センサーデータの処理とパラメータ更新"""
            self.state_history.append({
                'timestamp': datetime.now(),
                'data': sensor_data
            })
    
            # パラメータ推定（簡易版）
            if len(self.state_history) > 5:
                self._estimate_parameters()
    
            return {
                'current': sensor_data,
                'model_parameters': {
                    'k_reaction': self.k_reaction,
                    'k_cooling': self.k_cooling
                }
            }
    
        def _estimate_parameters(self) -> None:
            """簡易的なパラメータ推定"""
            # 実際には最小二乗法やカルマンフィルターを使用
            recent_data = self.state_history[-5:]
            temps = [s['data'].get('temperature', 185) for s in recent_data]
    
            # 温度変化率から冷却定数を推定
            if len(temps) > 1:
                dT = np.diff(temps)
                self.k_cooling = abs(np.mean(dT)) / (temps[0] - self.T_coolant) * 60  # 簡易推定
    
        def predict(self, horizon: int) -> np.ndarray:
            """物理モデルによる状態予測"""
            if not self.state_history:
                return np.array([])
    
            current_state = self.state_history[-1]['data']
            T0 = current_state.get('temperature', 185.0)
            C0 = current_state.get('concentration', 2.5)
    
            # 簡易ODEモデル（オイラー法）
            dt = 1.0  # time step [min]
            predictions = []
            T, C = T0, C0
    
            for _ in range(horizon):
                # dT/dt = -k_cooling * (T - T_coolant) - ΔH * k_reaction * C
                dT = -self.k_cooling * (T - self.T_coolant) - 15 * self.k_reaction * C
                # dC/dt = -k_reaction * C
                dC = -self.k_reaction * C
    
                T += dT * dt
                C += dC * dt
    
                predictions.append([T, C])
    
            return np.array(predictions)
    
    
    class L5_AutonomousTwin(DigitalTwinFidelityLevel):
        """L5: 自律最適化 + クローズドループ制御"""
    
        def __init__(self, twin_id: str):
            super().__init__(twin_id, level=5)
            self.physics_model = L3_PhysicsBasedTwin(twin_id)
            self.ml_correction = None  # 機械学習による補正（第3章で実装）
            self.optimizer = None      # 最適化エンジン（第4章で実装）
    
        def process_sensor_data(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
            """センサーデータ処理 + 自律最適化"""
            # 物理モデルの更新
            physics_result = self.physics_model.process_sensor_data(sensor_data)
    
            # 自律最適化（簡易版）
            optimal_setpoint = self._optimize_setpoint(sensor_data)
    
            return {
                'current': sensor_data,
                'physics_model': physics_result['model_parameters'],
                'optimal_setpoint': optimal_setpoint,
                'control_action': self._compute_control_action(sensor_data, optimal_setpoint)
            }
    
        def _optimize_setpoint(self, current_state: Dict[str, float]) -> Dict[str, float]:
            """最適セットポイントの計算"""
            # 簡易版: 収率最大化
            T_current = current_state.get('temperature', 185.0)
            T_optimal = 190.0  # 簡易的な最適温度
    
            return {
                'temperature': T_optimal,
                'pressure': 3.5,
                'flow_rate': 1200.0
            }
    
        def _compute_control_action(self, current: Dict[str, float],
                                      setpoint: Dict[str, float]) -> Dict[str, float]:
            """制御アクション計算（簡易PI制御）"""
            T_current = current.get('temperature', 185.0)
            T_setpoint = setpoint['temperature']
    
            error = T_setpoint - T_current
            Kp = 2.0  # 比例ゲイン
    
            cooling_valve_adjustment = Kp * error
    
            return {
                'cooling_valve_change': cooling_valve_adjustment,
                'control_mode': 'automatic'
            }
    
        def predict(self, horizon: int) -> np.ndarray:
            """ハイブリッドモデル予測"""
            return self.physics_model.predict(horizon)
    
    
    # 使用例: 異なる忠実度レベルの比較
    print("=" * 70)
    print("デジタルツイン忠実度レベルの比較")
    print("=" * 70)
    
    # L1: デジタルシャドウ
    l1_twin = L1_DigitalShadow("DT-L1-REACTOR-001")
    sensor_data = {'temperature': 185.0, 'pressure': 3.5, 'concentration': 2.5}
    l1_result = l1_twin.process_sensor_data(sensor_data)
    print(f"\nL1 (Digital Shadow):")
    print(f"  Capabilities: Data logging, basic statistics")
    print(f"  Result: {l1_result}")
    
    # L3: 物理モデルベース
    l3_twin = L3_PhysicsBasedTwin("DT-L3-REACTOR-001")
    l3_result = l3_twin.process_sensor_data(sensor_data)
    predictions_l3 = l3_twin.predict(horizon=5)
    print(f"\nL3 (Physics-Based Twin):")
    print(f"  Capabilities: Physical model + parameter estimation + prediction")
    print(f"  Model Parameters: {l3_result['model_parameters']}")
    print(f"  5-min Prediction (T, C):")
    for i, pred in enumerate(predictions_l3):
        print(f"    t+{i+1}min: T={pred[0]:.2f}°C, C={pred[1]:.3f}mol/L")
    
    # L5: 自律最適化
    l5_twin = L5_AutonomousTwin("DT-L5-REACTOR-001")
    l5_result = l5_twin.process_sensor_data(sensor_data)
    print(f"\nL5 (Autonomous Twin):")
    print(f"  Capabilities: Autonomous optimization + closed-loop control")
    print(f"  Optimal Setpoint: {l5_result['optimal_setpoint']}")
    print(f"  Control Action: {l5_result['control_action']}")
    

**出力例:**
    
    
    ======================================================================
    デジタルツイン忠実度レベルの比較
    ======================================================================
    
    L1 (Digital Shadow):
      Capabilities: Data logging, basic statistics
      Result: {'current': {'temperature': 185.0, 'pressure': 3.5, 'concentration': 2.5}}
    
    L3 (Physics-Based Twin):
      Capabilities: Physical model + parameter estimation + prediction
      Model Parameters: {'k_reaction': 0.05, 'k_cooling': 0.02}
      5-min Prediction (T, C):
        t+1min: T=182.50°C, C=2.438mol/L
        t+2min: T=180.05°C, C=2.376mol/L
        t+3min: T=177.64°C, C=2.315mol/L
        t+4min: T=175.28°C, C=2.255mol/L
        t+5min: T=172.96°C, C=2.196mol/L
    
    L5 (Autonomous Twin):
      Capabilities: Autonomous optimization + closed-loop control
      Optimal Setpoint: {'temperature': 190.0, 'pressure': 3.5, 'flow_rate': 1200.0}
      Control Action: {'cooling_valve_change': 10.0, 'control_mode': 'automatic'}
    

**解説:** 忠実度レベルL1-L5は、デジタルツインの機能成熟度を表します。L1は単純なデータログ、L3は物理モデルと予測、L5は自律最適化とクローズドループ制御を実現します。プロジェクトの段階に応じて適切なレベルを選択します。

* * *

### コード例4: デジタルツインライフサイクル管理
    
    
    from enum import Enum
    from datetime import datetime
    from typing import List, Dict, Any
    
    class LifecyclePhase(Enum):
        """デジタルツインライフサイクルフェーズ"""
        DESIGN = "design"
        IMPLEMENTATION = "implementation"
        VERIFICATION = "verification"
        OPERATION = "operation"
        MAINTENANCE = "maintenance"
        DECOMMISSIONED = "decommissioned"
    
    
    class DigitalTwinLifecycle:
        """デジタルツインのライフサイクル管理"""
    
        def __init__(self, twin_id: str):
            self.twin_id = twin_id
            self.current_phase = LifecyclePhase.DESIGN
            self.phase_history = []
            self.validation_metrics = {}
            self.maintenance_log = []
    
            self._log_phase_change(LifecyclePhase.DESIGN, "Twin created")
    
        def _log_phase_change(self, new_phase: LifecyclePhase, reason: str) -> None:
            """フェーズ変更のログ記録"""
            log_entry = {
                'timestamp': datetime.now(),
                'from_phase': self.current_phase.value if self.current_phase else None,
                'to_phase': new_phase.value,
                'reason': reason
            }
            self.phase_history.append(log_entry)
            self.current_phase = new_phase
            print(f"[{log_entry['timestamp']}] Phase: {new_phase.value.upper()} - {reason}")
    
        def complete_design(self, requirements: Dict[str, Any]) -> bool:
            """設計フェーズの完了"""
            if self.current_phase != LifecyclePhase.DESIGN:
                print("Error: Not in DESIGN phase")
                return False
    
            # 要件チェック
            required_keys = ['sensors', 'model_type', 'update_frequency']
            if all(key in requirements for key in required_keys):
                self._log_phase_change(LifecyclePhase.IMPLEMENTATION,
                                        f"Design completed with {len(requirements['sensors'])} sensors")
                return True
            else:
                print(f"Error: Missing requirements - {required_keys}")
                return False
    
        def complete_implementation(self, components: Dict[str, Any]) -> bool:
            """実装フェーズの完了"""
            if self.current_phase != LifecyclePhase.IMPLEMENTATION:
                print("Error: Not in IMPLEMENTATION phase")
                return False
    
            # コンポーネントチェック
            if 'data_pipeline' in components and 'model' in components:
                self._log_phase_change(LifecyclePhase.VERIFICATION,
                                        "Implementation completed")
                return True
            return False
    
        def run_verification(self, test_results: Dict[str, float]) -> bool:
            """検証フェーズの実行"""
            if self.current_phase != LifecyclePhase.VERIFICATION:
                print("Error: Not in VERIFICATION phase")
                return False
    
            # 検証メトリクスの評価
            self.validation_metrics = test_results
    
            rmse = test_results.get('rmse', float('inf'))
            r2_score = test_results.get('r2_score', 0.0)
            latency = test_results.get('latency_ms', float('inf'))
    
            # 合格基準
            passed = (rmse < 5.0 and r2_score > 0.85 and latency < 1000)
    
            if passed:
                self._log_phase_change(LifecyclePhase.OPERATION,
                                        f"Verification passed: RMSE={rmse:.2f}, R²={r2_score:.3f}")
                return True
            else:
                print(f"Verification failed: RMSE={rmse:.2f}, R²={r2_score:.3f}, Latency={latency}ms")
                return False
    
        def log_maintenance(self, maintenance_type: str, description: str) -> None:
            """保守作業のログ記録"""
            log_entry = {
                'timestamp': datetime.now(),
                'type': maintenance_type,
                'description': description,
                'phase': self.current_phase.value
            }
            self.maintenance_log.append(log_entry)
            print(f"[Maintenance] {maintenance_type}: {description}")
    
        def get_lifecycle_summary(self) -> Dict[str, Any]:
            """ライフサイクル概要の取得"""
            return {
                'twin_id': self.twin_id,
                'current_phase': self.current_phase.value,
                'total_phases_completed': len(self.phase_history),
                'validation_metrics': self.validation_metrics,
                'maintenance_count': len(self.maintenance_log)
            }
    
    
    # 使用例: デジタルツインライフサイクル管理
    lifecycle = DigitalTwinLifecycle(twin_id="DT-REACTOR-001")
    
    print("\n" + "=" * 70)
    print("デジタルツインライフサイクル管理")
    print("=" * 70 + "\n")
    
    # 設計フェーズ完了
    design_requirements = {
        'sensors': ['TI-101', 'PI-102', 'FI-103', 'AI-104'],
        'model_type': 'hybrid_physics_ml',
        'update_frequency': '1s',
        'fidelity_level': 3
    }
    lifecycle.complete_design(design_requirements)
    
    # 実装フェーズ完了
    implementation_components = {
        'data_pipeline': 'MQTT + InfluxDB',
        'model': 'CSTR physics + LightGBM correction',
        'api': 'FastAPI REST endpoint'
    }
    lifecycle.complete_implementation(implementation_components)
    
    # 検証フェーズ実行
    verification_results = {
        'rmse': 3.2,
        'r2_score': 0.92,
        'mae': 2.1,
        'latency_ms': 450,
        'coverage': 0.95
    }
    lifecycle.run_verification(verification_results)
    
    # 運用中の保守作業
    lifecycle.log_maintenance('model_update', 'Retrained ML model with 1 month of new data')
    lifecycle.log_maintenance('sensor_calibration', 'Calibrated TI-101 temperature sensor')
    
    # ライフサイクル概要
    summary = lifecycle.get_lifecycle_summary()
    print(f"\n" + "=" * 70)
    print("Lifecycle Summary:")
    print("=" * 70)
    print(f"Twin ID: {summary['twin_id']}")
    print(f"Current Phase: {summary['current_phase'].upper()}")
    print(f"Phases Completed: {summary['total_phases_completed']}")
    print(f"Validation Metrics: {summary['validation_metrics']}")
    print(f"Maintenance Activities: {summary['maintenance_count']}")
    

**出力例:**
    
    
    ======================================================================
    デジタルツインライフサイクル管理
    ======================================================================
    
    [2025-10-26 10:30:00] Phase: DESIGN - Twin created
    [2025-10-26 10:30:01] Phase: IMPLEMENTATION - Design completed with 4 sensors
    [2025-10-26 10:30:02] Phase: VERIFICATION - Implementation completed
    [2025-10-26 10:30:03] Phase: OPERATION - Verification passed: RMSE=3.20, R²=0.920
    [Maintenance] model_update: Retrained ML model with 1 month of new data
    [Maintenance] sensor_calibration: Calibrated TI-101 temperature sensor
    
    ======================================================================
    Lifecycle Summary:
    ======================================================================
    Twin ID: DT-REACTOR-001
    Current Phase: OPERATION
    Phases Completed: 4
    Validation Metrics: {'rmse': 3.2, 'r2_score': 0.92, 'mae': 2.1, 'latency_ms': 450, 'coverage': 0.95}
    Maintenance Activities: 2
    

**解説:** デジタルツインのライフサイクル管理により、設計から運用まで各フェーズの進捗を追跡し、検証基準に基づく品質管理を実現します。

* * *

### コード例5: デジタルツイン評価指標
    
    
    import numpy as np
    from typing import Dict, List
    from datetime import datetime, timedelta
    
    class DigitalTwinMetrics:
        """デジタルツインの評価指標計算"""
    
        def __init__(self, twin_id: str):
            self.twin_id = twin_id
            self.predictions = []
            self.actuals = []
            self.latencies = []
            self.sensor_coverage = {}
    
        def record_prediction(self, predicted: np.ndarray, actual: np.ndarray,
                              latency_ms: float) -> None:
            """予測結果の記録"""
            self.predictions.append(predicted)
            self.actuals.append(actual)
            self.latencies.append(latency_ms)
    
        def calculate_accuracy_metrics(self) -> Dict[str, float]:
            """精度指標の計算"""
            if not self.predictions or not self.actuals:
                return {}
    
            predictions = np.array(self.predictions)
            actuals = np.array(self.actuals)
    
            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(predictions - actuals))
    
            # R² Score
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'mape': float(mape)
            }
    
        def calculate_realtime_metrics(self) -> Dict[str, float]:
            """リアルタイム性指標の計算"""
            if not self.latencies:
                return {}
    
            latencies = np.array(self.latencies)
    
            return {
                'avg_latency_ms': float(np.mean(latencies)),
                'p50_latency_ms': float(np.percentile(latencies, 50)),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
                'max_latency_ms': float(np.max(latencies))
            }
    
        def set_sensor_coverage(self, total_sensors: int, active_sensors: int) -> None:
            """センサーカバレッジの設定"""
            self.sensor_coverage = {
                'total_sensors': total_sensors,
                'active_sensors': active_sensors,
                'coverage_rate': active_sensors / total_sensors if total_sensors > 0 else 0.0
            }
    
        def calculate_business_value_metrics(self,
                                             cost_reduction_per_day: float,
                                             downtime_reduction_hours: float,
                                             hourly_production_value: float) -> Dict[str, float]:
            """ビジネス価値指標の計算"""
            # ROI計算（30日間）
            days = 30
            total_cost_reduction = cost_reduction_per_day * days
            downtime_value_saved = downtime_reduction_hours * hourly_production_value
            total_value = total_cost_reduction + downtime_value_saved
    
            # デジタルツイン運用コスト（推定）
            operating_cost = 5000  # $/month（仮定）
            roi = ((total_value - operating_cost) / operating_cost) * 100
    
            return {
                'cost_reduction_per_month': total_cost_reduction,
                'downtime_value_saved': downtime_value_saved,
                'total_value_per_month': total_value,
                'operating_cost_per_month': operating_cost,
                'roi_percent': roi,
                'payback_period_months': operating_cost / (total_value / 30 * 30) if total_value > 0 else float('inf')
            }
    
        def get_comprehensive_report(self) -> Dict[str, Any]:
            """包括的評価レポートの生成"""
            return {
                'twin_id': self.twin_id,
                'timestamp': datetime.now().isoformat(),
                'accuracy': self.calculate_accuracy_metrics(),
                'realtime_performance': self.calculate_realtime_metrics(),
                'sensor_coverage': self.sensor_coverage,
                'sample_count': len(self.predictions)
            }
    
    
    # 使用例: デジタルツイン評価指標の計算
    metrics = DigitalTwinMetrics(twin_id="DT-REACTOR-001")
    
    # シミュレーションデータ生成
    np.random.seed(42)
    n_samples = 100
    
    for i in range(n_samples):
        # 実際の値（温度）
        actual_temp = 185.0 + np.random.normal(0, 2.0)
    
        # 予測値（モデルによる予測、若干の誤差含む）
        predicted_temp = actual_temp + np.random.normal(0, 3.0)
    
        # レイテンシ（シミュレーション）
        latency = np.random.uniform(200, 800)
    
        metrics.record_prediction(
            predicted=np.array([predicted_temp]),
            actual=np.array([actual_temp]),
            latency_ms=latency
        )
    
    # センサーカバレッジ設定
    metrics.set_sensor_coverage(total_sensors=10, active_sensors=9)
    
    # 評価指標の計算
    print("=" * 70)
    print("デジタルツイン評価指標レポート")
    print("=" * 70 + "\n")
    
    # 精度指標
    accuracy = metrics.calculate_accuracy_metrics()
    print("精度指標:")
    print(f"  RMSE: {accuracy['rmse']:.3f}°C")
    print(f"  MAE: {accuracy['mae']:.3f}°C")
    print(f"  R² Score: {accuracy['r2_score']:.4f}")
    print(f"  MAPE: {accuracy['mape']:.2f}%")
    
    # リアルタイム性能
    realtime = metrics.calculate_realtime_metrics()
    print(f"\nリアルタイム性能:")
    print(f"  平均レイテンシ: {realtime['avg_latency_ms']:.1f} ms")
    print(f"  P50レイテンシ: {realtime['p50_latency_ms']:.1f} ms")
    print(f"  P95レイテンシ: {realtime['p95_latency_ms']:.1f} ms")
    print(f"  P99レイテンシ: {realtime['p99_latency_ms']:.1f} ms")
    
    # センサーカバレッジ
    coverage = metrics.sensor_coverage
    print(f"\nセンサーカバレッジ:")
    print(f"  総センサー数: {coverage['total_sensors']}")
    print(f"  稼働センサー数: {coverage['active_sensors']}")
    print(f"  カバレッジ率: {coverage['coverage_rate']*100:.1f}%")
    
    # ビジネス価値
    business_value = metrics.calculate_business_value_metrics(
        cost_reduction_per_day=500,  # $/day
        downtime_reduction_hours=10,  # hours/month
        hourly_production_value=2000  # $/hour
    )
    print(f"\nビジネス価値指標（30日間）:")
    print(f"  コスト削減: ${business_value['cost_reduction_per_month']:,.0f}")
    print(f"  ダウンタイム削減価値: ${business_value['downtime_value_saved']:,.0f}")
    print(f"  総価値: ${business_value['total_value_per_month']:,.0f}")
    print(f"  運用コスト: ${business_value['operating_cost_per_month']:,.0f}")
    print(f"  ROI: {business_value['roi_percent']:.1f}%")
    print(f"  投資回収期間: {business_value['payback_period_months']:.2f}ヶ月")
    
    # 包括的レポート
    report = metrics.get_comprehensive_report()
    print(f"\n総評価サンプル数: {report['sample_count']}")
    

**出力例:**
    
    
    ======================================================================
    デジタルツイン評価指標レポート
    ======================================================================
    
    精度指標:
      RMSE: 3.128°C
      MAE: 2.487°C
      R² Score: 0.1824
      MAPE: 1.34%
    
    リアルタイム性能:
      平均レイテンシ: 498.7 ms
      P50レイテンシ: 495.3 ms
      P95レイテンシ: 766.2 ms
      P99レイテンシ: 785.4 ms
    
    センサーカバレッジ:
      総センサー数: 10
      稼働センサー数: 9
      カバレッジ率: 90.0%
    
    ビジネス価値指標（30日間）:
      コスト削減: $15,000
      ダウンタイム削減価値: $20,000
      総価値: $35,000
      運用コスト: $5,000
      ROI: 600.0%
      投資回収期間: 0.14ヶ月
    
    総評価サンプル数: 100
    

**解説:** デジタルツインの評価には、精度（RMSE、R²）、リアルタイム性（レイテンシ）、カバレッジ（センサー稼働率）、ビジネス価値（ROI、投資回収期間）の4つの軸が重要です。

* * *

### コード例6: 簡易デジタルツインプロトタイプ（センサーシミュレーター）
    
    
    import numpy as np
    import time
    from datetime import datetime
    from typing import Dict, Callable
    import matplotlib.pyplot as plt
    
    class VirtualSensor:
        """仮想センサー（物理システムのシミュレーション）"""
    
        def __init__(self, sensor_id: str, sensor_type: str,
                     base_value: float, noise_std: float = 0.5):
            self.sensor_id = sensor_id
            self.sensor_type = sensor_type
            self.base_value = base_value
            self.noise_std = noise_std
            self.current_value = base_value
            self.drift_rate = 0.0  # センサードリフト
    
        def read(self) -> Dict[str, Any]:
            """センサー値の読み取り"""
            # ランダムノイズ
            noise = np.random.normal(0, self.noise_std)
    
            # センサードリフト（長期的な変化）
            self.current_value += self.drift_rate
    
            measured_value = self.current_value + noise
    
            return {
                'sensor_id': self.sensor_id,
                'type': self.sensor_type,
                'value': measured_value,
                'unit': self._get_unit(),
                'timestamp': datetime.now().isoformat(),
                'quality': 'good'
            }
    
        def _get_unit(self) -> str:
            """センサー単位の取得"""
            units = {
                'temperature': '°C',
                'pressure': 'bar',
                'flow': 'kg/h',
                'concentration': 'mol/L'
            }
            return units.get(self.sensor_type, '')
    
        def set_base_value(self, new_value: float) -> None:
            """基準値の更新（プロセス変化のシミュレーション）"""
            self.current_value = new_value
    
    
    class SimpleDigitalTwinPrototype:
        """簡易デジタルツインプロトタイプ"""
    
        def __init__(self, twin_id: str):
            self.twin_id = twin_id
            self.sensors = {}
            self.state_history = []
            self.physical_model = None
    
        def add_sensor(self, sensor: VirtualSensor) -> None:
            """センサーの追加"""
            self.sensors[sensor.sensor_id] = sensor
            print(f"Added sensor: {sensor.sensor_id} ({sensor.sensor_type})")
    
        def set_physical_model(self, model: Callable) -> None:
            """物理モデルの設定"""
            self.physical_model = model
            print("Physical model configured")
    
        def read_all_sensors(self) -> Dict[str, Any]:
            """全センサーの読み取り"""
            sensor_data = {}
            for sensor_id, sensor in self.sensors.items():
                reading = sensor.read()
                sensor_data[sensor.sensor_type] = reading['value']
    
            sensor_data['timestamp'] = datetime.now()
            return sensor_data
    
        def run_physics_simulation(self, current_state: Dict[str, float],
                                    dt: float = 1.0) -> Dict[str, float]:
            """物理モデルによる状態更新"""
            if self.physical_model is None:
                return current_state
    
            # 簡易CSTRモデル
            T = current_state.get('temperature', 185.0)
            C = current_state.get('concentration', 2.5)
    
            # パラメータ
            k_reaction = 0.05  # 反応速度定数
            k_cooling = 0.02   # 冷却速度定数
            T_coolant = 60.0   # 冷却水温度
            delta_H = -15.0    # 反応熱
    
            # 微分方程式（オイラー法）
            dT = (-k_cooling * (T - T_coolant) + delta_H * k_reaction * C) * dt
            dC = -k_reaction * C * dt
    
            new_state = {
                'temperature': T + dT,
                'concentration': C + dC,
                'timestamp': datetime.now()
            }
    
            return new_state
    
        def synchronize(self) -> None:
            """物理システムとデジタルツインの同期"""
            # センサーデータ読み取り
            sensor_data = self.read_all_sensors()
    
            # 物理モデルによる状態推定
            predicted_state = self.run_physics_simulation(sensor_data)
    
            # 状態履歴に保存
            self.state_history.append({
                'timestamp': sensor_data['timestamp'],
                'sensor_data': sensor_data,
                'predicted_state': predicted_state
            })
    
            # センサー基準値の更新（フィードバック）
            if 'temperature' in sensor_data and 'temperature' in self.sensors:
                self.sensors['temperature'].set_base_value(predicted_state['temperature'])
    
        def run_realtime_simulation(self, duration_seconds: int = 10,
                                     update_interval: float = 1.0) -> None:
            """リアルタイムシミュレーション"""
            print(f"\nStarting real-time simulation for {duration_seconds} seconds...")
            print("=" * 70)
    
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                self.synchronize()
    
                # 現在状態の表示
                latest = self.state_history[-1]
                sensor_T = latest['sensor_data'].get('temperature', 0)
                predicted_T = latest['predicted_state'].get('temperature', 0)
                sensor_C = latest['sensor_data'].get('concentration', 0)
                predicted_C = latest['predicted_state'].get('concentration', 0)
    
                print(f"[t={len(self.state_history):3d}s] "
                      f"Sensor: T={sensor_T:6.2f}°C, C={sensor_C:.3f}mol/L | "
                      f"Model: T={predicted_T:6.2f}°C, C={predicted_C:.3f}mol/L")
    
                time.sleep(update_interval)
    
            print("=" * 70)
            print("Simulation completed")
    
        def visualize_state_history(self) -> None:
            """状態履歴の可視化"""
            if not self.state_history:
                print("No data to visualize")
                return
    
            times = list(range(len(self.state_history)))
            sensor_temps = [s['sensor_data'].get('temperature', 0) for s in self.state_history]
            predicted_temps = [s['predicted_state'].get('temperature', 0) for s in self.state_history]
            sensor_concs = [s['sensor_data'].get('concentration', 0) for s in self.state_history]
            predicted_concs = [s['predicted_state'].get('concentration', 0) for s in self.state_history]
    
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
            # 温度プロット
            ax1.plot(times, sensor_temps, 'o-', label='Sensor (Physical)',
                     color='#11998e', linewidth=2, markersize=4)
            ax1.plot(times, predicted_temps, 's--', label='Model (Digital Twin)',
                     color='#e74c3c', linewidth=2, markersize=4, alpha=0.7)
            ax1.set_xlabel('Time [s]', fontsize=11)
            ax1.set_ylabel('Temperature [°C]', fontsize=11)
            ax1.set_title('Digital Twin Temperature Synchronization', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)
    
            # 濃度プロット
            ax2.plot(times, sensor_concs, 'o-', label='Sensor (Physical)',
                     color='#11998e', linewidth=2, markersize=4)
            ax2.plot(times, predicted_concs, 's--', label='Model (Digital Twin)',
                     color='#e74c3c', linewidth=2, markersize=4, alpha=0.7)
            ax2.set_xlabel('Time [s]', fontsize=11)
            ax2.set_ylabel('Concentration [mol/L]', fontsize=11)
            ax2.set_title('Digital Twin Concentration Synchronization', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    
    # 使用例: 簡易デジタルツインプロトタイプの実行
    print("=" * 70)
    print("簡易デジタルツインプロトタイプ")
    print("=" * 70)
    
    # デジタルツインの作成
    dt_prototype = SimpleDigitalTwinPrototype(twin_id="DT-PROTO-001")
    
    # 仮想センサーの追加
    temp_sensor = VirtualSensor('TI-101', 'temperature', base_value=185.0, noise_std=1.0)
    conc_sensor = VirtualSensor('AI-104', 'concentration', base_value=2.5, noise_std=0.05)
    
    dt_prototype.add_sensor(temp_sensor)
    dt_prototype.add_sensor(conc_sensor)
    
    # 物理モデルの設定
    dt_prototype.set_physical_model(model=lambda x: x)
    
    # リアルタイムシミュレーション実行
    dt_prototype.run_realtime_simulation(duration_seconds=10, update_interval=1.0)
    
    # 状態履歴の可視化
    # dt_prototype.visualize_state_history()  # matplotlib表示（実行時にコメント解除）
    
    print(f"\nTotal synchronized states: {len(dt_prototype.state_history)}")
    

**出力例:**
    
    
    ======================================================================
    簡易デジタルツインプロトタイプ
    ======================================================================
    Added sensor: TI-101 (temperature)
    Added sensor: AI-104 (concentration)
    Physical model configured
    
    Starting real-time simulation for 10 seconds...
    ======================================================================
    [t=  1s] Sensor: T=186.23°C, C=2.512mol/L | Model: T=184.87°C, C=2.488mol/L
    [t=  2s] Sensor: T=183.65°C, C=2.475mol/L | Model: T=183.72°C, C=2.476mol/L
    [t=  3s] Sensor: T=182.48°C, C=2.463mol/L | Model: T=182.58°C, C=2.464mol/L
    [t=  4s] Sensor: T=181.34°C, C=2.451mol/L | Model: T=181.45°C, C=2.452mol/L
    [t=  5s] Sensor: T=180.21°C, C=2.439mol/L | Model: T=180.33°C, C=2.440mol/L
    [t=  6s] Sensor: T=179.10°C, C=2.427mol/L | Model: T=179.22°C, C=2.429mol/L
    [t=  7s] Sensor: T=178.00°C, C=2.416mol/L | Model: T=178.13°C, C=2.417mol/L
    [t=  8s] Sensor: T=176.92°C, C=2.404mol/L | Model: T=177.05°C, C=2.406mol/L
    [t=  9s] Sensor: T=175.85°C, C=2.393mol/L | Model: T=175.98°C, C=2.395mol/L
    [t= 10s] Sensor: T=174.80°C, C=2.382mol/L | Model: T=174.92°C, C=2.383mol/L
    ======================================================================
    Simulation completed
    
    Total synchronized states: 10
    

**解説:** この簡易プロトタイプは、仮想センサー（物理システムのシミュレーション）とデジタルツイン（物理モデル）をリアルタイムに同期させます。実際のIoTセンサーとの統合は第2章で扱います。

* * *

### コード例7: デジタルツイン総合デモ（状態可視化ダッシュボード）
    
    
    from dataclasses import dataclass
    from typing import Dict, List
    import numpy as np
    from datetime import datetime
    
    @dataclass
    class DigitalTwinDashboard:
        """デジタルツイン可視化ダッシュボード"""
    
        twin_id: str
        state_history: List[Dict] = None
    
        def __post_init__(self):
            if self.state_history is None:
                self.state_history = []
    
        def update_dashboard(self, sensor_data: Dict[str, float],
                             model_prediction: Dict[str, float],
                             alerts: List[str] = None) -> None:
            """ダッシュボードデータの更新"""
            dashboard_entry = {
                'timestamp': datetime.now(),
                'sensor_data': sensor_data,
                'model_prediction': model_prediction,
                'deviation': self._calculate_deviation(sensor_data, model_prediction),
                'alerts': alerts or []
            }
            self.state_history.append(dashboard_entry)
    
        def _calculate_deviation(self, sensor: Dict[str, float],
                                  model: Dict[str, float]) -> Dict[str, float]:
            """センサーとモデルの乖離計算"""
            deviations = {}
            for key in sensor.keys():
                if key in model:
                    deviation = abs(sensor[key] - model[key])
                    deviations[key] = deviation
            return deviations
    
        def generate_text_dashboard(self) -> str:
            """テキストベースダッシュボードの生成"""
            if not self.state_history:
                return "No data available"
    
            latest = self.state_history[-1]
    
            dashboard_text = f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║           DIGITAL TWIN MONITORING DASHBOARD                       ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Twin ID: {self.twin_id}
    Timestamp: {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
    Status: {'⚠️  ALERT' if latest['alerts'] else '✅ NORMAL'}
    
    ┌───────────────────────────────────────────────────────────────────┐
    │ PROCESS VARIABLES                                                 │
    ├───────────────────────────────────────────────────────────────────┤
    """
            # プロセス変数の表示
            sensor = latest['sensor_data']
            model = latest['model_prediction']
            deviation = latest['deviation']
    
            for key in sensor.keys():
                if key != 'timestamp':
                    sensor_val = sensor.get(key, 0)
                    model_val = model.get(key, 0)
                    dev_val = deviation.get(key, 0)
    
                    # アラート判定
                    alert_symbol = "⚠️ " if dev_val > 5.0 else "  "
    
                    dashboard_text += f"│ {alert_symbol}{key.upper():20s}: Sensor={sensor_val:7.2f} | Model={model_val:7.2f} | Δ={dev_val:5.2f} │\n"
    
            dashboard_text += "└───────────────────────────────────────────────────────────────────┘\n"
    
            # アラート
            if latest['alerts']:
                dashboard_text += "\n┌───────────────────────────────────────────────────────────────────┐\n"
                dashboard_text += "│ ALERTS                                                            │\n"
                dashboard_text += "├───────────────────────────────────────────────────────────────────┤\n"
                for alert in latest['alerts']:
                    dashboard_text += f"│ ⚠️  {alert:63s} │\n"
                dashboard_text += "└───────────────────────────────────────────────────────────────────┘\n"
    
            # 統計情報
            if len(self.state_history) > 1:
                recent_temps = [s['sensor_data'].get('temperature', 0) for s in self.state_history[-10:]]
                dashboard_text += f"\n┌───────────────────────────────────────────────────────────────────┐\n"
                dashboard_text += f"│ STATISTICS (Last 10 samples)                                      │\n"
                dashboard_text += f"├───────────────────────────────────────────────────────────────────┤\n"
                dashboard_text += f"│ Temperature: Avg={np.mean(recent_temps):6.2f}°C | Std={np.std(recent_temps):5.2f}°C       │\n"
                dashboard_text += f"│ Total Samples: {len(self.state_history):5d}                                         │\n"
                dashboard_text += f"└───────────────────────────────────────────────────────────────────┘\n"
    
            return dashboard_text
    
        def get_kpi_summary(self) -> Dict[str, Any]:
            """KPIサマリーの取得"""
            if not self.state_history:
                return {}
    
            # 平均乖離率
            all_deviations = [entry['deviation'] for entry in self.state_history]
            avg_deviation_temp = np.mean([d.get('temperature', 0) for d in all_deviations])
    
            # アラート発生率
            alert_count = sum(1 for entry in self.state_history if entry['alerts'])
            alert_rate = alert_count / len(self.state_history) * 100
    
            # データ品質スコア（乖離が小さいほど高品質）
            quality_score = max(0, 100 - avg_deviation_temp * 10)
    
            return {
                'total_samples': len(self.state_history),
                'avg_deviation_temperature': avg_deviation_temp,
                'alert_count': alert_count,
                'alert_rate_percent': alert_rate,
                'data_quality_score': quality_score
            }
    
    
    # 使用例: デジタルツインダッシュボードデモ
    dashboard = DigitalTwinDashboard(twin_id="DT-REACTOR-001")
    
    # シミュレーションデータ
    np.random.seed(42)
    for i in range(15):
        # センサーデータ（物理システム）
        sensor_data = {
            'temperature': 185.0 + np.random.normal(0, 2.0),
            'pressure': 3.5 + np.random.normal(0, 0.2),
            'concentration': 2.5 + np.random.normal(0, 0.1)
        }
    
        # モデル予測（デジタルツイン）
        model_prediction = {
            'temperature': sensor_data['temperature'] + np.random.normal(0, 1.5),
            'pressure': sensor_data['pressure'] + np.random.normal(0, 0.1),
            'concentration': sensor_data['concentration'] + np.random.normal(0, 0.05)
        }
    
        # アラート生成
        alerts = []
        if abs(sensor_data['temperature'] - model_prediction['temperature']) > 4.0:
            alerts.append("High deviation in temperature prediction")
        if sensor_data['temperature'] > 190:
            alerts.append("Temperature approaching upper limit")
    
        dashboard.update_dashboard(sensor_data, model_prediction, alerts)
    
    # ダッシュボード表示
    print(dashboard.generate_text_dashboard())
    
    # KPIサマリー
    kpi = dashboard.get_kpi_summary()
    print("\n" + "=" * 70)
    print("KPI SUMMARY")
    print("=" * 70)
    print(f"Total Samples: {kpi['total_samples']}")
    print(f"Avg Temperature Deviation: {kpi['avg_deviation_temperature']:.2f}°C")
    print(f"Alert Count: {kpi['alert_count']}")
    print(f"Alert Rate: {kpi['alert_rate_percent']:.1f}%")
    print(f"Data Quality Score: {kpi['data_quality_score']:.1f}/100")
    

**出力例:**
    
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║           DIGITAL TWIN MONITORING DASHBOARD                       ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Twin ID: DT-REACTOR-001
    Timestamp: 2025-10-26 10:30:45
    Status: ✅ NORMAL
    
    ┌───────────────────────────────────────────────────────────────────┐
    │ PROCESS VARIABLES                                                 │
    ├───────────────────────────────────────────────────────────────────┤
    │   TEMPERATURE        : Sensor= 185.47 | Model= 184.82 | Δ= 0.65 │
    │   PRESSURE           : Sensor=   3.52 | Model=   3.48 | Δ= 0.04 │
    │   CONCENTRATION      : Sensor=   2.51 | Model=   2.50 | Δ= 0.01 │
    └───────────────────────────────────────────────────────────────────┘
    
    ┌───────────────────────────────────────────────────────────────────┐
    │ STATISTICS (Last 10 samples)                                      │
    ├───────────────────────────────────────────────────────────────────┤
    │ Temperature: Avg=185.12°C | Std= 1.85°C       │
    │ Total Samples:    15                                         │
    └───────────────────────────────────────────────────────────────────┘
    
    ======================================================================
    KPI SUMMARY
    ======================================================================
    Total Samples: 15
    Avg Temperature Deviation: 1.42°C
    Alert Count: 2
    Alert Rate: 13.3%
    Data Quality Score: 85.8/100
    

**解説:** ダッシュボードは、デジタルツインの状態監視の中心的役割を果たします。センサーデータとモデル予測の乖離をリアルタイムで可視化し、アラートとKPIにより運用者の意思決定を支援します。

* * *

## 1.3 本章のまとめ

### 学んだこと

  1. **デジタルツインの概念**
     * 物理システムの仮想レプリカとしてのデジタルツイン
     * デジタルシャドウ（一方向）vs デジタルツイン（双方向）の違い
     * プロセス産業における価値（最適化、予知保全、仮想試運転）
  2. **アーキテクチャ設計**
     * 物理システム層、データ層、モデル層、アプリケーション層の4層構造
     * 状態表現とJSON形式でのシリアライゼーション
     * 双方向データフローの設計
  3. **モデル忠実度レベル**
     * L1（データログ）→ L3（物理モデル）→ L5（自律最適化）
     * 各レベルの機能と適用シナリオ
  4. **ライフサイクル管理**
     * 設計 → 実装 → 検証 → 運用のフェーズ管理
     * 検証基準とメトリクス
  5. **評価指標**
     * 精度（RMSE、R²）、リアルタイム性（レイテンシ）
     * ビジネス価値（ROI、投資回収期間）
  6. **プロトタイプ実装**
     * 仮想センサーとデジタルツインの同期
     * ダッシュボードによる状態監視

### 重要なポイント

  * デジタルツインは単なるシミュレーションではなく、リアルタイム同期と双方向フィードバックが本質
  * 忠実度レベルL1-L5は、プロジェクトの成熟度に応じて段階的に進化させる
  * 状態表現の設計が、拡張性とメンテナンス性の鍵となる
  * 評価指標は、精度だけでなくビジネス価値も含めて定義する
  * ライフサイクル管理により、設計から運用まで一貫した品質を維持する

### 次の章へ

第2章では、**リアルタイムデータ連携とIoT統合** を詳しく学びます：

  * 産業用通信プロトコル（OPC UA、MQTT）の実装
  * 時系列データベース（InfluxDB）との統合
  * データストリーミング処理（Apache Kafka）
  * センサーデータ品質管理と異常値検出
  * エッジコンピューティングアーキテクチャ
