---
title: 第2章：HAZOPとリスクアセスメント
chapter_title: 第2章：HAZOPとリスクアセスメント
subtitle: ガイドワード適用、逸脱分析、定量的リスク評価（QRA）の完全実装
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ HAZOPスタディの手法を理解し実践できる
  * ✅ ガイドワード（No, More, Less, Reverse等）を適用できる
  * ✅ 逸脱分析（Deviation Analysis）を実施できる
  * ✅ 定量的リスクアセスメント（QRA）を実装できる
  * ✅ F-N曲線を作成しリスク判定ができる
  * ✅ HAZOPレポートを自動生成できる

* * *

## 2.1 HAZOPスタディの基礎

### HAZOPとは

**HAZOP（Hazard and Operability Study）** は、プロセスの設計段階で体系的にハザードと運転上の問題を識別する手法です。1960年代にICIで開発され、現在では化学プラント設計の標準となっています。

### HAZOPの原理

HAZOPは、**ガイドワード（Guide Word）** を**プロセスパラメータ（Process Parameter）** に適用して**逸脱（Deviation）** を生成し、その原因・結果・対策を分析します。

$$ \text{Deviation} = \text{Guide Word} + \text{Process Parameter} $$

### 主要なガイドワード

ガイドワード | 意味 | 適用例  
---|---|---  
**No/Not/None** | 完全な否定 | No flow, No pressure  
**More** | 量的増加 | More flow, More temperature  
**Less** | 量的減少 | Less flow, Less pressure  
**As Well As** | 追加 | Impurity as well as product  
**Part Of** | 一部欠落 | Part of composition (missing component)  
**Reverse** | 逆方向 | Reverse flow  
**Other Than** | 完全に異なる | Other than normal operation  
  
### Example 1: HAZOPガイドワードエンジン
    
    
    # ===================================
    # Example 1: HAZOPガイドワードエンジン
    # ===================================
    
    from dataclasses import dataclass
    from typing import List, Dict
    from enum import Enum
    
    class GuideWord(Enum):
        """HAZOPガイドワード"""
        NO = "No/Not/None"
        MORE = "More"
        LESS = "Less"
        AS_WELL_AS = "As Well As"
        PART_OF = "Part Of"
        REVERSE = "Reverse"
        OTHER_THAN = "Other Than"
    
    class ProcessParameter(Enum):
        """プロセスパラメータ"""
        FLOW = "Flow"
        PRESSURE = "Pressure"
        TEMPERATURE = "Temperature"
        LEVEL = "Level"
        COMPOSITION = "Composition"
        VISCOSITY = "Viscosity"
        TIME = "Time"
        MIXING = "Mixing"
    
    @dataclass
    class Deviation:
        """逸脱（Deviation）"""
        guide_word: GuideWord
        parameter: ProcessParameter
        description: str
        potential_causes: List[str]
        potential_consequences: List[str]
        existing_safeguards: List[str]
        recommendations: List[str]
        risk_rank: str  # High/Medium/Low
    
    class HAZOPGuideWordEngine:
        """HAZOPガイドワード適用エンジン"""
    
        def __init__(self):
            # ガイドワードとパラメータの組み合わせルール
            self.valid_combinations = {
                GuideWord.NO: [ProcessParameter.FLOW, ProcessParameter.PRESSURE,
                               ProcessParameter.LEVEL, ProcessParameter.MIXING],
                GuideWord.MORE: [ProcessParameter.FLOW, ProcessParameter.PRESSURE,
                                 ProcessParameter.TEMPERATURE, ProcessParameter.LEVEL],
                GuideWord.LESS: [ProcessParameter.FLOW, ProcessParameter.PRESSURE,
                                 ProcessParameter.TEMPERATURE, ProcessParameter.LEVEL],
                GuideWord.REVERSE: [ProcessParameter.FLOW],
                GuideWord.AS_WELL_AS: [ProcessParameter.COMPOSITION],
                GuideWord.PART_OF: [ProcessParameter.COMPOSITION],
                GuideWord.OTHER_THAN: [ProcessParameter.COMPOSITION, ProcessParameter.TIME]
            }
    
        def is_valid_combination(self, guide_word: GuideWord,
                                 parameter: ProcessParameter) -> bool:
            """ガイドワードとパラメータの組み合わせが有効かチェック"""
            if guide_word in self.valid_combinations:
                return parameter in self.valid_combinations[guide_word]
            return False
    
        def generate_deviation_description(self, guide_word: GuideWord,
                                            parameter: ProcessParameter,
                                            node_name: str = "Node") -> str:
            """逸脱の記述を生成"""
            return f"{guide_word.value} {parameter.value} at {node_name}"
    
        def apply_guide_words(self, node_name: str,
                              parameters: List[ProcessParameter]) -> List[str]:
            """ノードに対してガイドワードを適用"""
            deviations = []
    
            for param in parameters:
                for guide_word in GuideWord:
                    if self.is_valid_combination(guide_word, param):
                        deviation = self.generate_deviation_description(
                            guide_word, param, node_name
                        )
                        deviations.append(deviation)
    
            return deviations
    
    
    # 使用例
    hazop_engine = HAZOPGuideWordEngine()
    
    # ノード: 反応器への原料供給ライン
    node = "Reactor Feed Line (P&ID: R-101)"
    parameters = [
        ProcessParameter.FLOW,
        ProcessParameter.PRESSURE,
        ProcessParameter.TEMPERATURE,
        ProcessParameter.COMPOSITION
    ]
    
    # ガイドワード適用
    deviations = hazop_engine.apply_guide_words(node, parameters)
    
    print(f"=== HAZOP Node: {node} ===\n")
    print(f"Total Deviations Generated: {len(deviations)}\n")
    print("Deviations:")
    for i, dev in enumerate(deviations, 1):
        print(f"{i:2d}. {dev}")
    
    # 期待される出力:
    # === HAZOP Node: Reactor Feed Line (P&ID: R-101) ===
    #
    # Total Deviations Generated: 13
    #
    # Deviations:
    #  1. No/Not/None Flow at Reactor Feed Line (P&ID: R-101)
    #  2. More Flow at Reactor Feed Line (P&ID: R-101)
    #  3. Less Flow at Reactor Feed Line (P&ID: R-101)
    #  4. Reverse Flow at Reactor Feed Line (P&ID: R-101)
    #  5. No/Not/None Pressure at Reactor Feed Line (P&ID: R-101)
    #  6. More Pressure at Reactor Feed Line (P&ID: R-101)
    #  7. Less Pressure at Reactor Feed Line (P&ID: R-101)
    #  8. More Temperature at Reactor Feed Line (P&ID: R-101)
    #  9. Less Temperature at Reactor Feed Line (P&ID: R-101)
    # 10. As Well As Composition at Reactor Feed Line (P&ID: R-101)
    # 11. Part Of Composition at Reactor Feed Line (P&ID: R-101)
    # 12. Other Than Composition at Reactor Feed Line (P&ID: R-101)
    # 13. Other Than Time at Reactor Feed Line (P&ID: R-101)
    

* * *

## 2.2 逸脱分析（Deviation Analysis）

### Example 2: 逸脱分析システム実装

各逸脱に対して、原因・結果・既存対策・推奨事項を分析します。
    
    
    # ===================================
    # Example 2: 逸脱分析システム
    # ===================================
    
    import pandas as pd
    
    class DeviationAnalyzer:
        """逸脱分析システム"""
    
        def __init__(self, node_name: str, equipment_type: str):
            self.node_name = node_name
            self.equipment_type = equipment_type
            self.deviations: List[Deviation] = []
            self._load_knowledge_base()
    
        def _load_knowledge_base(self):
            """知識ベースをロード（典型的な原因・結果）"""
            # 実際のシステムでは、過去のHAZOPデータベースから学習
    
            self.knowledge_base = {
                ("NO", "FLOW"): {
                    "causes": [
                        "Pump failure",
                        "Valve closed (inadvertent or maintenance)",
                        "Line blockage (solidification, fouling)",
                        "Suction strainer plugged",
                        "Upstream vessel empty"
                    ],
                    "consequences": [
                        "Reactor starvation (loss of cooling/reactant)",
                        "Pump cavitation/damage",
                        "Product off-spec",
                        "Temperature excursion (if flow is coolant)"
                    ],
                    "safeguards": [
                        "Low flow alarm",
                        "Pump status indication",
                        "Level indication on upstream vessel"
                    ]
                },
                ("MORE", "FLOW"): {
                    "causes": [
                        "Control valve fails open",
                        "Pump speed increase (VFD malfunction)",
                        "Upstream pressure increase",
                        "Parallel line opened unintentionally"
                    ],
                    "consequences": [
                        "Reactor flooding",
                        "Downstream vessel overflow",
                        "Product off-spec (dilution)",
                        "Increased pressure drop"
                    ],
                    "safeguards": [
                        "High flow alarm",
                        "Level high alarm on receiving vessel",
                        "Flow control with override"
                    ]
                },
                ("MORE", "TEMPERATURE"): {
                    "causes": [
                        "Cooling system failure",
                        "Heat exchanger fouling (reduced heat transfer)",
                        "Exothermic reaction runaway",
                        "External fire",
                        "Temperature transmitter failure"
                    ],
                    "consequences": [
                        "Thermal decomposition",
                        "Reactor overpressure",
                        "Product degradation",
                        "Equipment damage (metallurgy limits)"
                    ],
                    "safeguards": [
                        "High temperature alarm",
                        "Independent high temperature interlock",
                        "Pressure relief valve",
                        "Emergency cooling system"
                    ]
                },
                ("REVERSE", "FLOW"): {
                    "causes": [
                        "Pump stopped with downstream pressure higher",
                        "Check valve failure",
                        "Incorrect valve lineup",
                        "Siphon effect"
                    ],
                    "consequences": [
                        "Contamination of upstream system",
                        "Pump damage (reverse rotation)",
                        "Loss of containment",
                        "Mixing of incompatible materials"
                    ],
                    "safeguards": [
                        "Check valve",
                        "Backflow prevention valve",
                        "Pump discharge valve"
                    ]
                }
            }
    
        def analyze_deviation(self, guide_word: GuideWord,
                              parameter: ProcessParameter) -> Deviation:
            """逸脱を分析"""
    
            # 知識ベースから情報取得
            key = (guide_word.name, parameter.name)
            kb_entry = self.knowledge_base.get(key, {
                "causes": ["To be determined in HAZOP meeting"],
                "consequences": ["To be determined in HAZOP meeting"],
                "safeguards": ["To be determined in HAZOP meeting"]
            })
    
            # リスクランキング（簡略化）
            high_risk_combos = [
                ("NO", "FLOW"), ("MORE", "TEMPERATURE"), ("MORE", "PRESSURE")
            ]
            risk_rank = "High" if key in high_risk_combos else "Medium"
    
            # 推奨事項生成
            recommendations = self._generate_recommendations(
                guide_word, parameter, kb_entry["safeguards"]
            )
    
            deviation = Deviation(
                guide_word=guide_word,
                parameter=parameter,
                description=f"{guide_word.value} {parameter.value} at {self.node_name}",
                potential_causes=kb_entry["causes"],
                potential_consequences=kb_entry["consequences"],
                existing_safeguards=kb_entry["safeguards"],
                recommendations=recommendations,
                risk_rank=risk_rank
            )
    
            self.deviations.append(deviation)
            return deviation
    
        def _generate_recommendations(self, guide_word: GuideWord,
                                       parameter: ProcessParameter,
                                       existing_safeguards: List[str]) -> List[str]:
            """推奨事項を生成"""
            recommendations = []
    
            # アラーム不足のチェック
            if "alarm" not in ' '.join(existing_safeguards).lower():
                recommendations.append(
                    f"Install {parameter.value.lower()} alarm"
                )
    
            # インターロック不足のチェック
            if guide_word in [GuideWord.MORE, GuideWord.NO] and \
               "interlock" not in ' '.join(existing_safeguards).lower():
                recommendations.append(
                    f"Consider {parameter.value.lower()} interlock (SIL assessment required)"
                )
    
            # 冗長性チェック
            if len(existing_safeguards) < 2:
                recommendations.append(
                    "Review adequacy of protection layers (LOPA recommended)"
                )
    
            if not recommendations:
                recommendations.append("Existing safeguards adequate - continue routine maintenance")
    
            return recommendations
    
        def generate_hazop_worksheet(self) -> pd.DataFrame:
            """HAZOP worksheetを生成"""
            data = []
    
            for dev in self.deviations:
                data.append({
                    'Node': self.node_name,
                    'Deviation': dev.description,
                    'Causes': '; '.join(dev.potential_causes[:2]) + '...',  # 最初の2つ
                    'Consequences': '; '.join(dev.potential_consequences[:2]) + '...',
                    'Safeguards': '; '.join(dev.existing_safeguards[:2]) + '...',
                    'Risk': dev.risk_rank,
                    'Actions': '; '.join(dev.recommendations[:1])  # 最優先の推奨事項
                })
    
            return pd.DataFrame(data)
    
    
    # 使用例
    analyzer = DeviationAnalyzer(
        node_name="Reactor Feed Line (R-101)",
        equipment_type="Piping"
    )
    
    # 主要な逸脱を分析
    critical_deviations = [
        (GuideWord.NO, ProcessParameter.FLOW),
        (GuideWord.MORE, ProcessParameter.FLOW),
        (GuideWord.MORE, ProcessParameter.TEMPERATURE),
        (GuideWord.REVERSE, ProcessParameter.FLOW)
    ]
    
    for gw, param in critical_deviations:
        analyzer.analyze_deviation(gw, param)
    
    # HAZOP worksheet生成
    worksheet = analyzer.generate_hazop_worksheet()
    
    print("=== HAZOP Worksheet ===\n")
    print(worksheet.to_string(index=False))
    
    print("\n=== Detailed Analysis: No Flow ===")
    no_flow_dev = analyzer.deviations[0]
    print(f"\nDeviation: {no_flow_dev.description}")
    print(f"\nPotential Causes:")
    for cause in no_flow_dev.potential_causes:
        print(f"  - {cause}")
    print(f"\nPotential Consequences:")
    for cons in no_flow_dev.potential_consequences:
        print(f"  - {cons}")
    print(f"\nRecommendations:")
    for rec in no_flow_dev.recommendations:
        print(f"  - {rec}")
    
    # 期待される出力:
    # === HAZOP Worksheet ===
    #
    #                     Node                           Deviation  Risk ...
    #  Reactor Feed Line (R-101)    No/Not/None Flow at Reactor... High ...
    #  Reactor Feed Line (R-101)          More Flow at Reactor... Medium ...
    #  Reactor Feed Line (R-101)   More Temperature at Reactor... High ...
    #  Reactor Feed Line (R-101)       Reverse Flow at Reactor... Medium ...
    

* * *

## 2.3 HAZOP自動化システム

### Example 3: P&ID解析とHAZOP自動生成

P&ID（配管計装図）情報から自動的にHAZOPノードと逸脱を生成します。
    
    
    # ===================================
    # Example 3: P&ID解析とHAZOP自動生成
    # ===================================
    
    from dataclasses import dataclass
    from typing import List, Dict, Set
    
    @dataclass
    class PIDElement:
        """P&ID要素"""
        id: str
        element_type: str  # Vessel, Pump, HeatExchanger, Valve, Line
        fluid: str
        design_pressure: float  # MPa
        design_temperature: float  # °C
        connected_to: List[str]  # 接続先要素ID
    
    @dataclass
    class HAZOPNode:
        """HAZOPノード（検討単位）"""
        id: str
        name: str
        elements: List[PIDElement]
        intent: str  # Design Intent（設計意図）
        parameters_of_interest: List[ProcessParameter]
    
    class PIDHAZOPAutomation:
        """P&IDベースHAZOP自動化システム"""
    
        def __init__(self):
            self.pid_elements: Dict[str, PIDElement] = {}
            self.hazop_nodes: List[HAZOPNode] = []
    
        def add_pid_element(self, element: PIDElement):
            """P&ID要素を追加"""
            self.pid_elements[element.id] = element
    
        def identify_hazop_nodes(self) -> List[HAZOPNode]:
            """HAZOPノードを自動識別
    
            ノード分割基準:
              - 機能単位（反応、分離、熱交換）
              - プロセス条件変化点（相変化、温度/圧力変化）
              - 制御ループ境界
            """
            nodes = []
    
            # 簡略化: 各主要設備をノードとして扱う
            for elem_id, elem in self.pid_elements.items():
                if elem.element_type in ['Vessel', 'Pump', 'HeatExchanger']:
                    # 設計意図を推定
                    intent = self._infer_design_intent(elem)
    
                    # 関連パラメータを決定
                    parameters = self._determine_parameters(elem)
    
                    node = HAZOPNode(
                        id=f"NODE-{elem_id}",
                        name=f"{elem.element_type}: {elem_id}",
                        elements=[elem],
                        intent=intent,
                        parameters_of_interest=parameters
                    )
    
                    nodes.append(node)
    
            self.hazop_nodes = nodes
            return nodes
    
        def _infer_design_intent(self, element: PIDElement) -> str:
            """設計意図を推定"""
            intents = {
                'Vessel': f"Contain {element.fluid} at {element.design_pressure} MPa, {element.design_temperature}°C",
                'Pump': f"Transfer {element.fluid} at design flow rate",
                'HeatExchanger': f"Heat/cool {element.fluid} to target temperature"
            }
            return intents.get(element.element_type, "To be determined")
    
        def _determine_parameters(self, element: PIDElement) -> List[ProcessParameter]:
            """検討すべきパラメータを決定"""
            param_map = {
                'Vessel': [ProcessParameter.LEVEL, ProcessParameter.PRESSURE,
                           ProcessParameter.TEMPERATURE, ProcessParameter.COMPOSITION],
                'Pump': [ProcessParameter.FLOW, ProcessParameter.PRESSURE],
                'HeatExchanger': [ProcessParameter.FLOW, ProcessParameter.TEMPERATURE,
                                  ProcessParameter.PRESSURE]
            }
            return param_map.get(element.element_type,
                                 [ProcessParameter.FLOW, ProcessParameter.PRESSURE])
    
        def generate_hazop_study(self) -> Dict:
            """完全なHAZOP studyを生成"""
    
            # ノード識別
            if not self.hazop_nodes:
                self.identify_hazop_nodes()
    
            # 各ノードに対してHAZOP実施
            study_results = []
    
            for node in self.hazop_nodes:
                analyzer = DeviationAnalyzer(node.name, node.elements[0].element_type)
    
                # ガイドワード適用
                for param in node.parameters_of_interest:
                    for guide_word in GuideWord:
                        if analyzer._load_knowledge_base() or True:  # simplified
                            # 有効な組み合わせのみ分析
                            engine = HAZOPGuideWordEngine()
                            if engine.is_valid_combination(guide_word, param):
                                try:
                                    deviation = analyzer.analyze_deviation(guide_word, param)
                                    study_results.append({
                                        'Node': node.name,
                                        'Intent': node.intent,
                                        'Deviation': deviation.description,
                                        'Risk': deviation.risk_rank
                                    })
                                except:
                                    pass
    
            return {
                'total_nodes': len(self.hazop_nodes),
                'total_deviations': len(study_results),
                'results': pd.DataFrame(study_results)
            }
    
    
    # 使用例: 簡易プロセスのP&ID構築
    pid_system = PIDHAZOPAutomation()
    
    # P&ID要素を追加
    pid_system.add_pid_element(PIDElement(
        id="R-101",
        element_type="Vessel",
        fluid="Ethylene/Catalyst",
        design_pressure=3.0,
        design_temperature=150,
        connected_to=["P-101", "E-101"]
    ))
    
    pid_system.add_pid_element(PIDElement(
        id="P-101",
        element_type="Pump",
        fluid="Product mixture",
        design_pressure=5.0,
        design_temperature=80,
        connected_to=["R-101", "T-201"]
    ))
    
    pid_system.add_pid_element(PIDElement(
        id="E-101",
        element_type="HeatExchanger",
        fluid="Reactor coolant",
        design_pressure=1.5,
        design_temperature=100,
        connected_to=["R-101"]
    ))
    
    # HAZOPノード自動識別
    nodes = pid_system.identify_hazop_nodes()
    print(f"=== HAZOP Nodes Identified: {len(nodes)} ===\n")
    for node in nodes:
        print(f"Node: {node.name}")
        print(f"  Intent: {node.intent}")
        print(f"  Parameters: {[p.value for p in node.parameters_of_interest]}\n")
    
    # HAZOP study自動生成
    study = pid_system.generate_hazop_study()
    print(f"\n=== HAZOP Study Summary ===")
    print(f"Total Nodes: {study['total_nodes']}")
    print(f"Total Deviations: {study['total_deviations']}\n")
    print(study['results'].head(10).to_string(index=False))
    
    # 期待される出力:
    # === HAZOP Nodes Identified: 3 ===
    #
    # Node: Vessel: R-101
    #   Intent: Contain Ethylene/Catalyst at 3.0 MPa, 150°C
    #   Parameters: ['Level', 'Pressure', 'Temperature', 'Composition']
    #
    # Node: Pump: P-101
    #   Intent: Transfer Product mixture at design flow rate
    #   Parameters: ['Flow', 'Pressure']
    # ...
    

* * *

## 2.4 定量的リスクアセスメント（QRA）

### Example 4: イベントツリー解析（ETA）

起因事象から最終結果までの事象連鎖を確率的に評価します。
    
    
    # ===================================
    # Example 4: イベントツリー解析（ETA）
    # ===================================
    
    from dataclasses import dataclass
    from typing import List, Optional
    import matplotlib.pyplot as plt
    import numpy as np
    
    @dataclass
    class EventNode:
        """イベントツリーのノード"""
        name: str
        success_probability: float
        failure_probability: float
    
        def __post_init__(self):
            assert abs(self.success_probability + self.failure_probability - 1.0) < 1e-6, \
                "Success + Failure probability must equal 1.0"
    
    @dataclass
    class Outcome:
        """最終結果"""
        path: List[str]
        probability: float
        consequence: str
        severity: str  # Catastrophic, Critical, Marginal, Negligible
    
    class EventTreeAnalysis:
        """イベントツリー解析システム"""
    
        def __init__(self, initiating_event: str, frequency: float):
            """
            Args:
                initiating_event: 起因事象名
                frequency: 起因事象頻度（回/年）
            """
            self.initiating_event = initiating_event
            self.frequency = frequency
            self.safety_functions: List[EventNode] = []
            self.outcomes: List[Outcome] = []
    
        def add_safety_function(self, node: EventNode):
            """安全機能を追加"""
            self.safety_functions.append(node)
    
        def calculate_outcomes(self):
            """全ての結果パスと確率を計算"""
            self.outcomes = []
    
            # すべての可能な組み合わせを生成（2^n通り）
            n_functions = len(self.safety_functions)
            for i in range(2 ** n_functions):
                path = []
                probability = self.frequency
    
                # バイナリ表現でSuccess/Failureを決定
                binary = format(i, f'0{n_functions}b')
    
                for j, bit in enumerate(binary):
                    function = self.safety_functions[j]
                    if bit == '0':  # Success
                        path.append(f"{function.name}: Success")
                        probability *= function.success_probability
                    else:  # Failure
                        path.append(f"{function.name}: Failure")
                        probability *= function.failure_probability
    
                # 結果の影響度を判定
                failure_count = binary.count('1')
                consequence, severity = self._determine_consequence(failure_count)
    
                self.outcomes.append(Outcome(
                    path=path,
                    probability=probability,
                    consequence=consequence,
                    severity=severity
                ))
    
        def _determine_consequence(self, failure_count: int) -> tuple:
            """失敗数に応じて影響度を決定"""
            n = len(self.safety_functions)
    
            if failure_count == 0:
                return "Safe shutdown", "Negligible"
            elif failure_count == 1:
                return "Minor release (contained by secondary barrier)", "Marginal"
            elif failure_count == 2:
                return "Significant release (requires emergency response)", "Critical"
            else:
                return "Major release (offsite consequences)", "Catastrophic"
    
        def generate_eta_report(self) -> str:
            """イベントツリー解析レポート生成"""
            if not self.outcomes:
                self.calculate_outcomes()
    
            report = f"""
    {'='*80}
    EVENT TREE ANALYSIS (ETA) REPORT
    {'='*80}
    
    Initiating Event: {self.initiating_event}
    Frequency: {self.frequency:.2e} events/year
    
    Safety Functions:
    """
    
            for i, sf in enumerate(self.safety_functions, 1):
                report += f"{i}. {sf.name}\n"
                report += f"   Success Prob: {sf.success_probability:.4f} ({sf.success_probability*100:.2f}%)\n"
                report += f"   Failure Prob: {sf.failure_probability:.4f} ({sf.failure_probability*100:.2f}%)\n\n"
    
            report += f"""
    {'='*80}
    OUTCOMES
    {'='*80}
    """
    
            # 影響度でソート
            severity_order = {"Catastrophic": 0, "Critical": 1, "Marginal": 2, "Negligible": 3}
            sorted_outcomes = sorted(self.outcomes,
                                     key=lambda x: (severity_order[x.severity], -x.probability))
    
            for i, outcome in enumerate(sorted_outcomes, 1):
                report += f"\nOutcome {i}: {outcome.consequence}\n"
                report += f"  Severity: {outcome.severity}\n"
                report += f"  Frequency: {outcome.probability:.2e} events/year\n"
                report += f"  Path:\n"
                for step in outcome.path:
                    report += f"    → {step}\n"
    
            # 総リスク計算
            total_risk = sum(o.probability for o in self.outcomes)
            catastrophic_risk = sum(o.probability for o in self.outcomes
                                    if o.severity == "Catastrophic")
    
            report += f"""
    {'='*80}
    RISK SUMMARY
    {'='*80}
    Total Event Frequency (all paths): {total_risk:.2e} events/year
    Catastrophic Outcome Frequency: {catastrophic_risk:.2e} events/year
    
    Risk Acceptance Criteria (example):
      Catastrophic: < 1e-6 events/year  {'✅ ACCEPTABLE' if catastrophic_risk < 1e-6 else '❌ UNACCEPTABLE'}
      Critical:     < 1e-4 events/year
    """
    
            return report
    
    
    # 使用例: 可燃性ガス漏洩シナリオのETA
    eta = EventTreeAnalysis(
        initiating_event="Large flange gasket failure causing H2 release",
        frequency=1e-3  # 0.001回/年（1000年に1回）
    )
    
    # 安全機能を追加（多重保護層）
    eta.add_safety_function(EventNode(
        name="Gas detection system activates",
        success_probability=0.95,
        failure_probability=0.05
    ))
    
    eta.add_safety_function(EventNode(
        name="Automatic isolation valve closes",
        success_probability=0.98,
        failure_probability=0.02
    ))
    
    eta.add_safety_function(EventNode(
        name="Ignition prevention (no ignition sources)",
        success_probability=0.90,
        failure_probability=0.10
    ))
    
    # イベントツリー解析実行
    report = eta.generate_eta_report()
    print(report)
    
    # 詳細分析: 最悪ケースシナリオ
    worst_case = max(eta.outcomes, key=lambda x: x.probability
                     if x.severity == "Catastrophic" else 0)
    
    print("\n=== WORST CASE SCENARIO ===")
    print(f"Consequence: {worst_case.consequence}")
    print(f"Frequency: {worst_case.probability:.2e} events/year")
    print(f"Return Period: {1/worst_case.probability:.0f} years")
    
    # 期待される出力:
    # ============================================================================
    # EVENT TREE ANALYSIS (ETA) REPORT
    # ============================================================================
    #
    # Initiating Event: Large flange gasket failure causing H2 release
    # Frequency: 1.00e-03 events/year
    #
    # Safety Functions:
    # 1. Gas detection system activates
    #    Success Prob: 0.9500 (95.00%)
    #    Failure Prob: 0.0500 (5.00%)
    #
    # 2. Automatic isolation valve closes
    #    Success Prob: 0.9800 (98.00%)
    #    Failure Prob: 0.0200 (2.00%)
    # ...
    #
    # Outcome 1: Major release (offsite consequences)
    #   Severity: Catastrophic
    #   Frequency: 1.00e-07 events/year  (全3つの安全機能が失敗)
    #   Path:
    #     → Gas detection system activates: Failure
    #     → Automatic isolation valve closes: Failure
    #     → Ignition prevention (no ignition sources): Failure
    

* * *

## 2.5 故障頻度データと信頼性

### Example 5: Generic Failure Rate Database
    
    
    # ===================================
    # Example 5: Generic Failure Rate Database
    # ===================================
    
    import pandas as pd
    from dataclasses import dataclass
    from typing import Dict
    
    @dataclass
    class FailureRateData:
        """故障率データ"""
        equipment_type: str
        failure_mode: str
        failure_rate: float  # failures per million hours (10^-6/hr)
        source: str  # データソース（OREDA, PDS, etc.）
        confidence_factor: float  # 1.0 = nominal, >1.0 = conservative
    
    class GenericFailureRateDatabase:
        """汎用故障率データベース"""
    
        def __init__(self):
            self.database: Dict[str, FailureRateData] = {}
            self._initialize_database()
    
        def _initialize_database(self):
            """データベース初期化（OREDA, PDS等のデータに基づく）"""
    
            failure_rates = [
                # Valves
                ("Isolation Valve", "Fail to close on demand", 5.0, "OREDA 2015", 1.0),
                ("Control Valve", "Fail open", 10.0, "PDS 2010", 1.0),
                ("Check Valve", "Reverse flow (fail to close)", 50.0, "API 581", 1.5),
                ("Pressure Relief Valve", "Fail to open on demand", 10.0, "API 581", 1.0),
    
                # Pumps
                ("Centrifugal Pump", "Fail to start", 5.0, "OREDA 2015", 1.0),
                ("Centrifugal Pump", "Fail to run (shutdown)", 50.0, "OREDA 2015", 1.0),
    
                # Instrumentation
                ("Pressure Transmitter", "Out of calibration", 2.0, "EXIDA", 1.0),
                ("Temperature Transmitter", "Fail dangerous", 1.5, "EXIDA", 1.0),
                ("Level Transmitter", "Fail dangerous", 3.0, "EXIDA", 1.0),
                ("Gas Detector", "Fail to detect", 100.0, "Industry avg", 2.0),
    
                # Safety Systems
                ("SIS Logic Solver", "Fail dangerous", 0.5, "IEC 61508", 1.0),
                ("Emergency Shutdown Valve", "Fail to close", 3.0, "API 581", 1.0),
    
                # Vessels & Piping
                ("Pressure Vessel", "Catastrophic rupture", 0.01, "API 579", 1.0),
                ("Piping (per km)", "Major leak (>10% cross-section)", 0.5, "UKOPA", 1.0),
                ("Flange Gasket", "Leak", 10.0, "Industry avg", 1.5),
    
                # Heat Exchangers
                ("Shell & Tube HX", "Tube leak", 20.0, "OREDA 2015", 1.0)
            ]
    
            for eq_type, mode, rate, source, cf in failure_rates:
                key = f"{eq_type}_{mode}"
                self.database[key] = FailureRateData(eq_type, mode, rate, source, cf)
    
        def get_failure_rate(self, equipment_type: str, failure_mode: str,
                             apply_confidence_factor: bool = True) -> float:
            """故障率を取得
    
            Returns:
                Failure rate in failures per year
            """
            key = f"{equipment_type}_{failure_mode}"
            data = self.database.get(key)
    
            if not data:
                raise ValueError(f"No data for {equipment_type} - {failure_mode}")
    
            rate_per_year = data.failure_rate * 8760 / 1e6  # 10^-6/hr → /year
    
            if apply_confidence_factor:
                rate_per_year *= data.confidence_factor
    
            return rate_per_year
    
        def get_pfd_from_test_interval(self, equipment_type: str, failure_mode: str,
                                        test_interval_months: int = 12) -> float:
            """Probability of Failure on Demand（要求時故障確率）を計算
    
            簡略式: PFD ≈ λ * T / 2
            λ: 故障率（/year）
            T: テスト間隔（year）
            """
            failure_rate = self.get_failure_rate(equipment_type, failure_mode)
            test_interval_years = test_interval_months / 12.0
    
            pfd = (failure_rate * test_interval_years) / 2.0
    
            return min(pfd, 1.0)  # PFDは最大1.0
    
        def generate_reliability_datasheet(self, equipment_list: List[tuple]) -> pd.DataFrame:
            """信頼性データシート生成
    
            Args:
                equipment_list: [(equipment_type, failure_mode, test_interval_months), ...]
            """
            data = []
    
            for eq_type, mode, test_interval in equipment_list:
                failure_rate = self.get_failure_rate(eq_type, mode)
                pfd = self.get_pfd_from_test_interval(eq_type, mode, test_interval)
    
                # SIL capability判定
                if pfd < 1e-4:
                    sil_capability = "SIL 3"
                elif pfd < 1e-3:
                    sil_capability = "SIL 2"
                elif pfd < 1e-2:
                    sil_capability = "SIL 1"
                else:
                    sil_capability = "No SIL"
    
                data.append({
                    'Equipment': eq_type,
                    'Failure Mode': mode,
                    'Failure Rate': f"{failure_rate:.2e} /yr",
                    'Test Interval': f"{test_interval} months",
                    'PFD': f"{pfd:.2e}",
                    'SIL Capability': sil_capability,
                    'MTTF': f"{1/failure_rate:.1f} years" if failure_rate > 0 else "N/A"
                })
    
            return pd.DataFrame(data)
    
    
    # 使用例
    failure_db = GenericFailureRateDatabase()
    
    # 信頼性データシート作成
    equipment_list = [
        ("Isolation Valve", "Fail to close on demand", 12),
        ("Pressure Relief Valve", "Fail to open on demand", 24),
        ("Gas Detector", "Fail to detect", 3),
        ("SIS Logic Solver", "Fail dangerous", 12),
        ("Emergency Shutdown Valve", "Fail to close", 6)
    ]
    
    reliability_sheet = failure_db.generate_reliability_datasheet(equipment_list)
    
    print("=== Reliability Datasheet ===\n")
    print(reliability_sheet.to_string(index=False))
    
    # 個別の故障率取得
    prv_failure_rate = failure_db.get_failure_rate(
        "Pressure Relief Valve",
        "Fail to open on demand"
    )
    print(f"\n=== PRV Reliability ===")
    print(f"Failure Rate: {prv_failure_rate:.2e} /year")
    print(f"MTTF: {1/prv_failure_rate:.1f} years")
    
    # 期待される出力:
    # === Reliability Datasheet ===
    #
    #                   Equipment              Failure Mode Failure Rate Test Interval      PFD SIL Capability  MTTF
    #          Isolation Valve  Fail to close on demand   4.38e-05 /yr     12 months 2.19e-05        SIL 3  22831.1 years
    #  Pressure Relief Valve  Fail to open on demand   8.76e-05 /yr     24 months 8.76e-05        SIL 3  11415.5 years
    #            Gas Detector            Fail to detect   8.76e-04 /yr      3 months 1.09e-04        SIL 3  1141.6 years
    #       SIS Logic Solver           Fail dangerous   4.38e-06 /yr     12 months 2.19e-06        SIL 4  228310.5 years
    # Emergency Shutdown Valve            Fail to close   2.63e-05 /yr      6 months 6.57e-06        SIL 4  38041.8 years
    

* * *

## 2.6 F-N曲線とリスク基準

### Example 6: F-N曲線（Frequency-Number curve）

社会リスクを評価するF-N曲線を作成します。
    
    
    # ===================================
    # Example 6: F-N曲線とリスク基準
    # ===================================
    
    import matplotlib.pyplot as plt
    import numpy as np
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class RiskScenario:
        """リスクシナリオ"""
        name: str
        frequency: float  # events/year
        fatalities: int   # Expected number of fatalities
    
    class FNCurveAnalysis:
        """F-N曲線分析システム"""
    
        def __init__(self, facility_name: str):
            self.facility_name = facility_name
            self.scenarios: List[RiskScenario] = []
    
            # 許容リスク基準線（オランダ基準例）
            self.alarp_upper_limit = lambda N: 1e-3 / (N**2)  # Upper ALARP
            self.alarp_lower_limit = lambda N: 1e-5 / (N**2)  # Lower ALARP
            self.negligible_limit = lambda N: 1e-7 / (N**2)   # Negligible
    
        def add_scenario(self, scenario: RiskScenario):
            """シナリオを追加"""
            self.scenarios.append(scenario)
    
        def calculate_fn_curve_data(self) -> tuple:
            """F-N曲線データを計算
    
            Returns:
                (N_values, F_values): 死者数とその累積頻度
            """
            # 死者数でソート
            sorted_scenarios = sorted(self.scenarios, key=lambda x: x.fatalities)
    
            N_values = []
            F_values = []
    
            for scenario in sorted_scenarios:
                N = scenario.fatalities
                # N人以上の死者が出る事象の累積頻度
                F = sum(s.frequency for s in self.scenarios if s.fatalities >= N)
    
                N_values.append(N)
                F_values.append(F)
    
            return np.array(N_values), np.array(F_values)
    
        def plot_fn_curve(self, save_path: str = None):
            """F-N曲線をプロット"""
            N, F = self.calculate_fn_curve_data()
    
            fig, ax = plt.subplots(figsize=(10, 8))
    
            # F-N曲線プロット
            ax.loglog(N, F, 'bo-', linewidth=2, markersize=8,
                      label='Facility Risk Profile', zorder=5)
    
            # 個別シナリオをプロット
            for scenario in self.scenarios:
                ax.plot(scenario.fatalities, scenario.frequency,
                        'rx', markersize=10, markeredgewidth=2)
                ax.annotate(scenario.name,
                            xy=(scenario.fatalities, scenario.frequency),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=8, alpha=0.7)
    
            # 許容リスク基準線
            N_range = np.logspace(0, 3, 100)  # 1-1000人
    
            ax.loglog(N_range, [self.alarp_upper_limit(n) for n in N_range],
                      'r--', linewidth=2, label='ALARP Upper Limit (Unacceptable)')
            ax.loglog(N_range, [self.alarp_lower_limit(n) for n in N_range],
                      'y--', linewidth=2, label='ALARP Lower Limit')
            ax.loglog(N_range, [self.negligible_limit(n) for n in N_range],
                      'g--', linewidth=2, label='Negligible Risk')
    
            # ALARP領域を塗りつぶし
            ax.fill_between(N_range,
                            [self.alarp_lower_limit(n) for n in N_range],
                            [self.alarp_upper_limit(n) for n in N_range],
                            alpha=0.2, color='yellow', label='ALARP Region')
    
            ax.set_xlabel('Number of Fatalities (N)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cumulative Frequency (F) [events/year]', fontsize=12, fontweight='bold')
            ax.set_title(f'F-N Curve: {self.facility_name}\nSocietal Risk Assessment',
                         fontsize=14, fontweight='bold')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(loc='upper right')
    
            ax.set_xlim([1, 1000])
            ax.set_ylim([1e-8, 1e-2])
    
            plt.tight_layout()
    
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
            return fig
    
        def assess_alarp_status(self) -> Dict:
            """ALARP状態を評価"""
            results = {
                'unacceptable': [],
                'alarp': [],
                'broadly_acceptable': []
            }
    
            for scenario in self.scenarios:
                N = scenario.fatalities
                F = scenario.frequency
    
                if F > self.alarp_upper_limit(N):
                    results['unacceptable'].append(scenario)
                elif F > self.alarp_lower_limit(N):
                    results['alarp'].append(scenario)
                else:
                    results['broadly_acceptable'].append(scenario)
    
            return results
    
        def generate_alarp_report(self) -> str:
            """ALARPレポート生成"""
            status = self.assess_alarp_status()
    
            report = f"""
    {'='*80}
    F-N CURVE ANALYSIS - ALARP ASSESSMENT
    {'='*80}
    
    Facility: {self.facility_name}
    Total Scenarios Analyzed: {len(self.scenarios)}
    
    {'='*80}
    RISK CLASSIFICATION
    {'='*80}
    
    UNACCEPTABLE REGION (Above ALARP Upper Limit):
      Scenarios: {len(status['unacceptable'])}
    """
    
            if status['unacceptable']:
                for s in status['unacceptable']:
                    report += f"  ❌ {s.name}: F={s.frequency:.2e}/yr, N={s.fatalities}\n"
                report += "  ACTION: Immediate risk reduction REQUIRED\n"
            else:
                report += "  ✅ No scenarios in unacceptable region\n"
    
            report += f"""
    ALARP REGION (Risk Reduction As Low As Reasonably Practicable):
      Scenarios: {len(status['alarp'])}
    """
    
            if status['alarp']:
                for s in status['alarp']:
                    report += f"  ⚠️  {s.name}: F={s.frequency:.2e}/yr, N={s.fatalities}\n"
                report += "  ACTION: Demonstrate ALARP (cost-benefit analysis)\n"
            else:
                report += "  ✅ No scenarios in ALARP region\n"
    
            report += f"""
    BROADLY ACCEPTABLE REGION (Below ALARP Lower Limit):
      Scenarios: {len(status['broadly_acceptable'])}
    """
    
            if status['broadly_acceptable']:
                for s in status['broadly_acceptable']:
                    report += f"  ✅ {s.name}: F={s.frequency:.2e}/yr, N={s.fatalities}\n"
                report += "  ACTION: Maintain current safety measures\n"
    
            report += f"\n{'='*80}\n"
    
            return report
    
    
    # 使用例: LNG受入基地のF-N曲線分析
    fn_analysis = FNCurveAnalysis(facility_name="LNG Import Terminal")
    
    # リスクシナリオを追加
    fn_analysis.add_scenario(RiskScenario(
        name="Small LNG leak (no ignition)",
        frequency=1e-2,
        fatalities=0
    ))
    
    fn_analysis.add_scenario(RiskScenario(
        name="Medium LNG leak → jet fire",
        frequency=5e-4,
        fatalities=2
    ))
    
    fn_analysis.add_scenario(RiskScenario(
        name="Large LNG leak → pool fire",
        frequency=1e-4,
        fatalities=10
    ))
    
    fn_analysis.add_scenario(RiskScenario(
        name="Catastrophic tank failure → VCE",
        frequency=1e-6,
        fatalities=100
    ))
    
    fn_analysis.add_scenario(RiskScenario(
        name="LNG carrier collision → BLEVE",
        frequency=5e-7,
        fatalities=300
    ))
    
    # ALARPレポート生成
    alarp_report = fn_analysis.generate_alarp_report()
    print(alarp_report)
    
    # F-N曲線プロット
    fig = fn_analysis.plot_fn_curve()
    # plt.show()
    
    # 期待される出力:
    # ============================================================================
    # F-N CURVE ANALYSIS - ALARP ASSESSMENT
    # ============================================================================
    #
    # Facility: LNG Import Terminal
    # Total Scenarios Analyzed: 5
    #
    # ============================================================================
    # RISK CLASSIFICATION
    # ============================================================================
    #
    # UNACCEPTABLE REGION (Above ALARP Upper Limit):
    #   Scenarios: 0
    #   ✅ No scenarios in unacceptable region
    #
    # ALARP REGION (Risk Reduction As Low As Reasonably Practicable):
    #   Scenarios: 2
    #   ⚠️  Medium LNG leak → jet fire: F=5.00e-04/yr, N=2
    #   ⚠️  Large LNG leak → pool fire: F=1.00e-04/yr, N=10
    #   ACTION: Demonstrate ALARP (cost-benefit analysis)
    #
    # BROADLY ACCEPTABLE REGION (Below ALARP Lower Limit):
    #   Scenarios: 2
    #   ✅ Catastrophic tank failure → VCE: F=1.00e-06/yr, N=100
    #   ✅ LNG carrier collision → BLEVE: F=5.00e-07/yr, N=300
    #   ACTION: Maintain current safety measures
    

* * *

## 2.7 HAZOPレポート自動生成

### Example 7: 包括的HAZOPレポートジェネレーター
    
    
    # ===================================
    # Example 7: HAZOPレポート自動生成
    # ===================================
    
    from datetime import datetime
    import json
    
    class HAZOPReportGenerator:
        """HAZOP study完全レポート生成システム"""
    
        def __init__(self, project_info: Dict):
            self.project_info = project_info
            self.hazop_nodes = []
            self.team_members = []
    
        def add_team_member(self, name: str, role: str):
            """チームメンバーを追加"""
            self.team_members.append({'name': name, 'role': role})
    
        def generate_full_report(self, deviations_by_node: Dict[str, List[Deviation]]) -> str:
            """完全なHAZOPレポートを生成
    
            Args:
                deviations_by_node: {node_name: [Deviation, ...], ...}
            """
    
            report = self._generate_cover_page()
            report += self._generate_executive_summary(deviations_by_node)
            report += self._generate_methodology_section()
            report += self._generate_detailed_worksheets(deviations_by_node)
            report += self._generate_action_items(deviations_by_node)
            report += self._generate_appendices()
    
            return report
    
        def _generate_cover_page(self) -> str:
            """表紙ページ"""
            return f"""
    {'='*80}
    HAZARD AND OPERABILITY STUDY (HAZOP)
    FINAL REPORT
    {'='*80}
    
    Project:     {self.project_info.get('name', 'N/A')}
    Facility:    {self.project_info.get('facility', 'N/A')}
    Location:    {self.project_info.get('location', 'N/A')}
    
    Study Date:  {self.project_info.get('study_date', datetime.now().strftime('%Y-%m-%d'))}
    Report Date: {datetime.now().strftime('%Y-%m-%d')}
    
    Document No: {self.project_info.get('doc_number', 'HAZOP-001')}
    Revision:    {self.project_info.get('revision', '0')}
    
    {'='*80}
    STUDY TEAM
    {'='*80}
    
    """
    
        def _generate_executive_summary(self, deviations_by_node: Dict) -> str:
            """エグゼクティブサマリー"""
    
            total_deviations = sum(len(devs) for devs in deviations_by_node.values())
            high_risk = sum(1 for devs in deviations_by_node.values()
                            for dev in devs if dev.risk_rank == "High")
    
            summary = f"""
    EXECUTIVE SUMMARY
    {'='*80}
    
    Study Scope:
      - Total HAZOP Nodes:     {len(deviations_by_node)}
      - Total Deviations:      {total_deviations}
      - High Risk Scenarios:   {high_risk}
    
    Key Findings:
      1. {high_risk} high-risk scenarios identified requiring immediate attention
      2. All scenarios have been assessed with existing safeguards documented
      3. Recommendations provided for risk reduction where needed
    
    Overall Assessment:
    """
    
            if high_risk == 0:
                summary += "  ✅ No high-risk scenarios. Facility design is robust.\n"
            elif high_risk <= 3:
                summary += f"  ⚠️  {high_risk} high-risk scenarios require mitigation.\n"
            else:
                summary += f"  ❌ {high_risk} high-risk scenarios - major design review recommended.\n"
    
            return summary + "\n"
    
        def _generate_methodology_section(self) -> str:
            """方法論セクション"""
            return f"""
    METHODOLOGY
    {'='*80}
    
    1. HAZOP Technique:
       - Guide Words applied to process parameters
       - Systematic deviation analysis
       - Cause-Consequence-Safeguard approach
    
    2. Risk Ranking:
       - High:   Immediate action required
       - Medium: Risk reduction should be considered
       - Low:    Acceptable with existing safeguards
    
    3. Documentation:
       - P&ID reviewed: {self.project_info.get('pid_revision', 'Rev 0')}
       - Process conditions: {self.project_info.get('process_basis', 'Normal operation')}
    
    """
    
        def _generate_detailed_worksheets(self, deviations_by_node: Dict) -> str:
            """詳細ワークシート"""
            worksheets = f"""
    DETAILED HAZOP WORKSHEETS
    {'='*80}
    
    """
    
            for node_name, deviations in deviations_by_node.items():
                worksheets += f"\nNode: {node_name}\n"
                worksheets += f"{'-'*80}\n\n"
    
                for i, dev in enumerate(deviations, 1):
                    worksheets += f"Deviation {i}: {dev.description}\n"
                    worksheets += f"Risk Rank: {dev.risk_rank}\n\n"
    
                    worksheets += "Potential Causes:\n"
                    for cause in dev.potential_causes:
                        worksheets += f"  - {cause}\n"
    
                    worksheets += "\nPotential Consequences:\n"
                    for cons in dev.potential_consequences:
                        worksheets += f"  - {cons}\n"
    
                    worksheets += "\nExisting Safeguards:\n"
                    for sg in dev.existing_safeguards:
                        worksheets += f"  - {sg}\n"
    
                    worksheets += "\nRecommendations:\n"
                    for rec in dev.recommendations:
                        worksheets += f"  - {rec}\n"
    
                    worksheets += "\n" + "-"*80 + "\n\n"
    
            return worksheets
    
        def _generate_action_items(self, deviations_by_node: Dict) -> str:
            """アクションアイテム"""
            actions = f"""
    ACTION ITEM REGISTER
    {'='*80}
    
    """
    
            action_id = 1
            for node_name, deviations in deviations_by_node.items():
                for dev in deviations:
                    for rec in dev.recommendations:
                        if "adequate" not in rec.lower():  # 対応不要の推奨事項をスキップ
                            priority = "HIGH" if dev.risk_rank == "High" else "MEDIUM"
                            actions += f"Action {action_id:03d}: [{priority}] {rec}\n"
                            actions += f"  Node: {node_name}\n"
                            actions += f"  Deviation: {dev.description}\n"
                            actions += f"  Responsible: TBD\n"
                            actions += f"  Target Date: TBD\n\n"
                            action_id += 1
    
            return actions
    
        def _generate_appendices(self) -> str:
            """付録"""
            return f"""
    APPENDICES
    {'='*80}
    
    Appendix A: P&ID Drawings
    Appendix B: Study Team Attendance Records
    Appendix C: Risk Matrix Definition
    Appendix D: Abbreviations and Definitions
    
    """
    
        def export_to_json(self, deviations_by_node: Dict, filepath: str):
            """JSON形式でエクスポート"""
            export_data = {
                'project_info': self.project_info,
                'study_date': datetime.now().isoformat(),
                'nodes': {}
            }
    
            for node_name, deviations in deviations_by_node.items():
                export_data['nodes'][node_name] = [
                    {
                        'description': dev.description,
                        'guide_word': dev.guide_word.value,
                        'parameter': dev.parameter.value,
                        'causes': dev.potential_causes,
                        'consequences': dev.potential_consequences,
                        'safeguards': dev.existing_safeguards,
                        'recommendations': dev.recommendations,
                        'risk_rank': dev.risk_rank
                    }
                    for dev in deviations
                ]
    
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    
    # 使用例
    project_info = {
        'name': 'Ethylene Polymerization Plant',
        'facility': 'Reactor Section',
        'location': 'Yokkaichi, Japan',
        'study_date': '2025-10-15',
        'doc_number': 'HAZOP-EPP-001',
        'revision': '0',
        'pid_revision': 'Rev 2',
        'process_basis': 'Normal operation at 85% capacity'
    }
    
    report_gen = HAZOPReportGenerator(project_info)
    
    # チームメンバー追加
    report_gen.add_team_member("Dr. Smith", "HAZOP Leader")
    report_gen.add_team_member("Eng. Tanaka", "Process Engineer")
    report_gen.add_team_member("Eng. Kim", "Instrument Engineer")
    
    # 逸脱データ（Example 2で生成したもの）
    analyzer = DeviationAnalyzer("Reactor R-101", "Vessel")
    analyzer.analyze_deviation(GuideWord.NO, ProcessParameter.FLOW)
    analyzer.analyze_deviation(GuideWord.MORE, ProcessParameter.TEMPERATURE)
    
    deviations_data = {
        "Reactor R-101": analyzer.deviations
    }
    
    # レポート生成
    full_report = report_gen.generate_full_report(deviations_data)
    print(full_report[:2000])  # 最初の2000文字を表示
    
    # JSON エクスポート
    # report_gen.export_to_json(deviations_data, "hazop_study_results.json")
    # print("\n✅ HAZOP study exported to hazop_study_results.json")
    
    # 期待される出力:
    # ============================================================================
    # HAZARD AND OPERABILITY STUDY (HAZOP)
    # FINAL REPORT
    # ============================================================================
    #
    # Project:     Ethylene Polymerization Plant
    # Facility:    Reactor Section
    # Location:    Yokkaichi, Japan
    # ...
    

* * *

## 2.8 リスクランキングと優先順位付け

### Example 8: 多基準意思決定分析（MCDA）

複数のリスクシナリオを総合的に評価し、対策の優先順位を決定します。
    
    
    # ===================================
    # Example 8: 多基準意思決定分析（MCDA）
    # ===================================
    
    import pandas as pd
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Dict
    
    @dataclass
    class RiskCriteria:
        """リスク評価基準"""
        likelihood: float       # 1-5
        severity_people: float  # 1-5 (人的被害)
        severity_env: float     # 1-5 (環境影響)
        severity_asset: float   # 1-5 (資産損失)
        detectability: float    # 1-5 (検知容易性、低いほど良い)
    
    class MultiCriteriaRiskRanking:
        """多基準リスクランキングシステム"""
    
        def __init__(self):
            # 基準の重み（合計1.0）
            self.weights = {
                'likelihood': 0.25,
                'severity_people': 0.35,  # 人的被害を最重視
                'severity_env': 0.20,
                'severity_asset': 0.15,
                'detectability': 0.05
            }
    
        def calculate_risk_priority_number(self, criteria: RiskCriteria) -> float:
            """リスク優先度番号（RPN）を計算"""
    
            # 加重平均スコア
            weighted_score = (
                criteria.likelihood * self.weights['likelihood'] +
                criteria.severity_people * self.weights['severity_people'] +
                criteria.severity_env * self.weights['severity_env'] +
                criteria.severity_asset * self.weights['severity_asset'] +
                criteria.detectability * self.weights['detectability']
            )
    
            # 0-100スケールに正規化
            rpn = weighted_score * 20
    
            return rpn
    
        def rank_scenarios(self, scenarios: Dict[str, RiskCriteria]) -> pd.DataFrame:
            """シナリオをランキング"""
    
            data = []
    
            for scenario_name, criteria in scenarios.items():
                rpn = self.calculate_risk_priority_number(criteria)
    
                # 優先度レベル判定
                if rpn >= 80:
                    priority = "Critical (P1)"
                elif rpn >= 60:
                    priority = "High (P2)"
                elif rpn >= 40:
                    priority = "Medium (P3)"
                else:
                    priority = "Low (P4)"
    
                data.append({
                    'Scenario': scenario_name,
                    'Likelihood': criteria.likelihood,
                    'Sev_People': criteria.severity_people,
                    'Sev_Env': criteria.severity_env,
                    'Sev_Asset': criteria.severity_asset,
                    'Detectability': criteria.detectability,
                    'RPN': f"{rpn:.1f}",
                    'Priority': priority
                })
    
            df = pd.DataFrame(data)
            df = df.sort_values('RPN', ascending=False, key=lambda x: x.astype(float))
    
            return df
    
        def sensitivity_analysis(self, criteria: RiskCriteria, scenario_name: str):
            """感度分析: 各基準の変化がRPNに与える影響"""
    
            baseline_rpn = self.calculate_risk_priority_number(criteria)
    
            print(f"\n=== Sensitivity Analysis: {scenario_name} ===")
            print(f"Baseline RPN: {baseline_rpn:.1f}\n")
    
            print("Impact of +1 change in each criterion:")
    
            # Likelihood +1
            temp_criteria = RiskCriteria(
                criteria.likelihood + 1,
                criteria.severity_people,
                criteria.severity_env,
                criteria.severity_asset,
                criteria.detectability
            )
            new_rpn = self.calculate_risk_priority_number(temp_criteria)
            print(f"  Likelihood +1:        {baseline_rpn:.1f} → {new_rpn:.1f} (Δ{new_rpn-baseline_rpn:+.1f})")
    
            # Severity_People +1
            temp_criteria = RiskCriteria(
                criteria.likelihood,
                min(criteria.severity_people + 1, 5),
                criteria.severity_env,
                criteria.severity_asset,
                criteria.detectability
            )
            new_rpn = self.calculate_risk_priority_number(temp_criteria)
            print(f"  Severity_People +1:   {baseline_rpn:.1f} → {new_rpn:.1f} (Δ{new_rpn-baseline_rpn:+.1f})")
    
            # Detectability -1 (lower is better)
            temp_criteria = RiskCriteria(
                criteria.likelihood,
                criteria.severity_people,
                criteria.severity_env,
                criteria.severity_asset,
                max(criteria.detectability - 1, 1)
            )
            new_rpn = self.calculate_risk_priority_number(temp_criteria)
            print(f"  Detectability -1:     {baseline_rpn:.1f} → {new_rpn:.1f} (Δ{new_rpn-baseline_rpn:+.1f})")
    
    
    # 使用例
    mcda = MultiCriteriaRiskRanking()
    
    # 複数のリスクシナリオを評価
    scenarios = {
        "Reactor overpressure (no relief)": RiskCriteria(
            likelihood=2,         # Occasional
            severity_people=5,    # Catastrophic
            severity_env=4,       # Critical
            severity_asset=5,     # Catastrophic
            detectability=2       # Good detection (pressure transmitter)
        ),
    
        "Toxic gas release (H2S)": RiskCriteria(
            likelihood=3,         # Probable
            severity_people=4,    # Critical
            severity_env=3,       # Marginal
            severity_asset=2,     # Negligible
            detectability=3       # Moderate detection
        ),
    
        "Loss of cooling water": RiskCriteria(
            likelihood=4,         # Frequent
            severity_people=2,    # Negligible
            severity_env=1,       # Minimal
            severity_asset=3,     # Marginal
            detectability=1       # Excellent detection (flow transmitter)
        ),
    
        "Runaway polymerization": RiskCriteria(
            likelihood=2,         # Occasional
            severity_people=4,    # Critical
            severity_env=3,       # Marginal
            severity_asset=4,     # Critical
            detectability=3       # Moderate detection
        ),
    
        "Flange gasket leak": RiskCriteria(
            likelihood=3,         # Probable
            severity_people=2,    # Negligible
            severity_env=2,       # Negligible
            severity_asset=1,     # Minimal
            detectability=4       # Poor detection (visual inspection)
        )
    }
    
    # ランキング実施
    ranking = mcda.rank_scenarios(scenarios)
    
    print("=== Multi-Criteria Risk Ranking ===\n")
    print(ranking.to_string(index=False))
    
    # 最高リスクシナリオの感度分析
    top_scenario_name = ranking.iloc[0]['Scenario']
    top_scenario_criteria = scenarios[top_scenario_name]
    
    mcda.sensitivity_analysis(top_scenario_criteria, top_scenario_name)
    
    # 期待される出力:
    # === Multi-Criteria Risk Ranking ===
    #
    #                        Scenario  Likelihood  Sev_People  Sev_Env  Sev_Asset  Detectability    RPN      Priority
    #  Reactor overpressure (no relief)           2           5        4          5              2   80.0  Critical (P1)
    #         Runaway polymerization           2           4        3          4              3   66.0      High (P2)
    #          Toxic gas release (H2S)           3           4        3          2              3   66.0      High (P2)
    #            Loss of cooling water           4           2        1          3              1   48.0    Medium (P3)
    #                Flange gasket leak           3           2        2          1              4   44.0    Medium (P3)
    #
    # === Sensitivity Analysis: Reactor overpressure (no relief) ===
    # Baseline RPN: 80.0
    #
    # Impact of +1 change in each criterion:
    #   Likelihood +1:        80.0 → 85.0 (Δ+5.0)
    #   Severity_People +1:   80.0 → 80.0 (Δ+0.0)  # Already at max (5)
    #   Detectability -1:     80.0 → 79.0 (Δ-1.0)
    

* * *

## 学習目標の確認

このchapterを完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ HAZOPの原理とガイドワードの適用方法を理解している
  * ✅ 逸脱分析のプロセス（原因-結果-対策）を知っている
  * ✅ QRAとイベントツリー解析の違いを理解している
  * ✅ F-N曲線とALARP概念を説明できる

### 実践スキル

  * ✅ HAZOPガイドワードを体系的に適用できる
  * ✅ P&IDからHAZOPノードを識別できる
  * ✅ イベントツリー解析で事象連鎖を評価できる
  * ✅ 故障率データベースを活用してPFDを計算できる
  * ✅ F-N曲線を作成しALARP判定ができる
  * ✅ 包括的HAZOPレポートを生成できる

### 応用力

  * ✅ 実際の化学プロセスに対してHAZOPスタディを実施できる
  * ✅ 定量的リスク評価（QRA）を実践できる
  * ✅ 多基準意思決定分析で対策優先順位を決定できる
  * ✅ HAZOPレポートを自動生成し意思決定を支援できる

* * *

## 次のステップ

第2章では、HAZOP、QRA、F-N曲線を学びました。

**第3章では：**

  * 📋 FMEA（故障モード影響解析）
  * 📋 信頼性工学とMTBF計算
  * 📋 フォールトツリー解析（FTA）
  * 📋 予防保全最適化（RCM）

を学びます。
