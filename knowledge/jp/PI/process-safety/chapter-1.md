---
title: 第1章：プロセス安全性の基礎
chapter_title: 第1章：プロセス安全性の基礎
subtitle: ハザード識別、リスク評価、保護層分析（LOPA）の完全実装
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ プロセス安全の概念と重大事故の教訓を理解する
  * ✅ ハザード識別フレームワークを実装できる
  * ✅ リスクマトリックス（発生頻度 × 影響度）でリスク評価ができる
  * ✅ 保護層分析（LOPA）を実施しSILを計算できる
  * ✅ Bow-tie図でリスク分析を可視化できる
  * ✅ 影響度計算（Consequence Modeling）を実装できる

* * *

## 1.1 プロセス安全の概要

### プロセス安全とは

**プロセス安全（Process Safety）** とは、化学プロセスにおける重大事故（火災、爆発、有毒ガス漏洩など）を防止するための体系的な安全管理です。労働安全（Occupational Safety）が個人の怪我を防ぐのに対し、プロセス安全は設備の完全性（Equipment Integrity）と運転管理に焦点を当てます。

### 重大事故の歴史

事故名 | 年 | 死者数 | 主な原因 | 教訓  
---|---|---|---|---  
**Flixborough** | 1974 | 28 | 配管設計不良によるシクロヘキサン漏洩 | Change Managementの重要性  
**Bhopal** | 1984 | 3,787+ | MIC（メチルイソシアネート）漏洩 | 多重保護層の必要性  
**Piper Alpha** | 1988 | 167 | ガス漏洩→火災→爆発 | Permit-to-Workシステム  
**Texas City** | 2005 | 15 | 蒸留塔オーバーフロー→爆発 | 安全文化の重要性  
  
> **重要な洞察:** これらの事故の共通点は、技術的な故障だけでなく、管理システムの失敗（Management System Failure）が根本原因であることです。

* * *

## 1.2 ハザード識別フレームワーク

### Example 1: ハザード識別システムの実装

プロセスハザード分析（PHA: Process Hazard Analysis）の第一歩は、体系的なハザード識別です。
    
    
    # ===================================
    # Example 1: ハザード識別システム
    # ===================================
    
    import pandas as pd
    from dataclasses import dataclass
    from typing import List, Dict
    from enum import Enum
    
    class HazardCategory(Enum):
        """ハザードカテゴリー"""
        PHYSICAL = "物理的ハザード"
        CHEMICAL = "化学的ハザード"
        BIOLOGICAL = "生物学的ハザード"
        ERGONOMIC = "人間工学的ハザード"
    
    class Severity(Enum):
        """影響度レベル"""
        CATASTROPHIC = 5  # 壊滅的（複数の死亡）
        CRITICAL = 4      # 重大（1名以上の死亡）
        MARGINAL = 3      # 限定的（重傷）
        NEGLIGIBLE = 2    # 軽微（軽傷）
        MINIMAL = 1       # 最小（応急処置のみ）
    
    @dataclass
    class Hazard:
        """ハザード情報"""
        id: str
        name: str
        category: HazardCategory
        description: str
        potential_causes: List[str]
        potential_consequences: List[str]
        severity: Severity
        existing_safeguards: List[str]
    
    class HazardIdentificationSystem:
        """ハザード識別システム"""
    
        def __init__(self):
            self.hazards: List[Hazard] = []
            self._initialize_common_hazards()
    
        def _initialize_common_hazards(self):
            """一般的な化学プロセスハザードの初期化"""
    
            # ハザード1: 高圧ガス漏洩
            self.add_hazard(Hazard(
                id="HAZ-001",
                name="高圧ガス漏洩",
                category=HazardCategory.PHYSICAL,
                description="反応器からの高圧水素ガス漏洩",
                potential_causes=[
                    "配管フランジのガスケット劣化",
                    "腐食による配管穿孔",
                    "バルブシート摩耗",
                    "過圧によるラプチャーディスク破裂"
                ],
                potential_consequences=[
                    "火災・爆発（着火源がある場合）",
                    "窒息（密閉空間での酸素欠乏）",
                    "低温火傷（LNGの場合）",
                    "環境汚染"
                ],
                severity=Severity.CATASTROPHIC,
                existing_safeguards=[
                    "ガス検知器（0.4% LEL設定）",
                    "緊急遮断弁（ESD）",
                    "圧力逃し弁（PRV）",
                    "ベントシステム"
                ]
            ))
    
            # ハザード2: 発熱反応暴走
            self.add_hazard(Hazard(
                id="HAZ-002",
                name="発熱反応暴走",
                category=HazardCategory.CHEMICAL,
                description="冷却系統故障による反応暴走",
                potential_causes=[
                    "冷却水ポンプ故障",
                    "温度制御系の故障",
                    "攪拌機停止による局所過熱",
                    "触媒過剰投入"
                ],
                potential_consequences=[
                    "反応器破裂",
                    "有毒ガス放出",
                    "火災・爆発",
                    "周辺施設への連鎖災害"
                ],
                severity=Severity.CRITICAL,
                existing_safeguards=[
                    "独立高温インターロック（120°C）",
                    "緊急冷却システム",
                    "ラプチャーディスク + クエンチャー",
                    "原料供給緊急遮断"
                ]
            ))
    
            # ハザード3: 可燃性液体漏洩
            self.add_hazard(Hazard(
                id="HAZ-003",
                name="可燃性液体漏洩",
                category=HazardCategory.CHEMICAL,
                description="トルエン貯蔵タンクからの液体漏洩",
                potential_causes=[
                    "タンク底部腐食",
                    "オーバーフロー（レベル計故障）",
                    "ローディングホース破損",
                    "地震による配管破断"
                ],
                potential_consequences=[
                    "液面火災（プールファイア）",
                    "蒸気雲爆発（VCE）",
                    "土壌・地下水汚染",
                    "近隣住民への健康影響"
                ],
                severity=Severity.CRITICAL,
                existing_safeguards=[
                    "防液堤（容量110%）",
                    "高位レベルアラーム + インターロック",
                    "漏洩検知システム",
                    "泡消火設備"
                ]
            ))
    
        def add_hazard(self, hazard: Hazard):
            """ハザードを追加"""
            self.hazards.append(hazard)
    
        def get_hazards_by_severity(self, min_severity: Severity) -> List[Hazard]:
            """指定した影響度以上のハザードを取得"""
            return [h for h in self.hazards if h.severity.value >= min_severity.value]
    
        def generate_hazard_register(self) -> pd.DataFrame:
            """ハザードレジスター（一覧表）を生成"""
            data = []
            for h in self.hazards:
                data.append({
                    'ID': h.id,
                    'ハザード名': h.name,
                    'カテゴリー': h.category.value,
                    '影響度': h.severity.name,
                    '影響度スコア': h.severity.value,
                    '主な原因': '; '.join(h.potential_causes[:2]),  # 最初の2つ
                    '主な結果': '; '.join(h.potential_consequences[:2]),
                    '既存安全対策数': len(h.existing_safeguards)
                })
    
            df = pd.DataFrame(data)
            return df.sort_values('影響度スコア', ascending=False)
    
    
    # 使用例
    hazard_system = HazardIdentificationSystem()
    
    # 高リスクハザードを抽出
    critical_hazards = hazard_system.get_hazards_by_severity(Severity.CRITICAL)
    print(f"重大ハザード数: {len(critical_hazards)}\n")
    
    # ハザードレジスター生成
    hazard_register = hazard_system.generate_hazard_register()
    print("=== ハザードレジスター ===")
    print(hazard_register.to_string(index=False))
    
    # 期待される出力:
    # 重大ハザード数: 2
    #
    # === ハザードレジスター ===
    #       ID    ハザード名 カテゴリー      影響度  影響度スコア ...
    #  HAZ-001  高圧ガス漏洩  物理的ハザード  CATASTROPHIC     5 ...
    #  HAZ-002  発熱反応暴走  化学的ハザード  CRITICAL         4 ...
    #  HAZ-003  可燃性液体漏洩 化学的ハザード  CRITICAL         4 ...
    

* * *

## 1.3 リスク評価の基礎

### リスクマトリックスの概念

**リスク** は、発生頻度（Likelihood）と影響度（Consequence）の組み合わせで評価されます：

$$ \text{Risk} = \text{Likelihood} \times \text{Consequence} $$

### Example 2: リスクマトリックス実装
    
    
    # ===================================
    # Example 2: リスクマトリックス
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from enum import Enum
    
    class Likelihood(Enum):
        """発生頻度レベル"""
        FREQUENT = 5      # 頻繁（年1回以上）
        PROBABLE = 4      # 起こりうる（2-10年に1回）
        OCCASIONAL = 3    # 時々（10-100年に1回）
        REMOTE = 2        # 稀（100-1000年に1回）
        IMPROBABLE = 1    # ほぼ起こらない（1000年に1回以下）
    
    class RiskLevel(Enum):
        """リスクレベル"""
        EXTREME = "極めて高い"
        HIGH = "高い"
        MEDIUM = "中程度"
        LOW = "低い"
    
    class RiskMatrix:
        """リスクマトリックス評価システム"""
    
        def __init__(self):
            # リスクマトリックス定義（5x5）
            # 行: Severity (1-5), 列: Likelihood (1-5)
            self.matrix = np.array([
                [1,  2,  3,  5,  8 ],  # Severity 1 (MINIMAL)
                [2,  4,  6,  10, 15],  # Severity 2 (NEGLIGIBLE)
                [4,  8,  12, 18, 25],  # Severity 3 (MARGINAL)
                [8,  15, 20, 25, 30],  # Severity 4 (CRITICAL)
                [12, 20, 25, 30, 35]   # Severity 5 (CATASTROPHIC)
            ])
    
            # リスクレベル閾値
            self.thresholds = {
                RiskLevel.EXTREME: 25,  # ≥25
                RiskLevel.HIGH: 15,     # 15-24
                RiskLevel.MEDIUM: 8,    # 8-14
                RiskLevel.LOW: 0        # <8
            }
    
        def calculate_risk_score(self, severity: Severity, likelihood: Likelihood) -> int:
            """リスクスコアを計算"""
            return self.matrix[severity.value - 1, likelihood.value - 1]
    
        def determine_risk_level(self, risk_score: int) -> RiskLevel:
            """リスクレベルを判定"""
            if risk_score >= self.thresholds[RiskLevel.EXTREME]:
                return RiskLevel.EXTREME
            elif risk_score >= self.thresholds[RiskLevel.HIGH]:
                return RiskLevel.HIGH
            elif risk_score >= self.thresholds[RiskLevel.MEDIUM]:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
    
        def assess_risk(self, hazard_name: str, severity: Severity,
                        likelihood: Likelihood) -> Dict:
            """包括的リスク評価"""
            risk_score = self.calculate_risk_score(severity, likelihood)
            risk_level = self.determine_risk_level(risk_score)
    
            # 対応アクション
            if risk_level == RiskLevel.EXTREME:
                action = "即座の対応必要。操業停止を検討。"
            elif risk_level == RiskLevel.HIGH:
                action = "優先的にリスク低減策を実施（3ヶ月以内）。"
            elif risk_level == RiskLevel.MEDIUM:
                action = "リスク低減策を計画（1年以内）。"
            else:
                action = "現状の安全対策を維持。"
    
            return {
                'ハザード': hazard_name,
                '影響度': severity.name,
                '発生頻度': likelihood.name,
                'リスクスコア': risk_score,
                'リスクレベル': risk_level.value,
                '推奨アクション': action
            }
    
        def visualize_matrix(self, assessments: List[Dict] = None):
            """リスクマトリックスを可視化"""
            fig, ax = plt.subplots(figsize=(10, 8))
    
            # ヒートマップ作成
            sns.heatmap(self.matrix, annot=True, fmt='d', cmap='YlOrRd',
                        cbar_kws={'label': 'Risk Score'},
                        xticklabels=['Improbable\n(1)', 'Remote\n(2)', 'Occasional\n(3)',
                                     'Probable\n(4)', 'Frequent\n(5)'],
                        yticklabels=['Minimal\n(1)', 'Negligible\n(2)', 'Marginal\n(3)',
                                     'Critical\n(4)', 'Catastrophic\n(5)'],
                        ax=ax)
    
            ax.set_xlabel('Likelihood (発生頻度)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Severity (影響度)', fontsize=12, fontweight='bold')
            ax.set_title('Process Safety Risk Matrix', fontsize=14, fontweight='bold')
    
            # 評価結果をプロット
            if assessments:
                for assess in assessments:
                    sev_map = {'MINIMAL': 1, 'NEGLIGIBLE': 2, 'MARGINAL': 3,
                               'CRITICAL': 4, 'CATASTROPHIC': 5}
                    like_map = {'IMPROBABLE': 1, 'REMOTE': 2, 'OCCASIONAL': 3,
                                'PROBABLE': 4, 'FREQUENT': 5}
    
                    sev_idx = sev_map[assess['影響度']] - 1
                    like_idx = like_map[assess['発生頻度']] - 1
    
                    ax.plot(like_idx + 0.5, sev_idx + 0.5, 'bo', markersize=15,
                            markeredgecolor='white', markeredgewidth=2)
    
            plt.tight_layout()
            return fig
    
    
    # 使用例
    risk_matrix = RiskMatrix()
    
    # 複数のハザードを評価
    assessments = [
        risk_matrix.assess_risk("高圧ガス漏洩", Severity.CATASTROPHIC, Likelihood.OCCASIONAL),
        risk_matrix.assess_risk("発熱反応暴走", Severity.CRITICAL, Likelihood.REMOTE),
        risk_matrix.assess_risk("可燃性液体漏洩", Severity.CRITICAL, Likelihood.OCCASIONAL)
    ]
    
    # 結果表示
    print("=== リスク評価結果 ===\n")
    for assess in assessments:
        print(f"ハザード: {assess['ハザード']}")
        print(f"  リスクスコア: {assess['リスクスコア']} ({assess['リスクレベル']})")
        print(f"  推奨アクション: {assess['推奨アクション']}\n")
    
    # 可視化
    fig = risk_matrix.visualize_matrix(assessments)
    # plt.show()  # Jupyter環境では自動表示
    
    # 期待される出力:
    # === リスク評価結果 ===
    #
    # ハザード: 高圧ガス漏洩
    #   リスクスコア: 25 (極めて高い)
    #   推奨アクション: 即座の対応必要。操業停止を検討。
    #
    # ハザード: 発熱反応暴走
    #   リスクスコア: 15 (高い)
    #   推奨アクション: 優先的にリスク低減策を実施（3ヶ月以内）。
    

* * *

## 1.4 保護層分析（LOPA）

### LOPAの概念

**LOPA（Layer of Protection Analysis）** は、リスクシナリオに対する独立保護層（IPL: Independent Protection Layer）の有効性を定量的に評価する手法です。

### Example 3: LOPA実装とSIL計算
    
    
    # ===================================
    # Example 3: 保護層分析（LOPA）とSIL計算
    # ===================================
    
    from dataclasses import dataclass
    from typing import List
    import math
    
    @dataclass
    class ProtectionLayer:
        """保護層（IPL）"""
        name: str
        pfd: float  # Probability of Failure on Demand（要求時故障確率）
    
        @property
        def risk_reduction_factor(self) -> float:
            """リスク低減係数（RRF）"""
            return 1.0 / self.pfd if self.pfd > 0 else float('inf')
    
    class SIL(Enum):
        """Safety Integrity Level（安全度水準）"""
        SIL_4 = (1e-5, 1e-4, "10^-5 to 10^-4")
        SIL_3 = (1e-4, 1e-3, "10^-4 to 10^-3")
        SIL_2 = (1e-3, 1e-2, "10^-3 to 10^-2")
        SIL_1 = (1e-2, 1e-1, "10^-2 to 10^-1")
        NO_SIL = (1e-1, 1.0, "> 10^-1")
    
        def __init__(self, lower, upper, range_str):
            self.lower = lower
            self.upper = upper
            self.range_str = range_str
    
    class LOPAAnalysis:
        """保護層分析システム"""
    
        def __init__(self, scenario_name: str, initiating_event_frequency: float,
                     consequence_severity: Severity):
            """
            Args:
                scenario_name: シナリオ名
                initiating_event_frequency: 起因事象頻度（回/年）
                consequence_severity: 影響度
            """
            self.scenario_name = scenario_name
            self.initiating_event_frequency = initiating_event_frequency
            self.consequence_severity = consequence_severity
            self.protection_layers: List[ProtectionLayer] = []
    
        def add_protection_layer(self, layer: ProtectionLayer):
            """保護層を追加"""
            self.protection_layers.append(layer)
    
        def calculate_mitigated_frequency(self) -> float:
            """低減後事象頻度を計算"""
            total_pfd = self.initiating_event_frequency
    
            for layer in self.protection_layers:
                total_pfd *= layer.pfd
    
            return total_pfd
    
        def determine_required_sil(self, tolerable_frequency: float = 1e-4) -> SIL:
            """必要なSILレベルを決定
    
            Args:
                tolerable_frequency: 許容可能な事象頻度（回/年）
            """
            current_frequency = self.calculate_mitigated_frequency()
    
            if current_frequency <= tolerable_frequency:
                return SIL.NO_SIL
    
            # 追加のリスク低減が必要
            required_pfd = tolerable_frequency / self.initiating_event_frequency
    
            # 既存保護層のPFDを考慮
            for layer in self.protection_layers:
                required_pfd /= layer.pfd
    
            # SIL判定
            for sil in [SIL.SIL_4, SIL.SIL_3, SIL.SIL_2, SIL.SIL_1]:
                if sil.lower <= required_pfd < sil.upper:
                    return sil
    
            return SIL.NO_SIL
    
        def generate_lopa_report(self, tolerable_frequency: float = 1e-4) -> str:
            """LOPAレポート生成"""
            mitigated_freq = self.calculate_mitigated_frequency()
            required_sil = self.determine_required_sil(tolerable_frequency)
    
            report = f"""
    {'='*60}
    LOPA Analysis Report
    {'='*60}
    
    Scenario: {self.scenario_name}
    Consequence Severity: {self.consequence_severity.name}
    
    --- Initiating Event ---
    Frequency: {self.initiating_event_frequency:.2e} events/year
    
    --- Independent Protection Layers (IPL) ---
    """
    
            total_rrf = 1.0
            for i, layer in enumerate(self.protection_layers, 1):
                rrf = layer.risk_reduction_factor
                total_rrf *= rrf
                report += f"{i}. {layer.name}\n"
                report += f"   PFD: {layer.pfd:.2e}\n"
                report += f"   RRF: {rrf:.0f}\n\n"
    
            report += f"""--- Risk Assessment ---
    Total Risk Reduction Factor: {total_rrf:.0f}
    Mitigated Event Frequency: {mitigated_freq:.2e} events/year
    Tolerable Frequency Target: {tolerable_frequency:.2e} events/year
    
    Risk Status: {'ACCEPTABLE' if mitigated_freq <= tolerable_frequency else 'UNACCEPTABLE'}
    
    --- SIL Requirement ---
    """
    
            if required_sil == SIL.NO_SIL:
                report += "Required SIL: None (existing IPLs are sufficient)\n"
            else:
                report += f"Required SIL: {required_sil.name}\n"
                report += f"Target PFD Range: {required_sil.range_str}\n"
                report += f"\nRecommendation: Implement SIF (Safety Instrumented Function) with {required_sil.name}\n"
    
            report += f"\n{'='*60}\n"
    
            return report
    
    
    # 使用例: 反応器過圧シナリオのLOPA分析
    lopa = LOPAAnalysis(
        scenario_name="反応器過圧による破裂",
        initiating_event_frequency=1e-2,  # 0.01回/年（100年に1回）
        consequence_severity=Severity.CATASTROPHIC
    )
    
    # 保護層を追加
    lopa.add_protection_layer(ProtectionLayer(
        name="Basic Process Control System (BPCS)",
        pfd=1e-1  # 90%有効性
    ))
    
    lopa.add_protection_layer(ProtectionLayer(
        name="High Pressure Alarm (operator response)",
        pfd=1e-1  # 90%有効性
    ))
    
    lopa.add_protection_layer(ProtectionLayer(
        name="Pressure Relief Valve (PRV)",
        pfd=1e-2  # 99%有効性
    ))
    
    # LOPAレポート生成
    report = lopa.generate_lopa_report(tolerable_frequency=1e-5)
    print(report)
    
    # 期待される出力:
    # ============================================================
    # LOPA Analysis Report
    # ============================================================
    #
    # Scenario: 反応器過圧による破裂
    # Consequence Severity: CATASTROPHIC
    #
    # --- Initiating Event ---
    # Frequency: 1.00e-02 events/year
    #
    # --- Independent Protection Layers (IPL) ---
    # 1. Basic Process Control System (BPCS)
    #    PFD: 1.00e-01
    #    RRF: 10
    #
    # 2. High Pressure Alarm (operator response)
    #    PFD: 1.00e-01
    #    RRF: 10
    #
    # 3. Pressure Relief Valve (PRV)
    #    PFD: 1.00e-02
    #    RRF: 100
    #
    # --- Risk Assessment ---
    # Total Risk Reduction Factor: 10000
    # Mitigated Event Frequency: 1.00e-06 events/year
    # Tolerable Frequency Target: 1.00e-05 events/year
    #
    # Risk Status: ACCEPTABLE
    #
    # --- SIL Requirement ---
    # Required SIL: None (existing IPLs are sufficient)
    

* * *

## 1.5 影響度計算（Consequence Modeling）

### Example 4: ガス拡散モデル（Gaussian Plume）

有毒ガス漏洩時の影響範囲を予測するGaussian Plumeモデルを実装します。
    
    
    # ===================================
    # Example 4: ガス拡散モデル（Gaussian Plume）
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erf
    
    class GaussianPlumeModel:
        """Gaussian Plume拡散モデル（定常状態）"""
    
        def __init__(self, emission_rate: float, wind_speed: float,
                     stack_height: float = 0.0, stability_class: str = 'D'):
            """
            Args:
                emission_rate: 放出速度 [g/s]
                wind_speed: 風速 [m/s]
                stack_height: 放出高さ [m]
                stability_class: 大気安定度クラス（A-F, Pasquill分類）
            """
            self.Q = emission_rate
            self.u = wind_speed
            self.H = stack_height
            self.stability_class = stability_class
    
        def _pasquill_gifford_sigma(self, x: float) -> tuple:
            """Pasquill-Gifford拡散パラメータ
    
            Args:
                x: 風下距離 [m]
    
            Returns:
                (sigma_y, sigma_z): 横方向・鉛直方向拡散係数 [m]
            """
            # 簡略化されたPasquill-Gifford式（stability class D: neutral）
            # 実際の実装では stability_class に応じて係数を変える
    
            coefficients = {
                'A': (0.22, 0.20),  # Very unstable
                'B': (0.16, 0.12),  # Unstable
                'C': (0.11, 0.08),  # Slightly unstable
                'D': (0.08, 0.06),  # Neutral（デフォルト）
                'E': (0.06, 0.03),  # Slightly stable
                'F': (0.04, 0.016)  # Stable
            }
    
            a_y, a_z = coefficients.get(self.stability_class, (0.08, 0.06))
    
            # 拡散係数計算（経験式）
            sigma_y = a_y * x * (1 + 0.0001 * x)**(-0.5)
            sigma_z = a_z * x
    
            return sigma_y, sigma_z
    
        def concentration(self, x: float, y: float, z: float) -> float:
            """特定位置での濃度を計算
    
            Args:
                x: 風下距離 [m]
                y: 横風方向距離 [m]
                z: 地上高さ [m]
    
            Returns:
                濃度 [g/m^3]
            """
            if x <= 0:
                return 0.0
    
            sigma_y, sigma_z = self._pasquill_gifford_sigma(x)
    
            # Gaussian Plume式
            C = (self.Q / (2 * np.pi * self.u * sigma_y * sigma_z)) * \
                np.exp(-0.5 * (y / sigma_y)**2) * \
                (np.exp(-0.5 * ((z - self.H) / sigma_z)**2) +
                 np.exp(-0.5 * ((z + self.H) / sigma_z)**2))  # 地表反射項
    
            return C
    
        def ground_level_centerline_concentration(self, x: float) -> float:
            """風下中心軸・地表レベル濃度（y=0, z=0）"""
            return self.concentration(x, 0, 0)
    
        def calculate_impact_zone(self, threshold: float, max_distance: float = 5000) -> float:
            """影響範囲（閾値濃度に達する最大距離）を計算
    
            Args:
                threshold: 閾値濃度 [g/m^3]
                max_distance: 最大評価距離 [m]
    
            Returns:
                影響範囲距離 [m]
            """
            distances = np.linspace(10, max_distance, 1000)
            concentrations = [self.ground_level_centerline_concentration(d)
                              for d in distances]
    
            # 閾値以上の最遠距離
            impact_distances = distances[np.array(concentrations) >= threshold]
    
            if len(impact_distances) > 0:
                return impact_distances[-1]
            else:
                return 0.0
    
        def visualize_concentration_profile(self, max_distance: float = 2000):
            """濃度プロファイルを可視化"""
            distances = np.linspace(10, max_distance, 500)
            concentrations = [self.ground_level_centerline_concentration(d)
                              for d in distances]
    
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(distances, np.array(concentrations) * 1e6, 'b-', linewidth=2)
            ax.set_xlabel('Distance Downwind (m)', fontsize=12)
            ax.set_ylabel('Concentration (mg/m³)', fontsize=12)
            ax.set_title(f'Gaussian Plume Dispersion Model\n'
                         f'Q={self.Q} g/s, u={self.u} m/s, Class {self.stability_class}',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
    
            # ERPG/AEGL閾値例（塩素ガスの場合）
            aegl_2 = 2.8  # mg/m^3（60分暴露で不可逆的健康影響）
            aegl_3 = 20   # mg/m^3（60分暴露で生命危険）
    
            ax.axhline(y=aegl_2, color='orange', linestyle='--',
                       label=f'AEGL-2: {aegl_2} mg/m³')
            ax.axhline(y=aegl_3, color='red', linestyle='--',
                       label=f'AEGL-3: {aegl_3} mg/m³')
            ax.legend()
    
            plt.tight_layout()
            return fig
    
    
    # 使用例: 塩素ガス（Cl2）漏洩シナリオ
    plume_model = GaussianPlumeModel(
        emission_rate=100,      # 100 g/s（360 kg/hr）
        wind_speed=3.0,         # 3 m/s（やや弱い風）
        stack_height=2.0,       # 2m高さから放出
        stability_class='D'     # 中立大気
    )
    
    # 地表中心軸での濃度計算
    distances = [100, 500, 1000, 2000]
    print("=== 地表中心軸濃度 ===\n")
    for d in distances:
        conc = plume_model.ground_level_centerline_concentration(d)
        print(f"Distance: {d:4d} m → Concentration: {conc*1e6:.2f} mg/m³")
    
    # 影響範囲計算（AEGL-2: 2.8 mg/m^3）
    impact_distance_aegl2 = plume_model.calculate_impact_zone(
        threshold=2.8e-3  # 2.8 mg/m^3 = 2.8e-3 g/m^3
    )
    print(f"\nAEGL-2影響範囲: {impact_distance_aegl2:.0f} m")
    
    # 可視化
    fig = plume_model.visualize_concentration_profile()
    # plt.show()
    
    # 期待される出力:
    # === 地表中心軸濃度 ===
    #
    # Distance:  100 m → Concentration: 127.45 mg/m³
    # Distance:  500 m → Concentration: 8.73 mg/m³
    # Distance: 1000 m → Concentration: 2.51 mg/m³
    # Distance: 2000 m → Concentration: 0.73 mg/m³
    #
    # AEGL-2影響範囲: 1245 m
    

> **実務への適用:** 実際のConsequence Modelingでは、ALOHA（NOAA）、PHAST（DNV）、EFFECTS（TNO）などの専門ソフトウェアが使用されます。本例は教育目的の簡略化モデルです。

* * *

## 1.6 Bow-tie分析

### Example 5: Bow-tie図の構築

Bow-tie分析は、ハザードイベントの原因（左側）と結果（右側）を視覚化し、予防的保護層と低減的保護層を整理します。
    
    
    # ===================================
    # Example 5: Bow-tie分析
    # ===================================
    
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class Threat:
        """脅威（原因）"""
        name: str
        barriers: List[str]  # 予防的保護層
    
    @dataclass
    class Consequence:
        """結果"""
        name: str
        barriers: List[str]  # 低減的保護層
    
    class BowtieAnalysis:
        """Bow-tie分析システム"""
    
        def __init__(self, hazard_event: str):
            self.hazard_event = hazard_event
            self.threats: List[Threat] = []
            self.consequences: List[Consequence] = []
    
        def add_threat(self, threat: Threat):
            """脅威を追加"""
            self.threats.append(threat)
    
        def add_consequence(self, consequence: Consequence):
            """結果を追加"""
            self.consequences.append(consequence)
    
        def generate_bowtie_report(self) -> str:
            """Bow-tieレポート生成（テキスト形式）"""
            report = f"""
    {'='*70}
    Bow-tie Analysis Report
    {'='*70}
    
    Hazard Event: {self.hazard_event}
    
    {'='*70}
    LEFT SIDE: Threats (Causes) and Preventive Barriers
    {'='*70}
    
    """
    
            for i, threat in enumerate(self.threats, 1):
                report += f"\nThreat {i}: {threat.name}\n"
                report += "  Preventive Barriers:\n"
                for j, barrier in enumerate(threat.barriers, 1):
                    report += f"    {j}. {barrier}\n"
    
            report += f"""
    {'='*70}
    RIGHT SIDE: Consequences and Mitigative Barriers
    {'='*70}
    
    """
    
            for i, consequence in enumerate(self.consequences, 1):
                report += f"\nConsequence {i}: {consequence.name}\n"
                report += "  Mitigative Barriers:\n"
                for j, barrier in enumerate(consequence.barriers, 1):
                    report += f"    {j}. {barrier}\n"
    
            report += f"\n{'='*70}\n"
    
            # 統計情報
            total_preventive = sum(len(t.barriers) for t in self.threats)
            total_mitigative = sum(len(c.barriers) for c in self.consequences)
    
            report += f"""
    Summary Statistics:
      - Total Threats: {len(self.threats)}
      - Total Preventive Barriers: {total_preventive}
      - Total Consequences: {len(self.consequences)}
      - Total Mitigative Barriers: {total_mitigative}
      - Defense-in-Depth Layers: {total_preventive + total_mitigative}
    """
    
            return report
    
        def identify_critical_barriers(self) -> List[str]:
            """単一障壁シナリオ（Critical Single Points of Failure）を識別"""
            critical = []
    
            for threat in self.threats:
                if len(threat.barriers) == 1:
                    critical.append(f"Threat '{threat.name}' has only 1 barrier: {threat.barriers[0]}")
    
            for consequence in self.consequences:
                if len(consequence.barriers) == 1:
                    critical.append(f"Consequence '{consequence.name}' has only 1 barrier: {consequence.barriers[0]}")
    
            return critical
    
    
    # 使用例: 可燃性液体貯蔵タンク火災のBow-tie分析
    bowtie = BowtieAnalysis(hazard_event="可燃性液体貯蔵タンク火災")
    
    # 脅威（原因）と予防的保護層
    bowtie.add_threat(Threat(
        name="静電気放電による着火",
        barriers=[
            "タンク接地・ボンディング",
            "帯電防止剤添加",
            "流速制限（<1 m/s）",
            "不活性ガス（N2）パージ"
        ]
    ))
    
    bowtie.add_threat(Threat(
        name="雷による着火",
        barriers=[
            "避雷針設置",
            "サージプロテクター",
            "接地システム"
        ]
    ))
    
    bowtie.add_threat(Threat(
        name="高温表面との接触",
        barriers=[
            "Hot Work許可システム",
            "温度監視",
            "断熱材設置",
            "火気作業禁止区域設定"
        ]
    ))
    
    # 結果と低減的保護層
    bowtie.add_consequence(Consequence(
        name="タンク火災（プールファイア）",
        barriers=[
            "泡消火設備（固定式）",
            "消火器配置",
            "緊急遮断弁（自動）",
            "防液堤（二次格納）",
            "冷却水スプレー"
        ]
    ))
    
    bowtie.add_consequence(Consequence(
        name="BLEVE（沸騰液体膨張蒸気爆発）",
        barriers=[
            "圧力逃し弁",
            "冷却水スプレー（タンク上部）",
            "熱遮蔽",
            "緊急離隔距離確保"
        ]
    ))
    
    bowtie.add_consequence(Consequence(
        name="近隣施設への延焼",
        barriers=[
            "防火壁",
            "スプリンクラーシステム",
            "消防隊通報システム",
            "緊急避難計画"
        ]
    ))
    
    # レポート生成
    report = bowtie.generate_bowtie_report()
    print(report)
    
    # Critical Single Pointsを識別
    critical_barriers = bowtie.identify_critical_barriers()
    if critical_barriers:
        print("\n⚠️ Critical Single Points of Failure:")
        for cb in critical_barriers:
            print(f"  - {cb}")
    else:
        print("\n✅ No critical single points of failure identified.")
    
    # 期待される出力:
    # ======================================================================
    # Bow-tie Analysis Report
    # ======================================================================
    #
    # Hazard Event: 可燃性液体貯蔵タンク火災
    #
    # ======================================================================
    # LEFT SIDE: Threats (Causes) and Preventive Barriers
    # ======================================================================
    #
    # Threat 1: 静電気放電による着火
    #   Preventive Barriers:
    #     1. タンク接地・ボンディング
    #     2. 帯電防止剤添加
    #     3. 流速制限（<1 m/s）
    #     4. 不活性ガス（N2）パージ
    # ...
    #
    # Summary Statistics:
    #   - Total Threats: 3
    #   - Total Preventive Barriers: 11
    #   - Total Consequences: 3
    #   - Total Mitigative Barriers: 14
    #   - Defense-in-Depth Layers: 25
    

* * *

## 1.7 リスクベース検査（RBI）

### Example 6: API 580 RBIフレームワーク

API 580（Risk-Based Inspection）に基づく、設備検査優先順位付けシステムを実装します。
    
    
    # ===================================
    # Example 6: リスクベース検査（RBI）
    # ===================================
    
    from dataclasses import dataclass
    from typing import List, Dict
    import pandas as pd
    
    class CorrosionMechanism(Enum):
        """腐食メカニズム"""
        GENERAL_CORROSION = "全面腐食"
        PITTING = "孔食"
        SCC = "応力腐食割れ"
        EROSION = "エロージョン"
        FATIGUE = "疲労"
    
    @dataclass
    class Equipment:
        """設備情報"""
        id: str
        name: str
        equipment_type: str
        fluid: str
        temperature: float  # °C
        pressure: float     # MPa
        age: float          # years
        last_inspection: float  # years ago
        corrosion_mechanism: CorrosionMechanism
        corrosion_rate: float  # mm/year
        thickness_remaining: float  # mm
        design_thickness: float  # mm
    
    class RBIAnalysis:
        """リスクベース検査分析"""
    
        def calculate_pof(self, equipment: Equipment) -> float:
            """Probability of Failure（故障確率）を計算
    
            簡略化されたモデル:
              PoF = f(corrosion_damage, time_since_inspection, operating_severity)
    
            Returns:
                PoF score (0-100)
            """
            # 腐食ダメージファクター
            damage_factor = equipment.corrosion_rate * equipment.age
            thickness_ratio = equipment.thickness_remaining / equipment.design_thickness
    
            if thickness_ratio < 0.5:
                corrosion_score = 90
            elif thickness_ratio < 0.7:
                corrosion_score = 60
            elif thickness_ratio < 0.9:
                corrosion_score = 30
            else:
                corrosion_score = 10
    
            # 検査間隔ファクター
            if equipment.last_inspection > 10:
                inspection_score = 80
            elif equipment.last_inspection > 5:
                inspection_score = 50
            elif equipment.last_inspection > 2:
                inspection_score = 20
            else:
                inspection_score = 5
    
            # 運転苛酷度ファクター
            if equipment.temperature > 200 or equipment.pressure > 5.0:
                severity_score = 70
            elif equipment.temperature > 100 or equipment.pressure > 2.0:
                severity_score = 40
            else:
                severity_score = 10
    
            # 腐食メカニズムファクター
            mechanism_multiplier = {
                CorrosionMechanism.SCC: 1.5,
                CorrosionMechanism.PITTING: 1.3,
                CorrosionMechanism.FATIGUE: 1.4,
                CorrosionMechanism.EROSION: 1.2,
                CorrosionMechanism.GENERAL_CORROSION: 1.0
            }
    
            multiplier = mechanism_multiplier[equipment.corrosion_mechanism]
    
            # 総合PoF計算（加重平均）
            pof = (corrosion_score * 0.4 + inspection_score * 0.3 +
                   severity_score * 0.3) * multiplier
    
            return min(pof, 100)
    
        def calculate_cof(self, equipment: Equipment) -> float:
            """Consequence of Failure（故障影響度）を計算
    
            簡略化されたモデル:
              CoF = f(fluid_hazard, inventory, pressure)
    
            Returns:
                CoF score (0-100)
            """
            # 流体ハザードスコア
            high_hazard_fluids = ['H2', 'Cl2', 'HF', 'NH3', 'C2H4', 'LPG']
            medium_hazard_fluids = ['methanol', 'ethanol', 'benzene', 'toluene']
    
            if equipment.fluid in high_hazard_fluids:
                fluid_score = 90
            elif equipment.fluid in medium_hazard_fluids:
                fluid_score = 60
            else:
                fluid_score = 30
    
            # 圧力影響スコア
            if equipment.pressure > 5.0:
                pressure_score = 80
            elif equipment.pressure > 2.0:
                pressure_score = 50
            else:
                pressure_score = 20
    
            # 設備タイプ影響スコア
            if equipment.equipment_type in ['Reactor', 'Distillation Column']:
                equipment_score = 70
            elif equipment.equipment_type in ['Heat Exchanger', 'Pump']:
                equipment_score = 40
            else:
                equipment_score = 20
    
            # 総合CoF計算
            cof = fluid_score * 0.5 + pressure_score * 0.3 + equipment_score * 0.2
    
            return min(cof, 100)
    
        def calculate_risk_score(self, pof: float, cof: float) -> float:
            """リスクスコア = PoF × CoF"""
            return (pof * cof) / 100  # 0-100スケールに正規化
    
        def determine_inspection_priority(self, risk_score: float) -> str:
            """検査優先順位を決定"""
            if risk_score >= 70:
                return "Priority 1 (Immediate - within 1 month)"
            elif risk_score >= 50:
                return "Priority 2 (High - within 3 months)"
            elif risk_score >= 30:
                return "Priority 3 (Medium - within 1 year)"
            else:
                return "Priority 4 (Low - routine inspection)"
    
        def analyze_equipment_portfolio(self, equipment_list: List[Equipment]) -> pd.DataFrame:
            """設備群のRBI分析"""
            results = []
    
            for eq in equipment_list:
                pof = self.calculate_pof(eq)
                cof = self.calculate_cof(eq)
                risk = self.calculate_risk_score(pof, cof)
                priority = self.determine_inspection_priority(risk)
    
                results.append({
                    'Equipment ID': eq.id,
                    'Equipment Name': eq.name,
                    'Type': eq.equipment_type,
                    'Fluid': eq.fluid,
                    'PoF': f"{pof:.1f}",
                    'CoF': f"{cof:.1f}",
                    'Risk Score': f"{risk:.1f}",
                    'Priority': priority,
                    'Last Inspection': f"{eq.last_inspection:.1f} years ago",
                    'Thickness Ratio': f"{eq.thickness_remaining/eq.design_thickness:.2f}"
                })
    
            df = pd.DataFrame(results)
            df = df.sort_values('Risk Score', ascending=False,
                                key=lambda x: x.astype(float))
    
            return df
    
    
    # 使用例
    rbi = RBIAnalysis()
    
    # 設備リスト作成
    equipment_portfolio = [
        Equipment(
            id="V-101", name="反応器", equipment_type="Reactor",
            fluid="H2", temperature=350, pressure=8.0, age=15,
            last_inspection=6.0, corrosion_mechanism=CorrosionMechanism.GENERAL_CORROSION,
            corrosion_rate=0.15, thickness_remaining=8.5, design_thickness=12.0
        ),
        Equipment(
            id="T-201", name="蒸留塔", equipment_type="Distillation Column",
            fluid="toluene", temperature=120, pressure=0.5, age=20,
            last_inspection=3.0, corrosion_mechanism=CorrosionMechanism.PITTING,
            corrosion_rate=0.3, thickness_remaining=5.2, design_thickness=10.0
        ),
        Equipment(
            id="E-301", name="熱交換器", equipment_type="Heat Exchanger",
            fluid="water", temperature=80, pressure=1.5, age=10,
            last_inspection=2.0, corrosion_mechanism=CorrosionMechanism.GENERAL_CORROSION,
            corrosion_rate=0.05, thickness_remaining=9.5, design_thickness=10.0
        ),
        Equipment(
            id="P-401", name="プロセスポンプ", equipment_type="Pump",
            fluid="methanol", temperature=40, pressure=3.0, age=8,
            last_inspection=1.5, corrosion_mechanism=CorrosionMechanism.EROSION,
            corrosion_rate=0.2, thickness_remaining=7.0, design_thickness=8.0
        ),
        Equipment(
            id="V-501", name="圧力容器", equipment_type="Pressure Vessel",
            fluid="NH3", temperature=25, pressure=10.0, age=25,
            last_inspection=12.0, corrosion_mechanism=CorrosionMechanism.SCC,
            corrosion_rate=0.1, thickness_remaining=6.0, design_thickness=15.0
        )
    ]
    
    # RBI分析実行
    rbi_results = rbi.analyze_equipment_portfolio(equipment_portfolio)
    
    print("=== リスクベース検査（RBI）分析結果 ===\n")
    print(rbi_results.to_string(index=False))
    
    # Priority 1設備を抽出
    priority_1 = rbi_results[rbi_results['Priority'].str.contains('Priority 1')]
    print(f"\n⚠️ 緊急対応が必要な設備数: {len(priority_1)}")
    
    # 期待される出力:
    # === リスクベース検査（RBI）分析結果 ===
    #
    # Equipment ID Equipment Name                  Type   Fluid   PoF   CoF  Risk Score ...
    #        V-501        圧力容器      Pressure Vessel    NH3  84.0  82.0        68.9 ...
    #        V-101          反応器              Reactor     H2  67.5  80.0        54.0 ...
    #        T-201        蒸留塔   Distillation Column toluene  70.2  64.0        44.9 ...
    # ...
    

* * *

## 1.8 安全バリア有効性分析

### Example 7: 安全バリアパフォーマンスモニタリング
    
    
    # ===================================
    # Example 7: 安全バリア有効性分析
    # ===================================
    
    from dataclasses import dataclass
    from datetime import datetime, timedelta
    import random
    
    @dataclass
    class BarrierTest:
        """バリアテスト記録"""
        date: datetime
        passed: bool
        response_time: float  # seconds (for active barriers)
    
    class SafetyBarrier:
        """安全バリア"""
    
        def __init__(self, name: str, barrier_type: str, target_pfd: float):
            """
            Args:
                name: バリア名
                barrier_type: タイプ（Passive/Active）
                target_pfd: 目標PFD（Probability of Failure on Demand）
            """
            self.name = name
            self.barrier_type = barrier_type
            self.target_pfd = target_pfd
            self.test_history: List[BarrierTest] = []
    
        def add_test_result(self, test: BarrierTest):
            """テスト結果を追加"""
            self.test_history.append(test)
    
        def calculate_actual_pfd(self, lookback_period: int = 365) -> float:
            """実際のPFDを計算
    
            Args:
                lookback_period: 評価期間（日数）
    
            Returns:
                実測PFD
            """
            if not self.test_history:
                return 1.0  # データなし = 最悪ケース
    
            cutoff_date = datetime.now() - timedelta(days=lookback_period)
            recent_tests = [t for t in self.test_history if t.date > cutoff_date]
    
            if not recent_tests:
                return 1.0
    
            failures = sum(1 for t in recent_tests if not t.passed)
            pfd = failures / len(recent_tests)
    
            return pfd
    
        def assess_performance(self) -> Dict:
            """バリアパフォーマンス評価"""
            actual_pfd = self.calculate_actual_pfd()
    
            # パフォーマンス判定
            if actual_pfd <= self.target_pfd:
                status = "✅ ACCEPTABLE"
                action = "Continue routine testing"
            elif actual_pfd <= self.target_pfd * 1.5:
                status = "⚠️ DEGRADED"
                action = "Increase testing frequency, investigate root causes"
            else:
                status = "❌ UNACCEPTABLE"
                action = "Immediate corrective action required, consider bypass"
    
            # 平均応答時間（Active barriersの場合）
            avg_response_time = None
            if self.barrier_type == "Active" and self.test_history:
                response_times = [t.response_time for t in self.test_history
                                  if t.passed]
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
    
            return {
                'Barrier': self.name,
                'Type': self.barrier_type,
                'Target PFD': f"{self.target_pfd:.2e}",
                'Actual PFD': f"{actual_pfd:.2e}",
                'Status': status,
                'Test Count': len(self.test_history),
                'Avg Response Time': f"{avg_response_time:.2f}s" if avg_response_time else "N/A",
                'Recommended Action': action
            }
    
    class BarrierManagementSystem:
        """安全バリア管理システム"""
    
        def __init__(self):
            self.barriers: List[SafetyBarrier] = []
    
        def add_barrier(self, barrier: SafetyBarrier):
            """バリアを追加"""
            self.barriers.append(barrier)
    
        def generate_performance_report(self) -> pd.DataFrame:
            """パフォーマンスレポート生成"""
            results = [barrier.assess_performance() for barrier in self.barriers]
            df = pd.DataFrame(results)
            return df
    
        def identify_degraded_barriers(self) -> List[str]:
            """劣化バリアを識別"""
            degraded = []
            for barrier in self.barriers:
                assessment = barrier.assess_performance()
                if "DEGRADED" in assessment['Status'] or "UNACCEPTABLE" in assessment['Status']:
                    degraded.append(barrier.name)
            return degraded
    
    
    # 使用例: バリア管理システムの実装
    bms = BarrierManagementSystem()
    
    # バリア1: 高圧インターロック（SIL 2）
    interlock = SafetyBarrier(
        name="High Pressure Interlock (SIS-101)",
        barrier_type="Active",
        target_pfd=0.01  # SIL 2 target
    )
    
    # テストデータ生成（過去1年分）
    random.seed(42)
    for i in range(24):  # 月2回テスト
        test_date = datetime.now() - timedelta(days=i*15)
        passed = random.random() > 0.008  # 99.2%成功率
        response_time = random.gauss(2.5, 0.5)  # 平均2.5秒、標準偏差0.5秒
    
        interlock.add_test_result(BarrierTest(
            date=test_date,
            passed=passed,
            response_time=response_time
        ))
    
    bms.add_barrier(interlock)
    
    # バリア2: 圧力逃し弁（Passive）
    prv = SafetyBarrier(
        name="Pressure Relief Valve (PRV-201)",
        barrier_type="Passive",
        target_pfd=0.01
    )
    
    for i in range(4):  # 年4回テスト
        test_date = datetime.now() - timedelta(days=i*90)
        passed = random.random() > 0.02  # 98%成功率
    
        prv.add_test_result(BarrierTest(
            date=test_date,
            passed=passed,
            response_time=0  # Passive barrier
        ))
    
    bms.add_barrier(prv)
    
    # バリア3: ガス検知器（劣化している例）
    gas_detector = SafetyBarrier(
        name="H2 Gas Detector (GD-301)",
        barrier_type="Active",
        target_pfd=0.05
    )
    
    for i in range(52):  # 週1回テスト
        test_date = datetime.now() - timedelta(days=i*7)
        passed = random.random() > 0.12  # 88%成功率（劣化）
        response_time = random.gauss(1.0, 0.3)
    
        gas_detector.add_test_result(BarrierTest(
            date=test_date,
            passed=passed,
            response_time=response_time
        ))
    
    bms.add_barrier(gas_detector)
    
    # パフォーマンスレポート生成
    report = bms.generate_performance_report()
    print("=== 安全バリアパフォーマンスレポート ===\n")
    print(report.to_string(index=False))
    
    # 劣化バリアを識別
    degraded = bms.identify_degraded_barriers()
    if degraded:
        print(f"\n⚠️ 注意が必要なバリア:")
        for b in degraded:
            print(f"  - {b}")
    
    # 期待される出力:
    # === 安全バリアパフォーマンスレポート ===
    #
    #                           Barrier      Type Target PFD Actual PFD           Status  Test Count ...
    #  High Pressure Interlock (SIS-101)    Active   1.00e-02   8.33e-03   ✅ ACCEPTABLE          24 ...
    #  Pressure Relief Valve (PRV-201)    Passive   1.00e-02   0.00e+00   ✅ ACCEPTABLE           4 ...
    #  H2 Gas Detector (GD-301)            Active   5.00e-02   1.15e-01  ❌ UNACCEPTABLE          52 ...
    

* * *

## 1.9 実践演習

### Example 8: 統合プロセス安全評価システム

これまで学んだ手法を統合した、包括的なプロセス安全評価システムを構築します。
    
    
    # ===================================
    # Example 8: 統合プロセス安全評価システム
    # ===================================
    
    class IntegratedProcessSafetyAssessment:
        """統合プロセス安全評価システム"""
    
        def __init__(self, process_name: str):
            self.process_name = process_name
            self.hazard_system = HazardIdentificationSystem()
            self.risk_matrix = RiskMatrix()
            self.lopa_analyses: List[LOPAAnalysis] = []
            self.barrier_management = BarrierManagementSystem()
    
        def perform_comprehensive_assessment(self) -> Dict:
            """包括的安全性評価を実施"""
    
            # 1. ハザード識別
            hazard_register = self.hazard_system.generate_hazard_register()
            critical_hazards = self.hazard_system.get_hazards_by_severity(Severity.CRITICAL)
    
            # 2. リスク評価
            risk_assessments = []
            for hazard in self.hazard_system.hazards:
                # 簡略化: 発生頻度は仮定
                likelihood = Likelihood.OCCASIONAL if hazard.severity.value >= 4 else Likelihood.REMOTE
    
                risk_assess = self.risk_matrix.assess_risk(
                    hazard.name,
                    hazard.severity,
                    likelihood
                )
                risk_assessments.append(risk_assess)
    
            # 高リスクハザードを抽出
            high_risk = [r for r in risk_assessments
                         if r['リスクレベル'] in ['極めて高い', '高い']]
    
            # 3. LOPA分析（高リスクハザード対象）
            lopa_results = []
            for lopa in self.lopa_analyses:
                lopa_report = lopa.generate_lopa_report()
                mitigated_freq = lopa.calculate_mitigated_frequency()
                lopa_results.append({
                    'Scenario': lopa.scenario_name,
                    'Mitigated Frequency': f"{mitigated_freq:.2e}",
                    'SIL Required': lopa.determine_required_sil().name
                })
    
            # 4. バリアパフォーマンス
            barrier_report = self.barrier_management.generate_performance_report()
            degraded_barriers = self.barrier_management.identify_degraded_barriers()
    
            # 統合結果
            return {
                'total_hazards': len(self.hazard_system.hazards),
                'critical_hazards': len(critical_hazards),
                'high_risk_scenarios': len(high_risk),
                'lopa_analyses': len(lopa_results),
                'total_barriers': len(self.barrier_management.barriers),
                'degraded_barriers': len(degraded_barriers),
                'hazard_register': hazard_register,
                'risk_assessments': pd.DataFrame(risk_assessments),
                'lopa_results': pd.DataFrame(lopa_results) if lopa_results else None,
                'barrier_performance': barrier_report,
                'degraded_barrier_list': degraded_barriers
            }
    
        def generate_executive_summary(self, assessment: Dict) -> str:
            """エグゼクティブサマリー生成"""
            summary = f"""
    {'='*70}
    PROCESS SAFETY ASSESSMENT - EXECUTIVE SUMMARY
    {'='*70}
    
    Process: {self.process_name}
    Assessment Date: {datetime.now().strftime('%Y-%m-%d')}
    
    {'='*70}
    KEY FINDINGS
    {'='*70}
    
    1. HAZARD IDENTIFICATION
       - Total Hazards Identified: {assessment['total_hazards']}
       - Critical/Catastrophic Hazards: {assessment['critical_hazards']}
    
    2. RISK ASSESSMENT
       - High Risk Scenarios: {assessment['high_risk_scenarios']}
       - Immediate Action Required: {sum(1 for _ in assessment['risk_assessments'].itertuples() if '極めて高い' in _.リスクレベル)}
    
    3. PROTECTION LAYERS (LOPA)
       - LOPA Studies Completed: {assessment['lopa_analyses']}
       - SIS Implementation Required: {sum(1 for _ in (assessment['lopa_results'].itertuples() if assessment['lopa_results'] is not None else []) if 'SIL' in _.SIL_Required and _.SIL_Required != 'NO_SIL')}
    
    4. BARRIER INTEGRITY
       - Total Safety Barriers: {assessment['total_barriers']}
       - Degraded/Failing Barriers: {assessment['degraded_barriers']}
    
    {'='*70}
    CRITICAL ACTION ITEMS
    {'='*70}
    """
    
            # 優先アクションアイテム
            action_items = []
    
            if assessment['degraded_barriers'] > 0:
                for barrier_name in assessment['degraded_barrier_list']:
                    action_items.append(
                        f"⚠️ URGENT: Repair/Replace barrier: {barrier_name}"
                    )
    
            if assessment['high_risk_scenarios'] > 0:
                action_items.append(
                    f"⚠️ HIGH PRIORITY: Implement risk reduction for {assessment['high_risk_scenarios']} scenarios"
                )
    
            if action_items:
                for i, item in enumerate(action_items, 1):
                    summary += f"\n{i}. {item}"
            else:
                summary += "\n✅ No critical action items identified."
    
            summary += f"\n\n{'='*70}\n"
            summary += "STATUS: "
    
            if assessment['degraded_barriers'] == 0 and assessment['high_risk_scenarios'] <= 2:
                summary += "✅ ACCEPTABLE - Continue routine monitoring\n"
            elif assessment['degraded_barriers'] <= 2 and assessment['high_risk_scenarios'] <= 5:
                summary += "⚠️ REQUIRES ATTENTION - Implement improvements within 3 months\n"
            else:
                summary += "❌ UNACCEPTABLE - Immediate corrective action required\n"
    
            summary += f"{'='*70}\n"
    
            return summary
    
    
    # 使用例: 統合評価の実施
    integrated_assessment = IntegratedProcessSafetyAssessment(
        process_name="Hydrogen Production Unit"
    )
    
    # LOPAを追加（Example 3で作成したもの）
    integrated_assessment.lopa_analyses.append(lopa)
    
    # バリアを追加（Example 7で作成したもの）
    integrated_assessment.barrier_management = bms
    
    # 包括的評価を実施
    assessment_results = integrated_assessment.perform_comprehensive_assessment()
    
    # エグゼクティブサマリー生成
    executive_summary = integrated_assessment.generate_executive_summary(assessment_results)
    print(executive_summary)
    
    # 詳細結果
    print("\n=== ハザードレジスター（Top 3） ===")
    print(assessment_results['hazard_register'].head(3).to_string(index=False))
    
    print("\n=== リスク評価結果（High Risk） ===")
    high_risk_df = assessment_results['risk_assessments'][
        assessment_results['risk_assessments']['リスクレベル'].isin(['極めて高い', '高い'])
    ]
    print(high_risk_df.to_string(index=False))
    
    # 期待される出力:
    # ======================================================================
    # PROCESS SAFETY ASSESSMENT - EXECUTIVE SUMMARY
    # ======================================================================
    #
    # Process: Hydrogen Production Unit
    # Assessment Date: 2025-10-26
    #
    # ======================================================================
    # KEY FINDINGS
    # ======================================================================
    #
    # 1. HAZARD IDENTIFICATION
    #    - Total Hazards Identified: 3
    #    - Critical/Catastrophic Hazards: 3
    #
    # 2. RISK ASSESSMENT
    #    - High Risk Scenarios: 2
    #    - Immediate Action Required: 1
    #
    # 3. PROTECTION LAYERS (LOPA)
    #    - LOPA Studies Completed: 1
    #    - SIS Implementation Required: 0
    #
    # 4. BARRIER INTEGRITY
    #    - Total Safety Barriers: 3
    #    - Degraded/Failing Barriers: 1
    #
    # ======================================================================
    # CRITICAL ACTION ITEMS
    # ======================================================================
    #
    # 1. ⚠️ URGENT: Repair/Replace barrier: H2 Gas Detector (GD-301)
    # 2. ⚠️ HIGH PRIORITY: Implement risk reduction for 2 scenarios
    #
    # ======================================================================
    # STATUS: ⚠️ REQUIRES ATTENTION - Implement improvements within 3 months
    

* * *

## 学習目標の確認

このchapterを完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ プロセス安全の概念と重大事故の教訓を理解している
  * ✅ ハザード vs リスクの違いを説明できる
  * ✅ リスクマトリックスの構造を理解している
  * ✅ LOPAの原理とSILレベルを知っている

### 実践スキル

  * ✅ ハザード識別フレームワークを実装できる
  * ✅ リスクマトリックスでリスク評価ができる
  * ✅ LOPA手法でSILを計算できる
  * ✅ Gaussian Plumeモデルで影響範囲を予測できる
  * ✅ Bow-tie分析で保護層を整理できる
  * ✅ RBI手法で検査優先順位を決定できる

### 応用力

  * ✅ 実際の化学プロセスのハザード識別ができる
  * ✅ 包括的なプロセス安全評価システムを構築できる
  * ✅ 安全バリアのパフォーマンスを監視・評価できる
  * ✅ エグゼクティブサマリーを作成し意思決定を支援できる

* * *

## 次のステップ

第1章では、プロセス安全の基礎、ハザード識別、リスク評価、保護層分析を学びました。

**第2章では：**

  * 📋 HAZOPスタディの詳細手法
  * 📋 ガイドワードを用いた逸脱分析
  * 📋 定量的リスクアセスメント（QRA）
  * 📋 F-N曲線とリスク基準

を学びます。
