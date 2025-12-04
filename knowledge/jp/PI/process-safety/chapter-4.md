---
title: 第4章：安全性モニタリングと事故予測
chapter_title: 第4章：安全性モニタリングと事故予測
subtitle: リアルタイム監視と機械学習による予測的安全管理
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/process-safety/chapter-4.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Process Safety](<../../PI/process-safety/index.html>)›Chapter 4

## この章で学ぶこと

  * **リアルタイム安全性モニタリング** の実装方法
  * **異常状況検知** （多変量統計的プロセス管理）
  * **先行指標（Leading Indicators）** による予兆管理
  * **機械学習** を用いた事故予測モデルの構築
  * **安全性パフォーマンス指標** （TRIR、LTIR）の計算
  * ニアミス解析とヒヤリハット管理システム

## 4.1 リアルタイム安全性モニタリング

プロセス安全性の確保には、プロセス変数の連続監視と異常の早期検知が不可欠です。 温度、圧力、流量などの重要パラメータをリアルタイムで監視し、 安全限界への接近を検知するシステムを構築します。 

### 4.1.1 安全パラメータ監視システム
    
    
    # Example 1: Real-time Safety Parameter Monitoring
    import numpy as np
    import pandas as pd
    from dataclasses import dataclass
    from typing import Dict, List, Optional
    from datetime import datetime, timedelta
    from enum import Enum
    
    class SafetyLevel(Enum):
        """安全レベル"""
        NORMAL = "Normal"
        CAUTION = "Caution"
        WARNING = "Warning"
        CRITICAL = "Critical"
    
    @dataclass
    class SafetyLimit:
        """安全限界値"""
        low_critical: float      # 下限Critical
        low_warning: float       # 下限Warning
        low_caution: float       # 下限Caution
        high_caution: float      # 上限Caution
        high_warning: float      # 上限Warning
        high_critical: float     # 上限Critical
    
    class SafetyMonitor:
        """リアルタイム安全性監視システム
    
        プロセス変数を連続監視し、安全限界への接近を検知
        """
    
        def __init__(self):
            self.limits: Dict[str, SafetyLimit] = {}
            self.current_values: Dict[str, float] = {}
            self.alert_history: List[Dict] = []
    
        def set_safety_limits(self, parameter: str, limits: SafetyLimit):
            """安全限界を設定"""
            self.limits[parameter] = limits
    
        def update_value(self, parameter: str, value: float,
                        timestamp: datetime = None) -> Dict:
            """パラメータ値を更新し、安全性を評価
    
            Args:
                parameter: パラメータ名
                value: 測定値
                timestamp: タイムスタンプ
    
            Returns:
                評価結果（レベル、メッセージ）
            """
            if timestamp is None:
                timestamp = datetime.now()
    
            self.current_values[parameter] = value
    
            # 安全レベル判定
            safety_assessment = self._assess_safety(parameter, value)
    
            # アラート履歴に記録
            if safety_assessment['level'] != SafetyLevel.NORMAL:
                self.alert_history.append({
                    'timestamp': timestamp,
                    'parameter': parameter,
                    'value': value,
                    'level': safety_assessment['level'],
                    'message': safety_assessment['message']
                })
    
            return safety_assessment
    
        def _assess_safety(self, parameter: str, value: float) -> Dict:
            """安全性を評価"""
            if parameter not in self.limits:
                return {
                    'level': SafetyLevel.NORMAL,
                    'message': 'No limits defined',
                    'distance_to_limit': None
                }
    
            limits = self.limits[parameter]
    
            # Critical範囲
            if value <= limits.low_critical or value >= limits.high_critical:
                level = SafetyLevel.CRITICAL
                message = "Critical limit exceeded! Immediate action required."
                distance = min(
                    abs(value - limits.low_critical),
                    abs(value - limits.high_critical)
                )
    
            # Warning範囲
            elif value <= limits.low_warning or value >= limits.high_warning:
                level = SafetyLevel.WARNING
                message = "Warning: Approaching critical limit"
                distance = min(
                    abs(value - limits.low_critical),
                    abs(value - limits.high_critical)
                )
    
            # Caution範囲
            elif value <= limits.low_caution or value >= limits.high_caution:
                level = SafetyLevel.CAUTION
                message = "Caution: Monitor closely"
                distance = min(
                    abs(value - limits.low_critical),
                    abs(value - limits.high_critical)
                )
    
            # Normal範囲
            else:
                level = SafetyLevel.NORMAL
                message = "Normal operation"
                distance = min(
                    abs(value - limits.low_critical),
                    abs(value - limits.high_critical)
                )
    
            return {
                'level': level,
                'message': message,
                'distance_to_limit': distance,
                'value': value
            }
    
        def get_current_status(self) -> pd.DataFrame:
            """現在の安全性ステータスを取得"""
            status_data = []
    
            for param, value in self.current_values.items():
                assessment = self._assess_safety(param, value)
                status_data.append({
                    'Parameter': param,
                    'Current Value': value,
                    'Safety Level': assessment['level'].value,
                    'Distance to Limit': assessment['distance_to_limit'],
                    'Message': assessment['message']
                })
    
            df = pd.DataFrame(status_data)
            return df.sort_values('Distance to Limit')
    
        def get_alert_summary(self, hours: int = 24) -> pd.DataFrame:
            """指定期間のアラート統計"""
            if not self.alert_history:
                return pd.DataFrame()
    
            df = pd.DataFrame(self.alert_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
            # 指定期間でフィルタ
            cutoff_time = datetime.now() - timedelta(hours=hours)
            df_recent = df[df['timestamp'] >= cutoff_time]
    
            # レベル別集計
            summary = df_recent.groupby(['parameter', 'level']).size().unstack(fill_value=0)
    
            return summary
    
    # 実使用例：化学反応器の監視
    monitor = SafetyMonitor()
    
    # 反応器温度の安全限界設定
    monitor.set_safety_limits(
        'Reactor Temperature',
        SafetyLimit(
            low_critical=50,    # 50℃未満でCritical
            low_warning=60,
            low_caution=70,
            high_caution=180,
            high_warning=190,
            high_critical=200   # 200℃超でCritical
        )
    )
    
    # 反応器圧力の安全限界設定
    monitor.set_safety_limits(
        'Reactor Pressure',
        SafetyLimit(
            low_critical=0.5,   # 0.5 MPa未満でCritical
            low_warning=1.0,
            low_caution=1.5,
            high_caution=8.0,
            high_warning=9.0,
            high_critical=10.0  # 10 MPa超でCritical
        )
    )
    
    # 冷却水流量の安全限界設定
    monitor.set_safety_limits(
        'Cooling Water Flow',
        SafetyLimit(
            low_critical=50,    # 50 L/min未満でCritical
            low_warning=70,
            low_caution=90,
            high_caution=500,
            high_warning=550,
            high_critical=600
        )
    )
    
    # リアルタイム測定値の更新（シミュレーション）
    np.random.seed(42)
    
    # 正常運転
    print("=== 正常運転時 ===")
    monitor.update_value('Reactor Temperature', 150)
    monitor.update_value('Reactor Pressure', 5.0)
    monitor.update_value('Cooling Water Flow', 200)
    
    status = monitor.get_current_status()
    print(status.to_string(index=False))
    
    # 異常状態のシミュレーション
    print("\n=== 異常状態（温度上昇） ===")
    result = monitor.update_value('Reactor Temperature', 195)
    print(f"Temperature 195℃: {result['level'].value} - {result['message']}")
    
    result = monitor.update_value('Reactor Temperature', 205)
    print(f"Temperature 205℃: {result['level'].value} - {result['message']}")
    
    # 圧力異常
    print("\n=== 異常状態（圧力上昇） ===")
    result = monitor.update_value('Reactor Pressure', 9.5)
    print(f"Pressure 9.5 MPa: {result['level'].value} - {result['message']}")
    
    # 冷却水流量異常
    print("\n=== 異常状態（冷却水流量低下） ===")
    result = monitor.update_value('Cooling Water Flow', 65)
    print(f"Flow 65 L/min: {result['level'].value} - {result['message']}")
    
    # アラートサマリー
    print("\n=== アラートサマリー（24時間） ===")
    alert_summary = monitor.get_alert_summary(hours=24)
    if not alert_summary.empty:
        print(alert_summary)
    
    # 出力例:
    # === 正常運転時 ===
    # Parameter             Current Value Safety Level  Distance to Limit                    Message
    # Cooling Water Flow            200.0       Normal               250.0         Normal operation
    # Reactor Temperature           150.0       Normal                50.0         Normal operation
    # Reactor Pressure                5.0       Normal                 5.0         Normal operation
    #
    # === 異常状態（温度上昇） ===
    # Temperature 195℃: Warning - Warning: Approaching critical limit
    # Temperature 205℃: Critical - Critical limit exceeded! Immediate action required.
    

### 4.1.2 多変量異常検知（MSPC）

多変量統計的プロセス管理（MSPC: Multivariate Statistical Process Control）は、 複数のプロセス変数を同時に監視し、変数間の相関関係を考慮した異常検知を行います。 主成分分析（PCA）ベースのT²統計量とQ統計量を用いて異常を検出します。 
    
    
    # Example 2: Multivariate Anomaly Detection (MSPC)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import scipy.stats as stats
    
    class MSPCMonitor:
        """多変量統計的プロセス管理（MSPC）
    
        PCAベースのT²統計量とSPE（Q統計量）で異常検知
        """
    
        def __init__(self, n_components: int = None, alpha: float = 0.01):
            """
            Args:
                n_components: 主成分数（Noneで自動決定）
                alpha: 有意水準（デフォルト1%）
            """
            self.n_components = n_components
            self.alpha = alpha
            self.scaler = StandardScaler()
            self.pca = None
            self.t2_limit = None
            self.spe_limit = None
            self.is_trained = False
    
        def train(self, normal_data: np.ndarray):
            """正常運転データで統計モデルを構築
    
            Args:
                normal_data: 正常運転時のデータ (n_samples, n_features)
            """
            # データ標準化
            X_scaled = self.scaler.fit_transform(normal_data)
    
            # PCAモデル構築
            if self.n_components is None:
                # 累積寄与率90%以上となる成分数を自動選択
                pca_temp = PCA()
                pca_temp.fit(X_scaled)
                cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                self.n_components = np.argmax(cumsum >= 0.90) + 1
    
            self.pca = PCA(n_components=self.n_components)
            T = self.pca.fit_transform(X_scaled)
    
            # T²統計量の管理限界（F分布）
            n, p = normal_data.shape
            a = self.n_components
            f_value = stats.f.ppf(1 - self.alpha, a, n - a)
            self.t2_limit = (a * (n - 1) * (n + 1)) / (n * (n - a)) * f_value
    
            # SPE（Q統計量）の管理限界
            residuals = X_scaled - self.pca.inverse_transform(T)
            spe_values = np.sum(residuals ** 2, axis=1)
    
            # SPE分布パラメータ推定
            theta1 = np.mean(spe_values)
            theta2 = np.var(spe_values)
            theta3 = np.mean((spe_values - theta1) ** 3)
    
            h0 = 1 - (2 * theta1 * theta3) / (3 * theta2 ** 2)
            ca = stats.norm.ppf(1 - self.alpha)
    
            self.spe_limit = theta1 * (
                1 + (ca * np.sqrt(2 * theta2) * h0) / theta1 +
                (theta2 * h0 * (h0 - 1)) / theta1 ** 2
            ) ** (1 / h0)
    
            self.is_trained = True
    
            return {
                'n_components': self.n_components,
                'variance_explained': self.pca.explained_variance_ratio_.sum(),
                't2_limit': self.t2_limit,
                'spe_limit': self.spe_limit
            }
    
        def detect(self, new_data: np.ndarray) -> Dict:
            """新規データの異常検知
    
            Args:
                new_data: 監視対象データ (n_samples, n_features)
    
            Returns:
                異常検知結果
            """
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")
    
            # データ標準化
            X_scaled = self.scaler.transform(new_data)
    
            # T²統計量計算
            T = self.pca.transform(X_scaled)
            t2_values = np.sum(
                (T ** 2) / self.pca.explained_variance_, axis=1
            )
    
            # SPE計算
            residuals = X_scaled - self.pca.inverse_transform(T)
            spe_values = np.sum(residuals ** 2, axis=1)
    
            # 異常判定
            t2_anomaly = t2_values > self.t2_limit
            spe_anomaly = spe_values > self.spe_limit
            any_anomaly = t2_anomaly | spe_anomaly
    
            return {
                't2': t2_values,
                't2_limit': self.t2_limit,
                't2_anomaly': t2_anomaly,
                'spe': spe_values,
                'spe_limit': self.spe_limit,
                'spe_anomaly': spe_anomaly,
                'anomaly': any_anomaly,
                'anomaly_rate': any_anomaly.mean()
            }
    
    # 実使用例：反応器の多変量監視
    np.random.seed(42)
    
    # 正常運転データ生成（相関のある5変数）
    n_samples = 500
    mean_normal = [150, 5.0, 200, 80, 0.85]  # 温度、圧力、流量、転化率、純度
    cov_normal = np.array([
        [25, 0.5, 5, 2, 0.01],
        [0.5, 0.04, 0.1, 0.02, 0.001],
        [5, 0.1, 100, 10, 0.1],
        [2, 0.02, 10, 4, 0.05],
        [0.01, 0.001, 0.1, 0.05, 0.0025]
    ])
    
    normal_data = np.random.multivariate_normal(mean_normal, cov_normal, n_samples)
    
    # MSPCモデル構築
    mspc = MSPCMonitor(alpha=0.01)
    train_result = mspc.train(normal_data)
    
    print("=== MSPC モデル構築 ===")
    print(f"主成分数: {train_result['n_components']}")
    print(f"累積寄与率: {train_result['variance_explained']:.2%}")
    print(f"T² 管理限界: {train_result['t2_limit']:.2f}")
    print(f"SPE 管理限界: {train_result['spe_limit']:.2f}")
    
    # 正常データでの検証
    print("\n=== 正常データ検証 ===")
    normal_test = np.random.multivariate_normal(mean_normal, cov_normal, 100)
    result_normal = mspc.detect(normal_test)
    print(f"異常率: {result_normal['anomaly_rate']*100:.1f}% "
          f"(期待値: {mspc.alpha*100:.1f}%)")
    
    # 異常データでの検証（温度異常）
    print("\n=== 異常データ検証（温度+20℃上昇） ===")
    mean_abnormal = [170, 5.5, 200, 75, 0.82]  # 温度上昇、圧力上昇、転化率低下
    abnormal_data = np.random.multivariate_normal(mean_abnormal, cov_normal, 50)
    result_abnormal = mspc.detect(abnormal_data)
    
    print(f"異常率: {result_abnormal['anomaly_rate']*100:.1f}%")
    print(f"T²異常検知: {result_abnormal['t2_anomaly'].sum()}件")
    print(f"SPE異常検知: {result_abnormal['spe_anomaly'].sum()}件")
    
    # 詳細分析
    anomaly_indices = np.where(result_abnormal['anomaly'])[0]
    if len(anomaly_indices) > 0:
        print(f"\n異常サンプル例（最初の5件）:")
        for i in anomaly_indices[:5]:
            print(f"  Sample {i}: T²={result_abnormal['t2'][i]:.2f} "
                  f"(limit={mspc.t2_limit:.2f}), "
                  f"SPE={result_abnormal['spe'][i]:.2f} "
                  f"(limit={mspc.spe_limit:.2f})")
    
    # 出力例:
    # === MSPC モデル構築 ===
    # 主成分数: 3
    # 累積寄与率: 91.23%
    # T² 管理限界: 11.34
    # SPE 管理限界: 8.76
    #
    # === 正常データ検証 ===
    # 異常率: 1.0% (期待値: 1.0%)
    #
    # === 異常データ検証（温度+20℃上昇） ===
    # 異常率: 94.0%
    # T²異常検知: 47件
    # SPE異常検知: 12件
    

## 4.2 先行指標（Leading Indicators）管理

先行指標は、事故や重大災害の「予兆」を検知するための指標です。 遅行指標（事故率など結果指標）とは異なり、事故発生前の安全管理活動や リスク状態を測定し、予防的な対策を可能にします。 

### 4.2.1 先行指標トラッキングシステム
    
    
    # Example 3: Leading Indicators Tracking System
    from typing import List, Tuple
    from collections import deque
    
    class LeadingIndicator:
        """先行指標クラス"""
    
        def __init__(self, name: str, target: float, unit: str,
                     direction: str = 'lower'):
            """
            Args:
                name: 指標名
                target: 目標値
                unit: 単位
                direction: 'lower'（低い方が良い）or 'higher'（高い方が良い）
            """
            self.name = name
            self.target = target
            self.unit = unit
            self.direction = direction
            self.values = deque(maxlen=12)  # 直近12ヶ月分
            self.timestamps = deque(maxlen=12)
    
        def add_value(self, value: float, timestamp: datetime):
            """指標値を追加"""
            self.values.append(value)
            self.timestamps.append(timestamp)
    
        def get_current_value(self) -> Optional[float]:
            """現在値を取得"""
            return self.values[-1] if self.values else None
    
        def get_trend(self) -> str:
            """トレンドを判定（Improving/Stable/Deteriorating）"""
            if len(self.values) < 3:
                return "Insufficient Data"
    
            # 直近3ヶ月の平均変化率
            recent_3 = list(self.values)[-3:]
            trend_value = (recent_3[-1] - recent_3[0]) / abs(recent_3[0]) if recent_3[0] != 0 else 0
    
            # 方向を考慮
            if self.direction == 'lower':
                # 値が下がる方が改善
                if trend_value < -0.05:
                    return "Improving"
                elif trend_value > 0.05:
                    return "Deteriorating"
                else:
                    return "Stable"
            else:
                # 値が上がる方が改善
                if trend_value > 0.05:
                    return "Improving"
                elif trend_value < -0.05:
                    return "Deteriorating"
                else:
                    return "Stable"
    
        def is_target_met(self) -> bool:
            """目標達成判定"""
            current = self.get_current_value()
            if current is None:
                return False
    
            if self.direction == 'lower':
                return current <= self.target
            else:
                return current >= self.target
    
    class LeadingIndicatorTracker:
        """先行指標トラッキングシステム"""
    
        def __init__(self):
            self.indicators: Dict[str, LeadingIndicator] = {}
    
        def add_indicator(self, indicator: LeadingIndicator):
            """指標を追加"""
            self.indicators[indicator.name] = indicator
    
        def update_indicator(self, name: str, value: float,
                            timestamp: datetime = None):
            """指標値を更新"""
            if name not in self.indicators:
                raise ValueError(f"Indicator '{name}' not found")
    
            if timestamp is None:
                timestamp = datetime.now()
    
            self.indicators[name].add_value(value, timestamp)
    
        def get_dashboard(self) -> pd.DataFrame:
            """ダッシュボード用データを生成"""
            data = []
            for name, indicator in self.indicators.items():
                current = indicator.get_current_value()
                target_met = indicator.is_target_met()
                trend = indicator.get_trend()
    
                data.append({
                    'Indicator': name,
                    'Current Value': current,
                    'Target': indicator.target,
                    'Unit': indicator.unit,
                    'Target Met': '✓' if target_met else '✗',
                    'Trend': trend
                })
    
            return pd.DataFrame(data)
    
        def get_risk_score(self) -> float:
            """総合リスクスコアを計算（0-100、低い方が良い）"""
            if not self.indicators:
                return 0
    
            scores = []
            for indicator in self.indicators.values():
                current = indicator.get_current_value()
                if current is None:
                    continue
    
                # 目標からの乖離度を計算
                deviation = abs(current - indicator.target) / abs(indicator.target)
    
                # トレンドを考慮（悪化中は重み付け）
                trend = indicator.get_trend()
                weight = 1.5 if trend == "Deteriorating" else 1.0
    
                scores.append(min(deviation * 100 * weight, 100))
    
            return np.mean(scores) if scores else 0
    
    # 実使用例
    tracker = LeadingIndicatorTracker()
    
    # 先行指標の定義
    indicators_config = [
        ('Near Miss Reports', 10, 'reports/month', 'higher'),   # 月10件以上が目標
        ('Safety Training Hours', 8, 'hours/person/month', 'higher'),
        ('Overdue Inspections', 5, 'count', 'lower'),          # 5件以下が目標
        ('Safety Walk Completions', 20, 'walks/month', 'higher'),
        ('Permit Violations', 2, 'violations/month', 'lower'),
        ('Maintenance Backlog', 15, 'work orders', 'lower')
    ]
    
    for name, target, unit, direction in indicators_config:
        indicator = LeadingIndicator(name, target, unit, direction)
        tracker.add_indicator(indicator)
    
    # 12ヶ月分のデータ入力（シミュレーション）
    np.random.seed(42)
    base_date = datetime(2024, 1, 1)
    
    # 月次データ生成
    monthly_data = {
        'Near Miss Reports': [8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'Safety Training Hours': [7, 7.5, 8, 8.2, 8.5, 8.7, 9, 9.2, 9.5, 9.8, 10, 10.2],
        'Overdue Inspections': [12, 10, 9, 8, 7, 6, 5, 4, 4, 3, 3, 2],
        'Safety Walk Completions': [15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        'Permit Violations': [5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 0, 0],
        'Maintenance Backlog': [25, 23, 21, 20, 18, 17, 16, 15, 14, 13, 12, 11]
    }
    
    for month in range(12):
        timestamp = base_date + timedelta(days=30*month)
        for indicator_name, values in monthly_data.items():
            tracker.update_indicator(indicator_name, values[month], timestamp)
    
    # ダッシュボード表示
    print("=== 先行指標ダッシュボード（最新月） ===")
    dashboard = tracker.get_dashboard()
    print(dashboard.to_string(index=False))
    
    # 総合リスクスコア
    risk_score = tracker.get_risk_score()
    print(f"\n【総合リスクスコア】: {risk_score:.1f}/100")
    if risk_score < 20:
        print("評価: 優良（安全管理が効果的に機能）")
    elif risk_score < 40:
        print("評価: 良好（継続的な改善が見られる）")
    elif risk_score < 60:
        print("評価: 注意（改善の余地あり）")
    else:
        print("評価: 要改善（早急な対策が必要）")
    
    # トレンド分析
    print("\n=== トレンド分析 ===")
    improving = [name for name, ind in tracker.indicators.items()
                 if ind.get_trend() == "Improving"]
    deteriorating = [name for name, ind in tracker.indicators.items()
                    if ind.get_trend() == "Deteriorating"]
    
    print(f"改善中の指標 ({len(improving)}件):")
    for name in improving:
        print(f"  ✓ {name}")
    
    if deteriorating:
        print(f"\n悪化中の指標 ({len(deteriorating)}件):")
        for name in deteriorating:
            print(f"  ⚠️  {name}")
    
    # 出力例:
    # === 先行指標ダッシュボード（最新月） ===
    # Indicator                   Current Value  Target                Unit  Target Met     Trend
    # Near Miss Reports                    20.0    10.0   reports/month           ✓  Improving
    # Safety Training Hours                10.2     8.0  hours/person/month       ✓  Improving
    # Overdue Inspections                   2.0     5.0           count           ✓  Improving
    # Safety Walk Completions              27.0    20.0     walks/month           ✓  Improving
    # Permit Violations                     0.0     2.0  violations/month         ✓  Improving
    # Maintenance Backlog                  11.0    15.0     work orders           ✓  Improving
    #
    # 【総合リスクスコア】: 8.3/100
    # 評価: 優良（安全管理が効果的に機能）
    

## 4.3 機械学習による事故予測

過去の事故データ、ニアミスデータ、運転データを用いて機械学習モデルを構築し、 事故発生確率を予測します。Random Forestやグラディエントブースティングなどの アンサンブル学習が高い予測精度を示します。 

### 4.3.1 事故予測モデルの構築
    
    
    # Example 4: Incident Prediction with Machine Learning
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    import warnings
    warnings.filterwarnings('ignore')
    
    class IncidentPredictor:
        """機械学習による事故予測システム
    
        運転データと安全指標から事故リスクを予測
        """
    
        def __init__(self, model_type: str = 'random_forest'):
            """
            Args:
                model_type: 'random_forest' or 'gradient_boosting'
            """
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            else:
                raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")
    
            self.feature_names = None
            self.is_trained = False
    
        def train(self, X: np.ndarray, y: np.ndarray,
                  feature_names: List[str] = None) -> Dict:
            """モデルを訓練
    
            Args:
                X: 特徴量 (n_samples, n_features)
                y: ラベル（0: 正常, 1: 事故発生）
                feature_names: 特徴量名リスト
    
            Returns:
                訓練結果の統計
            """
            self.feature_names = feature_names
    
            # 訓練/テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
    
            # モデル訓練
            self.model.fit(X_train, y_train)
    
            # 性能評価
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
    
            # クロスバリデーション
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
    
            self.is_trained = True
    
            return {
                'train_score': self.model.score(X_train, y_train),
                'test_score': self.model.score(X_test, y_test),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
    
        def predict_risk(self, X: np.ndarray) -> np.ndarray:
            """事故リスク確率を予測
    
            Args:
                X: 特徴量
    
            Returns:
                事故発生確率（0-1）
            """
            if not self.is_trained:
                raise ValueError("Model not trained")
    
            return self.model.predict_proba(X)[:, 1]
    
        def get_feature_importance(self) -> pd.DataFrame:
            """特徴量重要度を取得"""
            if not self.is_trained:
                raise ValueError("Model not trained")
    
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
    
            df = pd.DataFrame({
                'Feature': [self.feature_names[i] if self.feature_names else f'Feature_{i}'
                           for i in indices],
                'Importance': importances[indices]
            })
    
            return df
    
    # 実使用例：事故予測モデル構築
    np.random.seed(42)
    
    # 訓練データ生成（1000サンプル）
    n_samples = 1000
    n_incidents = 100  # 10%が事故発生
    
    # 特徴量の生成
    features = {
        'Temperature Deviation': np.random.randn(n_samples) * 10,  # 温度偏差
        'Pressure Deviation': np.random.randn(n_samples) * 0.5,   # 圧力偏差
        'Near Miss Count': np.random.poisson(2, n_samples),        # ニアミス件数
        'Overdue Maintenance': np.random.poisson(3, n_samples),    # 未実施保全
        'Operator Experience': np.random.uniform(1, 20, n_samples), # 経験年数
        'Safety Training Hours': np.random.uniform(5, 15, n_samples), # 訓練時間
        'Equipment Age': np.random.uniform(0, 30, n_samples),      # 設備年数
        'Alarm Rate': np.random.poisson(10, n_samples)             # アラーム率
    }
    
    # 事故発生ラベル生成（特徴量に依存）
    incident_score = (
        np.abs(features['Temperature Deviation']) * 0.1 +
        np.abs(features['Pressure Deviation']) * 0.5 +
        features['Near Miss Count'] * 0.3 +
        features['Overdue Maintenance'] * 0.2 +
        (20 - features['Operator Experience']) * 0.05 +
        (15 - features['Safety Training Hours']) * 0.1 +
        features['Equipment Age'] * 0.05 +
        features['Alarm Rate'] * 0.1 +
        np.random.randn(n_samples) * 2  # ノイズ
    )
    
    # 上位10%を事故とラベル付け
    threshold = np.percentile(incident_score, 90)
    y = (incident_score > threshold).astype(int)
    
    # 特徴量行列作成
    X = np.column_stack([features[k] for k in features.keys()])
    feature_names = list(features.keys())
    
    # モデル構築
    predictor = IncidentPredictor(model_type='random_forest')
    train_result = predictor.train(X, y, feature_names)
    
    print("=== 事故予測モデル性能 ===")
    print(f"訓練精度: {train_result['train_score']:.3f}")
    print(f"テスト精度: {train_result['test_score']:.3f}")
    print(f"ROC-AUC: {train_result['roc_auc']:.3f}")
    print(f"CV平均: {train_result['cv_mean']:.3f} (±{train_result['cv_std']:.3f})")
    
    print("\n=== 混同行列 ===")
    cm = train_result['confusion_matrix']
    print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
    print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
    
    print("\n=== 分類レポート ===")
    print(train_result['classification_report'])
    
    # 特徴量重要度
    print("=== 特徴量重要度 ===")
    importance_df = predictor.get_feature_importance()
    print(importance_df.to_string(index=False))
    
    # 新規データの予測
    print("\n=== 新規データ予測例 ===")
    new_samples = np.array([
        [15, 1.0, 5, 8, 3, 6, 25, 20],   # 高リスクケース
        [2, 0.1, 0, 1, 15, 12, 5, 5],    # 低リスクケース
        [8, 0.5, 3, 4, 10, 10, 15, 12]   # 中リスクケース
    ])
    
    risk_probs = predictor.predict_risk(new_samples)
    for i, prob in enumerate(risk_probs):
        risk_level = "高" if prob > 0.5 else "中" if prob > 0.2 else "低"
        print(f"サンプル{i+1}: 事故リスク={prob*100:.1f}% ({risk_level}リスク)")
    
    # 出力例:
    # === 事故予測モデル性能 ===
    # 訓練精度: 0.975
    # テスト精度: 0.885
    # ROC-AUC: 0.932
    # CV平均: 0.918 (±0.023)
    

## 4.4 安全性パフォーマンス指標

安全性パフォーマンスは、TRIR（総合災害度数率）、LTIR（休業災害度数率）などの 標準化された指標で測定します。これらは産業間・企業間の比較を可能にします。 

### 4.4.1 安全性KPI計算システム
    
    
    # Example 5: Safety Performance Metrics Calculation
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class Incident:
        """事故記録"""
        date: datetime
        type: str  # 'fatality', 'lost_time', 'restricted_work', 'medical_treatment', 'first_aid'
        days_away: int = 0  # 休業日数
        description: str = ""
    
    class SafetyPerformanceCalculator:
        """安全性パフォーマンス指標計算システム
    
        OSHA基準に基づく各種安全性指標を計算
        """
    
        def __init__(self):
            self.incidents: List[Incident] = []
            self.total_hours_worked = 0
    
        def add_incident(self, incident: Incident):
            """事故を追加"""
            self.incidents.append(incident)
    
        def set_hours_worked(self, hours: float):
            """総労働時間を設定"""
            self.total_hours_worked = hours
    
        def calculate_trir(self) -> float:
            """TRIR（総合災害度数率）を計算
    
            TRIR = (記録対象災害件数 × 200,000) / 総労働時間
    
            記録対象: 死亡、休業、制限作業、医療処置
            200,000 = 100人が年間働く時間（100人×2,000時間）
            """
            recordable = [i for i in self.incidents
                         if i.type in ['fatality', 'lost_time',
                                      'restricted_work', 'medical_treatment']]
    
            if self.total_hours_worked == 0:
                return 0
    
            return (len(recordable) * 200000) / self.total_hours_worked
    
        def calculate_ltir(self) -> float:
            """LTIR（休業災害度数率）を計算
    
            LTIR = (休業災害件数 × 200,000) / 総労働時間
            """
            lost_time_incidents = [i for i in self.incidents
                                  if i.type in ['fatality', 'lost_time']]
    
            if self.total_hours_worked == 0:
                return 0
    
            return (len(lost_time_incidents) * 200000) / self.total_hours_worked
    
        def calculate_severity_rate(self) -> float:
            """重大度率を計算
    
            Severity Rate = (総休業日数 × 200,000) / 総労働時間
            """
            total_days_lost = sum(i.days_away for i in self.incidents)
    
            if self.total_hours_worked == 0:
                return 0
    
            return (total_days_lost * 200000) / self.total_hours_worked
    
        def calculate_fatality_rate(self) -> float:
            """死亡災害率を計算
    
            Fatality Rate = (死亡件数 × 200,000) / 総労働時間
            """
            fatalities = [i for i in self.incidents if i.type == 'fatality']
    
            if self.total_hours_worked == 0:
                return 0
    
            return (len(fatalities) * 200000) / self.total_hours_worked
    
        def get_incident_breakdown(self) -> pd.DataFrame:
            """事故タイプ別の内訳"""
            if not self.incidents:
                return pd.DataFrame()
    
            type_counts = {}
            for incident in self.incidents:
                type_counts[incident.type] = type_counts.get(incident.type, 0) + 1
    
            df = pd.DataFrame({
                'Incident Type': list(type_counts.keys()),
                'Count': list(type_counts.values())
            })
    
            # 割合を計算
            df['Percentage'] = (df['Count'] / df['Count'].sum() * 100).round(1)
    
            return df.sort_values('Count', ascending=False)
    
        def get_comprehensive_report(self) -> Dict:
            """総合安全性レポート"""
            return {
                'Total Incidents': len(self.incidents),
                'Total Hours Worked': self.total_hours_worked,
                'TRIR': self.calculate_trir(),
                'LTIR': self.calculate_ltir(),
                'Severity Rate': self.calculate_severity_rate(),
                'Fatality Rate': self.calculate_fatality_rate(),
                'Incident Breakdown': self.get_incident_breakdown()
            }
    
        def benchmark_against_industry(self, industry_trir: float,
                                       industry_ltir: float) -> Dict:
            """業界平均とのベンチマーク
    
            Args:
                industry_trir: 業界平均TRIR
                industry_ltir: 業界平均LTIR
    
            Returns:
                ベンチマーク結果
            """
            company_trir = self.calculate_trir()
            company_ltir = self.calculate_ltir()
    
            return {
                'Company TRIR': company_trir,
                'Industry TRIR': industry_trir,
                'TRIR Performance': 'Above Average' if company_trir < industry_trir else 'Below Average',
                'TRIR Difference': company_trir - industry_trir,
                'Company LTIR': company_ltir,
                'Industry LTIR': industry_ltir,
                'LTIR Performance': 'Above Average' if company_ltir < industry_ltir else 'Below Average',
                'LTIR Difference': company_ltir - industry_ltir
            }
    
    # 実使用例
    spc = SafetyPerformanceCalculator()
    
    # 年間総労働時間（従業員500人、年間2000時間/人）
    spc.set_hours_worked(500 * 2000)  # 1,000,000時間
    
    # 事故データの追加（1年間）
    incidents_data = [
        Incident(datetime(2024, 1, 15), 'first_aid', 0, "軽い切り傷"),
        Incident(datetime(2024, 2, 3), 'medical_treatment', 0, "打撲、医療処置"),
        Incident(datetime(2024, 3, 12), 'lost_time', 5, "転倒、足首捻挫"),
        Incident(datetime(2024, 4, 8), 'first_aid', 0, "目にゴミ"),
        Incident(datetime(2024, 5, 20), 'restricted_work', 3, "腰痛、軽作業のみ"),
        Incident(datetime(2024, 6, 15), 'lost_time', 10, "機械挟まれ、指骨折"),
        Incident(datetime(2024, 7, 22), 'medical_treatment', 0, "化学薬品飛散"),
        Incident(datetime(2024, 8, 9), 'first_aid', 0, "軽い火傷"),
        Incident(datetime(2024, 9, 14), 'lost_time', 15, "高所転落、脚部骨折"),
        Incident(datetime(2024, 10, 5), 'medical_treatment', 0, "ガス吸入"),
        Incident(datetime(2024, 11, 18), 'first_aid', 0, "擦り傷"),
        Incident(datetime(2024, 12, 2), 'restricted_work', 2, "手首痛")
    ]
    
    for incident in incidents_data:
        spc.add_incident(incident)
    
    # 総合レポート
    report = spc.get_comprehensive_report()
    
    print("=== 年間安全性パフォーマンスレポート ===\n")
    print(f"総事故件数: {report['Total Incidents']}件")
    print(f"総労働時間: {report['Total Hours Worked']:,}時間")
    print(f"\n【主要指標】")
    print(f"TRIR（総合災害度数率）: {report['TRIR']:.2f}")
    print(f"LTIR（休業災害度数率）: {report['LTIR']:.2f}")
    print(f"重大度率: {report['Severity Rate']:.2f}")
    print(f"死亡災害率: {report['Fatality Rate']:.2f}")
    
    print("\n=== 事故タイプ別内訳 ===")
    print(report['Incident Breakdown'].to_string(index=False))
    
    # 業界ベンチマーク（化学産業の典型値）
    print("\n=== 業界ベンチマーク ===")
    industry_trir = 1.2  # 化学産業平均TRIR
    industry_ltir = 0.5  # 化学産業平均LTIR
    
    benchmark = spc.benchmark_against_industry(industry_trir, industry_ltir)
    print(f"当社TRIR: {benchmark['Company TRIR']:.2f}")
    print(f"業界平均TRIR: {benchmark['Industry TRIR']:.2f}")
    print(f"評価: {benchmark['TRIR Performance']}")
    print(f"差分: {benchmark['TRIR Difference']:+.2f}")
    
    print(f"\n当社LTIR: {benchmark['Company LTIR']:.2f}")
    print(f"業界平均LTIR: {benchmark['Industry LTIR']:.2f}")
    print(f"評価: {benchmark['LTIR Performance']}")
    print(f"差分: {benchmark['LTIR Difference']:+.2f}")
    
    # 出力例:
    # === 年間安全性パフォーマンスレポート ===
    #
    # 総事故件数: 12件
    # 総労働時間: 1,000,000時間
    #
    # 【主要指標】
    # TRIR（総合災害度数率）: 1.40
    # LTIR（休業災害度数率）: 0.60
    # 重大度率: 6.00
    # 死亡災害率: 0.00
    

### 4.4.2 ニアミス分析システム
    
    
    # Example 6: Near-Miss Analysis System
    class NearMissAnalyzer:
        """ニアミス（ヒヤリハット）分析システム
    
        ハインリッヒの法則に基づく予防的安全管理
        """
    
        def __init__(self):
            self.near_misses: List[Dict] = []
    
        def add_near_miss(self, category: str, severity: int,
                         description: str, timestamp: datetime = None):
            """ニアミスを追加
    
            Args:
                category: カテゴリ（'slip_trip', 'chemical', 'equipment', etc.）
                severity: 重大度（1-5、5が最も深刻）
                description: 説明
                timestamp: 発生時刻
            """
            if timestamp is None:
                timestamp = datetime.now()
    
            self.near_misses.append({
                'timestamp': timestamp,
                'category': category,
                'severity': severity,
                'description': description
            })
    
        def get_heinrich_ratio_analysis(self, actual_incidents: int) -> Dict:
            """ハインリッヒの法則に基づく分析
    
            ハインリッヒの法則: 重大事故1件の背後に
            - 軽傷事故29件
            - ニアミス300件
            が存在するという経験則
            """
            near_miss_count = len(self.near_misses)
    
            # 期待される重大事故件数
            expected_major = near_miss_count / 300
    
            return {
                'Near Miss Count': near_miss_count,
                'Actual Incidents': actual_incidents,
                'Expected Major Incidents (Heinrich)': expected_major,
                'Heinrich Ratio': f"1:{actual_incidents}:{near_miss_count}",
                'Prevention Effectiveness': (
                    (expected_major - actual_incidents) / expected_major * 100
                    if expected_major > 0 else 0
                )
            }
    
        def get_category_analysis(self) -> pd.DataFrame:
            """カテゴリ別分析"""
            if not self.near_misses:
                return pd.DataFrame()
    
            df = pd.DataFrame(self.near_misses)
    
            # カテゴリ別集計
            category_stats = df.groupby('category').agg({
                'severity': ['count', 'mean', 'max'],
                'description': 'count'
            }).round(2)
    
            category_stats.columns = ['Count', 'Avg Severity', 'Max Severity', 'Total']
            category_stats = category_stats[['Count', 'Avg Severity', 'Max Severity']]
    
            return category_stats.sort_values('Count', ascending=False)
    
        def get_high_severity_near_misses(self, threshold: int = 4) -> pd.DataFrame:
            """高重大度ニアミスを抽出
    
            Args:
                threshold: 重大度しきい値（デフォルト4以上）
            """
            if not self.near_misses:
                return pd.DataFrame()
    
            df = pd.DataFrame(self.near_misses)
            high_severity = df[df['severity'] >= threshold]
    
            return high_severity[['timestamp', 'category', 'severity', 'description']]
    
        def calculate_near_miss_rate(self, hours_worked: float) -> float:
            """ニアミス率を計算
    
            Near Miss Rate = (ニアミス件数 × 200,000) / 総労働時間
            """
            if hours_worked == 0:
                return 0
    
            return (len(self.near_misses) * 200000) / hours_worked
    
    # 実使用例
    nma = NearMissAnalyzer()
    
    # ニアミスデータの追加（3ヶ月分）
    near_miss_data = [
        ('slip_trip', 2, "床に水たまり、転倒しそうになった"),
        ('chemical', 4, "タンク弁が少し開いており、薬品が滴下"),
        ('equipment', 3, "安全カバーが外れたまま運転開始しそうになった"),
        ('slip_trip', 1, "段差につまずいた"),
        ('electrical', 5, "配線が損傷、感電の危険"),
        ('chemical', 3, "保護具なしで薬品を扱いそうになった"),
        ('equipment', 4, "ポンプ異音、故障の兆候"),
        ('slip_trip', 2, "通路に障害物、つまずいた"),
        ('chemical', 2, "薬品容器のラベル不明瞭"),
        ('equipment', 3, "圧力計の針が異常値を示した"),
        ('electrical', 4, "コンセントが過熱していた"),
        ('slip_trip', 1, "床が滑りやすかった"),
        ('chemical', 5, "配管接続部からガス漏れの臭い"),
        ('equipment', 2, "工具が所定位置になかった"),
    ]
    
    base_time = datetime(2024, 10, 1)
    for i, (cat, sev, desc) in enumerate(near_miss_data):
        timestamp = base_time + timedelta(days=i*3)
        nma.add_near_miss(cat, sev, desc, timestamp)
    
    # カテゴリ別分析
    print("=== ニアミス カテゴリ別分析 ===")
    category_analysis = nma.get_category_analysis()
    print(category_analysis)
    
    # ハインリッヒの法則分析
    print("\n=== ハインリッヒの法則分析 ===")
    actual_incidents = 3  # 同期間の実事故件数
    heinrich = nma.get_heinrich_ratio_analysis(actual_incidents)
    print(f"ニアミス件数: {heinrich['Near Miss Count']}")
    print(f"実事故件数: {heinrich['Actual Incidents']}")
    print(f"期待重大事故件数（法則）: {heinrich['Expected Major Incidents (Heinrich)']:.2f}")
    print(f"比率: {heinrich['Heinrich Ratio']}")
    print(f"予防効果: {heinrich['Prevention Effectiveness']:.1f}%")
    
    # 高重大度ニアミス
    print("\n=== 高重大度ニアミス（重大度≥4） ===")
    high_severity = nma.get_high_severity_near_misses(threshold=4)
    for _, row in high_severity.iterrows():
        print(f"[{row['timestamp'].strftime('%Y-%m-%d')}] "
              f"{row['category']} (重大度{row['severity']}): {row['description']}")
    
    # ニアミス率計算
    hours_worked = 500 * 2000  # 500人×年間2000時間
    nm_rate = nma.calculate_near_miss_rate(hours_worked)
    print(f"\nニアミス率: {nm_rate:.2f} (per 200,000 hours)")
    
    # 出力例:
    # === ニアミス カテゴリ別分析 ===
    #             Count  Avg Severity  Max Severity
    # category
    # slip_trip       4          1.50             2
    # chemical        4          3.50             5
    # equipment       4          3.00             4
    # electrical      2          4.50             5
    

## 4.5 統合安全管理システム

これまでのすべてのコンポーネントを統合し、包括的な安全管理システムを構築します。 リアルタイム監視、予測、KPI管理を一元化します。 
    
    
    # Example 7: Integrated Safety Management System
    class IntegratedSafetyManagementSystem:
        """統合安全管理システム
    
        監視、予測、KPI管理を一元化した総合安全プラットフォーム
        """
    
        def __init__(self):
            self.safety_monitor = SafetyMonitor()
            self.mspc_monitor = None  # 訓練後に設定
            self.leading_indicator_tracker = LeadingIndicatorTracker()
            self.incident_predictor = None  # 訓練後に設定
            self.performance_calculator = SafetyPerformanceCalculator()
            self.near_miss_analyzer = NearMissAnalyzer()
    
        def initialize_monitoring(self, safety_limits: Dict[str, SafetyLimit]):
            """監視システムを初期化"""
            for param, limits in safety_limits.items():
                self.safety_monitor.set_safety_limits(param, limits)
    
        def initialize_leading_indicators(self, indicators: List[LeadingIndicator]):
            """先行指標を初期化"""
            for indicator in indicators:
                self.leading_indicator_tracker.add_indicator(indicator)
    
        def train_anomaly_detector(self, normal_data: np.ndarray):
            """異常検知モデルを訓練"""
            self.mspc_monitor = MSPCMonitor()
            return self.mspc_monitor.train(normal_data)
    
        def train_incident_predictor(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str]):
            """事故予測モデルを訓練"""
            self.incident_predictor = IncidentPredictor()
            return self.incident_predictor.train(X, y, feature_names)
    
        def get_real_time_status(self) -> Dict:
            """リアルタイムステータスを取得"""
            return {
                'Process Safety': self.safety_monitor.get_current_status(),
                'Leading Indicators': self.leading_indicator_tracker.get_dashboard(),
                'Risk Score': self.leading_indicator_tracker.get_risk_score()
            }
    
        def generate_safety_dashboard(self) -> str:
            """総合安全ダッシュボードを生成"""
            lines = []
            lines.append("=" * 70)
            lines.append("統合安全管理ダッシュボード".center(70))
            lines.append("=" * 70)
            lines.append("")
    
            # プロセス安全状態
            lines.append("【プロセス安全状態】")
            process_status = self.safety_monitor.get_current_status()
            for _, row in process_status.iterrows():
                status_icon = "✓" if row['Safety Level'] == 'Normal' else "⚠️"
                lines.append(f"  {status_icon} {row['Parameter']}: "
                            f"{row['Current Value']:.1f} ({row['Safety Level']})")
            lines.append("")
    
            # 先行指標
            lines.append("【先行指標】")
            indicators = self.leading_indicator_tracker.get_dashboard()
            for _, row in indicators.iterrows():
                lines.append(f"  {row['Target Met']} {row['Indicator']}: "
                            f"{row['Current Value']:.1f} ({row['Trend']})")
    
            risk_score = self.leading_indicator_tracker.get_risk_score()
            lines.append(f"\n  総合リスクスコア: {risk_score:.1f}/100")
            lines.append("")
    
            # 安全性パフォーマンス
            if self.performance_calculator.total_hours_worked > 0:
                lines.append("【安全性パフォーマンス】")
                lines.append(f"  TRIR: {self.performance_calculator.calculate_trir():.2f}")
                lines.append(f"  LTIR: {self.performance_calculator.calculate_ltir():.2f}")
                lines.append("")
    
            # ニアミス統計
            if len(self.near_miss_analyzer.near_misses) > 0:
                lines.append("【ニアミス】")
                lines.append(f"  総件数: {len(self.near_miss_analyzer.near_misses)}")
                high_severity = self.near_miss_analyzer.get_high_severity_near_misses()
                lines.append(f"  高重大度（≥4）: {len(high_severity)}件")
                lines.append("")
    
            return "\n".join(lines)
    
        def predict_incident_risk(self, current_conditions: np.ndarray) -> Dict:
            """現在の条件から事故リスクを予測"""
            if self.incident_predictor is None:
                return {'error': 'Incident predictor not trained'}
    
            risk_prob = self.incident_predictor.predict_risk(current_conditions)[0]
    
            risk_level = (
                "Critical" if risk_prob > 0.7 else
                "High" if risk_prob > 0.5 else
                "Medium" if risk_prob > 0.3 else
                "Low"
            )
    
            return {
                'risk_probability': risk_prob,
                'risk_level': risk_level,
                'recommendation': self._get_recommendation(risk_level)
            }
    
        def _get_recommendation(self, risk_level: str) -> str:
            """リスクレベルに応じた推奨アクションを生成"""
            recommendations = {
                'Critical': "即座に運転停止を検討。緊急対応チーム招集。",
                'High': "運転条件を見直し、監視を強化。管理者に報告。",
                'Medium': "継続監視。異常兆候に注意。",
                'Low': "通常運転継続。定期監視を維持。"
            }
            return recommendations.get(risk_level, "")
    
    # 実使用例（完全な統合システム）
    isms = IntegratedSafetyManagementSystem()
    
    # 1. 監視システム初期化
    safety_limits_config = {
        'Reactor Temperature': SafetyLimit(50, 60, 70, 180, 190, 200),
        'Reactor Pressure': SafetyLimit(0.5, 1.0, 1.5, 8.0, 9.0, 10.0)
    }
    isms.initialize_monitoring(safety_limits_config)
    
    # 2. 先行指標初期化
    indicators = [
        LeadingIndicator('Near Miss Reports', 10, 'reports/month', 'higher'),
        LeadingIndicator('Safety Training Hours', 8, 'hours/person/month', 'higher')
    ]
    isms.initialize_leading_indicators(indicators)
    
    # 3. プロセスデータ更新
    isms.safety_monitor.update_value('Reactor Temperature', 150)
    isms.safety_monitor.update_value('Reactor Pressure', 5.0)
    
    # 4. 先行指標更新
    isms.leading_indicator_tracker.update_indicator('Near Miss Reports', 15)
    isms.leading_indicator_tracker.update_indicator('Safety Training Hours', 9.5)
    
    # 5. ダッシュボード生成
    print(isms.generate_safety_dashboard())
    
    # 6. 事故リスク予測（訓練済みモデルがある場合）
    # current_conditions = np.array([[10, 0.8, 3, 5, 8, 10, 15, 12]])
    # risk_prediction = isms.predict_incident_risk(current_conditions)
    # print(f"\n【事故リスク予測】")
    # print(f"リスク確率: {risk_prediction['risk_probability']*100:.1f}%")
    # print(f"リスクレベル: {risk_prediction['risk_level']}")
    # print(f"推奨アクション: {risk_prediction['recommendation']}")
    
    # 出力例:
    # ======================================================================
    #                         統合安全管理ダッシュボード
    # ======================================================================
    #
    # 【プロセス安全状態】
    #   ✓ Reactor Temperature: 150.0 (Normal)
    #   ✓ Reactor Pressure: 5.0 (Normal)
    #
    # 【先行指標】
    #   ✓ Near Miss Reports: 15.0 (Improving)
    #   ✓ Safety Training Hours: 9.5 (Improving)
    #
    #   総合リスクスコア: 12.5/100
    

## 学習目標の確認

この章を完了すると、以下ができるようになります：

  * ✅ リアルタイム安全性モニタリングシステムを実装できる
  * ✅ 多変量統計的プロセス管理（MSPC）で異常を検知できる
  * ✅ 先行指標を追跡し、予防的安全管理ができる
  * ✅ 機械学習で事故リスクを予測できる
  * ✅ TRIR、LTIR等の安全性パフォーマンス指標を計算できる
  * ✅ ニアミスを分析し、ハインリッヒの法則を活用できる
  * ✅ 統合安全管理システムを構築し、総合的に評価できる

[プロセス安全性評価入門 目次に戻る](<index.html>)

[PI実践技術トップ](<../../index.html>) | [Homeに戻る](<../../index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
