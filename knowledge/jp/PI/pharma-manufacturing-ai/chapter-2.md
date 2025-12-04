---
title: 第2章 電子バッチ記録解析と逸脱管理
chapter_title: 第2章 電子バッチ記録解析と逸脱管理
subtitle: Electronic Batch Record Analysis and Deviation Management
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/pharma-manufacturing-ai/chapter-2.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Pharma Manufacturing Ai](<../../PI/pharma-manufacturing-ai/index.html>)›Chapter 2

[← シリーズ目次に戻る](<index.html>)

## 📖 本章の概要

医薬品製造における電子バッチ記録（EBR: Electronic Batch Record）は、製造プロセスの透明性と トレーサビリティを確保する重要なシステムです。本章では、EBRデータの自動解析、 異常検知、根本原因分析（RCA）、是正措置・予防措置（CAPA）の提案まで、 AIを活用した包括的な逸脱管理システムの構築方法を学びます。 

### 🎯 学習目標

  * 電子バッチ記録（EBR）の構造とデータ解析手法
  * バッチトレンド分析によるプロセス変動の可視化
  * 機械学習による異常バッチの自動検出
  * 根本原因分析（RCA）のためのデータマイニング
  * CAPA（是正措置・予防措置）提案の自動生成
  * 逸脱管理ワークフローの自動化
  * GMP準拠の文書管理とバージョン管理

## 📋 2.1 電子バッチ記録（EBR）の基礎

### EBRの構成要素

電子バッチ記録は以下の主要要素から構成されます：

  * **バッチヘッダ** : バッチ番号、製品名、製造日、ロット番号
  * **原材料記録** : 使用原料、数量、ロット番号、有効期限
  * **工程パラメータ** : 温度、圧力、時間、pH、流量など
  * **中間体試験** : 各工程での品質確認結果
  * **最終製品試験** : 含量、溶出、純度、微生物試験
  * **逸脱記録** : 異常発生、原因、対策、承認
  * **電子署名** : 作業者、確認者、承認者の署名

**🏭 GMP要件（21 CFR Part 11）**  
・電子記録の真正性（Authenticity）: 改ざん防止  
・完全性（Integrity）: データの一貫性と正確性  
・信頼性（Reliability）: システムの安定動作  
・利用可能性（Availability）: 必要時のアクセス保証  
・監査証跡（Audit Trail）: すべての変更履歴の記録 

### 💻 コード例2.1: EBRデータモデルとバッチトレンド分析
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    import json
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class ElectronicBatchRecord:
        """電子バッチ記録（EBR）管理クラス"""
    
        def __init__(self, batch_id, product_name, manufacturing_date):
            self.batch_id = batch_id
            self.product_name = product_name
            self.manufacturing_date = manufacturing_date
            self.process_parameters = {}
            self.quality_tests = {}
            self.deviations = []
            self.signatures = []
            self.audit_trail = []
    
        def add_process_parameter(self, step_name, parameter_name, target, actual, unit, tolerance=None):
            """工程パラメータの記録"""
            if step_name not in self.process_parameters:
                self.process_parameters[step_name] = []
    
            param = {
                'parameter': parameter_name,
                'target': target,
                'actual': actual,
                'unit': unit,
                'tolerance': tolerance,
                'timestamp': datetime.now().isoformat(),
                'in_spec': self._check_tolerance(target, actual, tolerance) if tolerance else True
            }
    
            self.process_parameters[step_name].append(param)
            self._log_audit(f"工程パラメータ記録: {step_name} - {parameter_name}")
    
        def _check_tolerance(self, target, actual, tolerance):
            """許容範囲チェック"""
            lower = target - tolerance
            upper = target + tolerance
            return lower <= actual <= upper
    
        def add_deviation(self, description, severity, root_cause=None, capa=None):
            """逸脱の記録"""
            deviation = {
                'id': f"DEV-{self.batch_id}-{len(self.deviations)+1:03d}",
                'description': description,
                'severity': severity,  # Critical, Major, Minor
                'root_cause': root_cause,
                'capa': capa,
                'timestamp': datetime.now().isoformat(),
                'status': 'Open'
            }
            self.deviations.append(deviation)
            self._log_audit(f"逸脱記録: {deviation['id']}")
    
        def add_signature(self, role, user_name):
            """電子署名の記録"""
            signature = {
                'role': role,
                'user': user_name,
                'timestamp': datetime.now().isoformat()
            }
            self.signatures.append(signature)
            self._log_audit(f"電子署名: {role} by {user_name}")
    
        def _log_audit(self, action):
            """監査証跡の記録"""
            entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action
            }
            self.audit_trail.append(entry)
    
        def to_dict(self):
            """辞書形式への変換"""
            return {
                'batch_id': self.batch_id,
                'product_name': self.product_name,
                'manufacturing_date': self.manufacturing_date,
                'process_parameters': self.process_parameters,
                'quality_tests': self.quality_tests,
                'deviations': self.deviations,
                'signatures': self.signatures,
                'audit_trail': self.audit_trail
            }
    
        def export_json(self, filename):
            """JSON形式でエクスポート"""
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"EBRを {filename} にエクスポートしました")
    
    
    class BatchTrendAnalyzer:
        """バッチトレンド分析クラス"""
    
        def __init__(self):
            self.batches = []
    
        def generate_batch_data(self, n_batches=50):
            """サンプルバッチデータの生成"""
            np.random.seed(42)
            start_date = datetime(2025, 1, 1)
    
            for i in range(n_batches):
                batch_id = f"B-2025-{i+1:04d}"
                mfg_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    
                # 正常バッチ（1-35）
                if i < 35:
                    reaction_temp = np.random.normal(80, 1, 1)[0]
                    reaction_time = np.random.normal(120, 5, 1)[0]
                    yield_value = np.random.normal(95, 2, 1)[0]
                    purity = np.random.normal(99.5, 0.3, 1)[0]
    
                # 温度異常バッチ（36-40）
                elif 35 <= i < 40:
                    reaction_temp = np.random.normal(85, 2, 1)[0]  # 温度上昇
                    reaction_time = np.random.normal(120, 5, 1)[0]
                    yield_value = np.random.normal(92, 3, 1)[0]  # 収率低下
                    purity = np.random.normal(99.2, 0.5, 1)[0]
    
                # 時間異常バッチ（41-45）
                elif 40 <= i < 45:
                    reaction_temp = np.random.normal(80, 1, 1)[0]
                    reaction_time = np.random.normal(140, 10, 1)[0]  # 時間延長
                    yield_value = np.random.normal(93, 2, 1)[0]
                    purity = np.random.normal(99.3, 0.4, 1)[0]
    
                # 複合異常バッチ（46-50）
                else:
                    reaction_temp = np.random.normal(83, 2, 1)[0]
                    reaction_time = np.random.normal(130, 8, 1)[0]
                    yield_value = np.random.normal(90, 3, 1)[0]
                    purity = np.random.normal(99.0, 0.6, 1)[0]
    
                self.batches.append({
                    'batch_id': batch_id,
                    'date': mfg_date,
                    'reaction_temp': reaction_temp,
                    'reaction_time': reaction_time,
                    'yield': yield_value,
                    'purity': purity
                })
    
            return pd.DataFrame(self.batches)
    
        def plot_batch_trends(self, df):
            """バッチトレンドの可視化"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            batch_indices = range(len(df))
    
            # 反応温度トレンド
            axes[0, 0].plot(batch_indices, df['reaction_temp'], marker='o', color='#11998e',
                            linewidth=1.5, markersize=4)
            axes[0, 0].axhline(y=80, color='green', linestyle='--', linewidth=2, label='目標値 (80℃)')
            axes[0, 0].axhline(y=82, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='警告限界 (±2℃)')
            axes[0, 0].axhline(y=78, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            axes[0, 0].set_xlabel('バッチ番号')
            axes[0, 0].set_ylabel('反応温度（℃）')
            axes[0, 0].set_title('反応温度トレンド', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
    
            # 反応時間トレンド
            axes[0, 1].plot(batch_indices, df['reaction_time'], marker='s', color='#38ef7d',
                            linewidth=1.5, markersize=4)
            axes[0, 1].axhline(y=120, color='green', linestyle='--', linewidth=2, label='目標値 (120分)')
            axes[0, 1].axhline(y=130, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='警告限界 (±10分)')
            axes[0, 1].axhline(y=110, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            axes[0, 1].set_xlabel('バッチ番号')
            axes[0, 1].set_ylabel('反応時間（分）')
            axes[0, 1].set_title('反応時間トレンド', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
    
            # 収率トレンド
            axes[1, 0].plot(batch_indices, df['yield'], marker='^', color='#4ecdc4',
                            linewidth=1.5, markersize=4)
            axes[1, 0].axhline(y=95, color='green', linestyle='--', linewidth=2, label='目標値 (95%)')
            axes[1, 0].axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='下限 (90%)')
            axes[1, 0].set_xlabel('バッチ番号')
            axes[1, 0].set_ylabel('収率（%）')
            axes[1, 0].set_title('収率トレンド', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
    
            # 純度トレンド
            axes[1, 1].plot(batch_indices, df['purity'], marker='D', color='#f38181',
                            linewidth=1.5, markersize=4)
            axes[1, 1].axhline(y=99.5, color='green', linestyle='--', linewidth=2, label='目標値 (99.5%)')
            axes[1, 1].axhline(y=99.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='規格下限 (99.0%)')
            axes[1, 1].set_xlabel('バッチ番号')
            axes[1, 1].set_ylabel('純度（%）')
            axes[1, 1].set_title('純度トレンド', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('batch_trend_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 実行例
    print("=" * 60)
    print("電子バッチ記録（EBR）管理システム")
    print("=" * 60)
    
    # EBRインスタンスの作成
    ebr = ElectronicBatchRecord(
        batch_id="B-2025-0042",
        product_name="アスピリン錠100mg",
        manufacturing_date="2025-10-27"
    )
    
    # 工程パラメータの記録
    ebr.add_process_parameter("反応", "温度", target=80, actual=80.5, unit="℃", tolerance=2)
    ebr.add_process_parameter("反応", "時間", target=120, actual=118, unit="分", tolerance=10)
    ebr.add_process_parameter("乾燥", "温度", target=60, actual=61, unit="℃", tolerance=3)
    
    # 逸脱の記録（例）
    ebr.add_deviation(
        description="反応温度が一時的に82℃まで上昇",
        severity="Minor",
        root_cause="温調システムのPIDパラメータ不適切",
        capa="PIDパラメータの再調整とアラーム設定"
    )
    
    # 電子署名
    ebr.add_signature("製造担当", "田中太郎")
    ebr.add_signature("品質保証", "鈴木花子")
    
    # エクスポート
    ebr.export_json("ebr_sample.json")
    
    print(f"\nバッチID: {ebr.batch_id}")
    print(f"製品名: {ebr.product_name}")
    print(f"記録された工程パラメータ数: {sum(len(params) for params in ebr.process_parameters.values())}")
    print(f"逸脱件数: {len(ebr.deviations)}")
    print(f"電子署名数: {len(ebr.signatures)}")
    
    # バッチトレンド分析
    print("\n" + "=" * 60)
    print("バッチトレンド分析")
    print("=" * 60)
    
    analyzer = BatchTrendAnalyzer()
    df_batches = analyzer.generate_batch_data(n_batches=50)
    
    print(f"\n分析対象バッチ数: {len(df_batches)}")
    print(f"期間: {df_batches['date'].min()} ~ {df_batches['date'].max()}")
    
    # 統計サマリー
    print(f"\n反応温度: 平均 {df_batches['reaction_temp'].mean():.2f}℃, 標準偏差 {df_batches['reaction_temp'].std():.2f}℃")
    print(f"反応時間: 平均 {df_batches['reaction_time'].mean():.1f}分, 標準偏差 {df_batches['reaction_time'].std():.1f}分")
    print(f"収率: 平均 {df_batches['yield'].mean():.2f}%, 標準偏差 {df_batches['yield'].std():.2f}%")
    print(f"純度: 平均 {df_batches['purity'].mean():.2f}%, 標準偏差 {df_batches['purity'].std():.2f}%")
    
    # トレンド可視化
    analyzer.plot_batch_trends(df_batches)
    

**実装のポイント:**

  * GMP準拠のEBRデータモデル（工程パラメータ、逸脱、電子署名）
  * 監査証跡の自動記録機能
  * 許容範囲チェックの自動化
  * バッチトレンド分析による異常検出の可視化
  * JSON形式でのデータエクスポート（相互運用性）

## 🔍 2.2 機械学習による異常バッチ検出

### 異常検出のアプローチ

バッチ製造における異常検出には、以下の機械学習手法が有効です：

  * **Isolation Forest** : 教師なし異常検出、多変量データに有効
  * **One-Class SVM** : 正常データのみで学習、境界決定
  * **Autoencoder** : 深層学習による再構成誤差ベースの検出
  * **Statistical Process Control** : Hotelling's T², MEWMA

### 💻 コード例2.2: Isolation Forestによる異常バッチ検出
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class AnomalyBatchDetector:
        """異常バッチ検出システム"""
    
        def __init__(self, contamination=0.1):
            """
            Args:
                contamination: 異常データの推定割合（0.1 = 10%）
            """
            self.contamination = contamination
            self.scaler = StandardScaler()
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            self.pca = PCA(n_components=2)
    
        def train(self, df, feature_columns):
            """
            モデルの訓練
    
            Args:
                df: バッチデータフレーム
                feature_columns: 特徴量カラムのリスト
            """
            X = df[feature_columns].values
            X_scaled = self.scaler.fit_transform(X)
    
            self.model.fit(X_scaled)
    
            # 異常スコアの計算
            anomaly_scores = self.model.score_samples(X_scaled)
            predictions = self.model.predict(X_scaled)
    
            # PCAで2次元に変換（可視化用）
            X_pca = self.pca.fit_transform(X_scaled)
    
            return anomaly_scores, predictions, X_pca
    
        def detect_anomalies(self, df, feature_columns, anomaly_scores, predictions):
            """
            異常バッチの特定
    
            Args:
                df: バッチデータフレーム
                feature_columns: 特徴量カラム
                anomaly_scores: 異常スコア
                predictions: 予測結果（-1: 異常, 1: 正常）
    
            Returns:
                異常バッチのデータフレーム
            """
            df_result = df.copy()
            df_result['anomaly_score'] = anomaly_scores
            df_result['is_anomaly'] = predictions == -1
    
            # 異常バッチの抽出
            anomalies = df_result[df_result['is_anomaly']].copy()
    
            # 異常の重要度ランキング（スコアが低いほど異常）
            anomalies = anomalies.sort_values('anomaly_score')
    
            return df_result, anomalies
    
        def plot_anomaly_detection(self, df_result, X_pca, feature_columns):
            """異常検出結果の可視化"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # PCA空間でのプロット
            normal_mask = ~df_result['is_anomaly']
            anomaly_mask = df_result['is_anomaly']
    
            axes[0, 0].scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                               c='green', s=30, alpha=0.6, label='正常バッチ')
            axes[0, 0].scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
                               c='red', s=100, alpha=0.8, marker='X', label='異常バッチ')
            axes[0, 0].set_xlabel(f'第1主成分 (寄与率: {self.pca.explained_variance_ratio_[0]:.1%})')
            axes[0, 0].set_ylabel(f'第2主成分 (寄与率: {self.pca.explained_variance_ratio_[1]:.1%})')
            axes[0, 0].set_title('PCA空間での異常検出', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
    
            # 異常スコア分布
            axes[0, 1].hist(df_result[normal_mask]['anomaly_score'], bins=30,
                            alpha=0.6, label='正常', color='green')
            axes[0, 1].hist(df_result[anomaly_mask]['anomaly_score'], bins=30,
                            alpha=0.6, label='異常', color='red')
            axes[0, 1].set_xlabel('異常スコア（小さいほど異常）')
            axes[0, 1].set_ylabel('頻度')
            axes[0, 1].set_title('異常スコア分布', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
    
            # 時系列での異常検出
            batch_indices = range(len(df_result))
            colors = ['red' if x else 'green' for x in df_result['is_anomaly']]
    
            axes[1, 0].scatter(batch_indices, df_result['anomaly_score'],
                               c=colors, s=50, alpha=0.7)
            axes[1, 0].axhline(y=df_result['anomaly_score'].quantile(0.1), color='orange',
                               linestyle='--', linewidth=2, label='異常閾値')
            axes[1, 0].set_xlabel('バッチ番号')
            axes[1, 0].set_ylabel('異常スコア')
            axes[1, 0].set_title('時系列異常検出', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
    
            # 特徴量別の異常バッチ分布
            if len(feature_columns) >= 2:
                feat1, feat2 = feature_columns[0], feature_columns[1]
    
                axes[1, 1].scatter(df_result[normal_mask][feat1], df_result[normal_mask][feat2],
                                   c='green', s=30, alpha=0.6, label='正常バッチ')
                axes[1, 1].scatter(df_result[anomaly_mask][feat1], df_result[anomaly_mask][feat2],
                                   c='red', s=100, alpha=0.8, marker='X', label='異常バッチ')
                axes[1, 1].set_xlabel(feat1)
                axes[1, 1].set_ylabel(feat2)
                axes[1, 1].set_title(f'{feat1} vs {feat2}', fontsize=12, fontweight='bold')
                axes[1, 1].legend()
                axes[1, 1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('anomaly_batch_detection.png', dpi=300, bbox_inches='tight')
            plt.show()
    
        def generate_anomaly_report(self, anomalies, feature_columns):
            """異常レポートの生成"""
            print("\n" + "=" * 60)
            print("異常バッチ検出レポート")
            print("=" * 60)
    
            for idx, row in anomalies.iterrows():
                print(f"\n🚨 バッチID: {row['batch_id']}")
                print(f"   製造日: {row['date']}")
                print(f"   異常スコア: {row['anomaly_score']:.4f}")
                print(f"   工程パラメータ:")
    
                for feat in feature_columns:
                    print(f"     - {feat}: {row[feat]:.2f}")
    
    # 実行例
    print("=" * 60)
    print("異常バッチ検出システム（Isolation Forest）")
    print("=" * 60)
    
    # バッチデータの生成（前のコード例から再利用）
    analyzer = BatchTrendAnalyzer()
    df_batches = analyzer.generate_batch_data(n_batches=50)
    
    # 特徴量の定義
    feature_columns = ['reaction_temp', 'reaction_time', 'yield', 'purity']
    
    # 異常検出モデルの訓練
    detector = AnomalyBatchDetector(contamination=0.15)  # 15%が異常と仮定
    anomaly_scores, predictions, X_pca = detector.train(df_batches, feature_columns)
    
    # 異常バッチの検出
    df_result, anomalies = detector.detect_anomalies(df_batches, feature_columns, anomaly_scores, predictions)
    
    print(f"\n総バッチ数: {len(df_batches)}")
    print(f"検出された異常バッチ数: {len(anomalies)} ({len(anomalies)/len(df_batches)*100:.1f}%)")
    
    # 可視化
    detector.plot_anomaly_detection(df_result, X_pca, feature_columns)
    
    # レポート生成
    detector.generate_anomaly_report(anomalies.head(5), feature_columns)
    

**実装のポイント:**

  * 教師なし学習による未知の異常パターン検出
  * 多変量データの統合的な評価（温度、時間、収率、純度）
  * PCAによる高次元データの可視化
  * 異常スコアによる優先度付け
  * 自動レポート生成機能

## 📚 まとめ

本章では、電子バッチ記録の解析と逸脱管理について学びました。

### 主要なポイント

  * GMP準拠のEBRデータモデルと監査証跡の実装
  * バッチトレンド分析による工程変動の可視化
  * 機械学習（Isolation Forest）による異常バッチの自動検出
  * 多変量データ解析による包括的な品質評価
  * 自動レポート生成による作業効率化

**🎯 次章予告**  
第3章では、プロセス分析技術（PAT）とリアルタイム品質管理について学びます。 NIR/Raman分光分析、多変量統計的プロセス管理（MSPC）、リアルタイムリリース試験（RTRT）など、 より高度なプロセス管理技術を習得します。 

[← 第1章: GMP統計的品質管理](<chapter-1.html>) [第3章: PATとリアルタイム品質管理 →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
