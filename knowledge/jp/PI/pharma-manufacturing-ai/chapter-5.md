---
title: 第5章 規制対応とCSV実装戦略
chapter_title: 第5章 規制対応とCSV実装戦略
subtitle: Regulatory Compliance and Computer System Validation Strategy
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/pharma-manufacturing-ai/chapter-5.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Pharma Manufacturing Ai](<../../PI/pharma-manufacturing-ai/index.html>)›Chapter 5

[← シリーズ目次に戻る](<index.html>)

## 📖 本章の概要

医薬品製造におけるAIシステムの導入には、厳格な規制要件への適合が必須です。 本章では、Computer System Validation（CSV）、21 CFR Part 11、EU Annex 11などの 規制要件を満たすAIシステムの実装戦略を学びます。監査証跡、電子署名、 データインテグリティの確保方法を実践的に習得します。 

### 🎯 学習目標

  * CSV（Computer System Validation）の基本概念とライフサイクルアプローチ
  * 21 CFR Part 11（電子記録・電子署名）の要件と実装
  * ALCOA+原則に基づくデータインテグリティの確保
  * 監査証跡（Audit Trail）の設計と実装
  * リスクベースドアプローチによる適格性評価
  * AIモデルのバリデーション戦略
  * 変更管理とバージョン管理の実装

## 📋 5.1 CSV（Computer System Validation）の基礎

### CSVライフサイクルアプローチ

GAMP 5（Good Automated Manufacturing Practice）に基づくCSVプロセス：

  1. **計画（Planning）** : バリデーションマスタープラン（VMP）
  2. **仕様（Specification）** : URS、FS、DS作成
  3. **構成・開発（Configuration/Development）** : システム構築
  4. **検証（Verification）** : IQ、OQ、PQ実施
  5. **報告・リリース（Reporting/Release）** : バリデーション報告書
  6. **運用・保守（Operation/Maintenance）** : 変更管理、定期レビュー
  7. **廃棄（Retirement）** : データ移行、アーカイブ

**🏭 GAMP 5カテゴリ分類**  
**カテゴリ1** : インフラストラクチャソフトウェア（OS）  
**カテゴリ3** : 非設定製品（標準パッケージ）  
**カテゴリ4** : 設定製品（カスタマイズ可能）  
**カテゴリ5** : カスタムソフトウェア（独自開発）  
※AIシステムは通常カテゴリ4または5 

### 💻 コード例5.1: 監査証跡システムの実装
    
    
    import json
    import hashlib
    from datetime import datetime
    from typing import Dict, List, Any
    import warnings
    warnings.filterwarnings('ignore')
    
    class AuditTrailSystem:
        """GMP準拠の監査証跡システム"""
    
        def __init__(self, system_name: str):
            """
            Args:
                system_name: システム名
            """
            self.system_name = system_name
            self.audit_records: List[Dict] = []
            self.sequence_number = 0
    
        def log_event(self, event_type: str, user: str, action: str,
                      record_type: str, record_id: str, old_value: Any = None,
                      new_value: Any = None, reason: str = None) -> Dict:
            """
            イベントの記録（21 CFR Part 11準拠）
    
            Args:
                event_type: イベント種別（CREATE, READ, UPDATE, DELETE, SIGN）
                user: ユーザーID
                action: アクション説明
                record_type: レコード種別（BATCH_RECORD, DEVIATION, SOP など）
                record_id: レコードID
                old_value: 変更前の値
                new_value: 変更後の値
                reason: 変更理由
    
            Returns:
                監査証跡レコード
            """
            self.sequence_number += 1
    
            # タイムスタンプ（UTC）
            timestamp = datetime.utcnow().isoformat() + 'Z'
    
            # 監査証跡レコード
            audit_record = {
                'sequence_number': self.sequence_number,
                'timestamp': timestamp,
                'system_name': self.system_name,
                'event_type': event_type,
                'user_id': user,
                'action': action,
                'record_type': record_type,
                'record_id': record_id,
                'old_value': old_value,
                'new_value': new_value,
                'reason': reason,
                'ip_address': '192.168.1.100',  # 実際はリクエストから取得
                'session_id': 'SESSION-' + hashlib.md5(user.encode()).hexdigest()[:8]
            }
    
            # ハッシュ値の計算（改ざん検出用）
            audit_record['hash'] = self._calculate_hash(audit_record)
    
            # 前レコードとのチェーン（ブロックチェーン的アプローチ）
            if len(self.audit_records) > 0:
                audit_record['previous_hash'] = self.audit_records[-1]['hash']
            else:
                audit_record['previous_hash'] = None
    
            self.audit_records.append(audit_record)
    
            return audit_record
    
        def _calculate_hash(self, record: Dict) -> str:
            """レコードのハッシュ値計算（SHA-256）"""
            # ハッシュ計算対象データ
            data_to_hash = {
                'sequence_number': record['sequence_number'],
                'timestamp': record['timestamp'],
                'user_id': record['user_id'],
                'action': record['action'],
                'record_id': record['record_id'],
                'old_value': str(record.get('old_value')),
                'new_value': str(record.get('new_value'))
            }
    
            data_string = json.dumps(data_to_hash, sort_keys=True)
            return hashlib.sha256(data_string.encode()).hexdigest()
    
        def verify_integrity(self) -> tuple:
            """
            監査証跡の完全性検証
    
            Returns:
                (検証結果, エラーリスト)
            """
            errors = []
    
            for i, record in enumerate(self.audit_records):
                # ハッシュ値の再計算と検証
                calculated_hash = self._calculate_hash(record)
                if calculated_hash != record['hash']:
                    errors.append(f"シーケンス {record['sequence_number']}: ハッシュ不一致")
    
                # チェーン検証
                if i > 0:
                    expected_prev_hash = self.audit_records[i-1]['hash']
                    if record['previous_hash'] != expected_prev_hash:
                        errors.append(f"シーケンス {record['sequence_number']}: チェーン断絶")
    
            is_valid = len(errors) == 0
            return is_valid, errors
    
        def search_audit_trail(self, user: str = None, record_id: str = None,
                               event_type: str = None, start_date: str = None,
                               end_date: str = None) -> List[Dict]:
            """
            監査証跡の検索
    
            Args:
                user: ユーザーID
                record_id: レコードID
                event_type: イベント種別
                start_date: 開始日時（ISO8601形式）
                end_date: 終了日時（ISO8601形式）
    
            Returns:
                検索結果のリスト
            """
            results = self.audit_records.copy()
    
            # フィルタリング
            if user:
                results = [r for r in results if r['user_id'] == user]
    
            if record_id:
                results = [r for r in results if r['record_id'] == record_id]
    
            if event_type:
                results = [r for r in results if r['event_type'] == event_type]
    
            if start_date:
                results = [r for r in results if r['timestamp'] >= start_date]
    
            if end_date:
                results = [r for r in results if r['timestamp'] <= end_date]
    
            return results
    
        def export_audit_trail(self, filename: str, format: str = 'json'):
            """
            監査証跡のエクスポート
    
            Args:
                filename: 出力ファイル名
                format: 出力形式（json, csv）
            """
            if format == 'json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        'system_name': self.system_name,
                        'export_timestamp': datetime.utcnow().isoformat() + 'Z',
                        'total_records': len(self.audit_records),
                        'audit_records': self.audit_records
                    }, f, ensure_ascii=False, indent=2)
    
            elif format == 'csv':
                import csv
                with open(filename, 'w', encoding='utf-8', newline='') as f:
                    if len(self.audit_records) > 0:
                        fieldnames = list(self.audit_records[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.audit_records)
    
            print(f"監査証跡を {filename} にエクスポートしました")
    
        def generate_audit_report(self):
            """監査証跡レポートの生成"""
            print("=" * 60)
            print("監査証跡レポート")
            print("=" * 60)
            print(f"システム名: {self.system_name}")
            print(f"総レコード数: {len(self.audit_records)}")
    
            if len(self.audit_records) > 0:
                print(f"最初の記録: {self.audit_records[0]['timestamp']}")
                print(f"最後の記録: {self.audit_records[-1]['timestamp']}")
    
                # イベント種別ごとの集計
                event_counts = {}
                for record in self.audit_records:
                    event_type = record['event_type']
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
                print(f"\nイベント種別集計:")
                for event_type, count in sorted(event_counts.items()):
                    print(f"  {event_type}: {count} 件")
    
                # ユーザーごとの集計
                user_counts = {}
                for record in self.audit_records:
                    user = record['user_id']
                    user_counts[user] = user_counts.get(user, 0) + 1
    
                print(f"\nユーザー別活動:")
                for user, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {user}: {count} 件")
    
            # 完全性検証
            is_valid, errors = self.verify_integrity()
            print(f"\n監査証跡の完全性: {'✅ 検証OK' if is_valid else '❌ エラー検出'}")
            if errors:
                print(f"エラー詳細:")
                for error in errors:
                    print(f"  - {error}")
    
    
    class ElectronicSignatureSystem:
        """電子署名システム（21 CFR Part 11準拠）"""
    
        def __init__(self, audit_trail: AuditTrailSystem):
            """
            Args:
                audit_trail: 監査証跡システム
            """
            self.audit_trail = audit_trail
            self.signatures: Dict[str, List[Dict]] = {}
            self.user_credentials = {
                'user001': hashlib.sha256('password123'.encode()).hexdigest(),
                'user002': hashlib.sha256('securepass456'.encode()).hexdigest()
            }
    
        def authenticate_user(self, user_id: str, password: str) -> bool:
            """ユーザー認証"""
            if user_id not in self.user_credentials:
                return False
    
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return password_hash == self.user_credentials[user_id]
    
        def sign_record(self, user_id: str, password: str, record_id: str,
                        record_type: str, meaning: str, reason: str = None) -> Dict:
            """
            電子署名の実行
    
            Args:
                user_id: ユーザーID
                password: パスワード
                record_id: 署名対象レコードID
                record_type: レコード種別
                meaning: 署名の意味（Reviewed, Approved, など）
                reason: 署名理由
    
            Returns:
                署名レコード
            """
            # ユーザー認証
            if not self.authenticate_user(user_id, password):
                raise ValueError("認証失敗: ユーザーIDまたはパスワードが無効です")
    
            # 署名レコードの作成
            signature = {
                'signature_id': f"SIG-{len(self.signatures.get(record_id, [])) + 1:04d}",
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'user_id': user_id,
                'record_id': record_id,
                'record_type': record_type,
                'meaning': meaning,
                'reason': reason,
                'signature_hash': self._create_signature_hash(user_id, record_id, meaning)
            }
    
            # 署名の保存
            if record_id not in self.signatures:
                self.signatures[record_id] = []
            self.signatures[record_id].append(signature)
    
            # 監査証跡への記録
            self.audit_trail.log_event(
                event_type='SIGN',
                user=user_id,
                action=f"電子署名実行: {meaning}",
                record_type=record_type,
                record_id=record_id,
                new_value=meaning,
                reason=reason
            )
    
            return signature
    
        def _create_signature_hash(self, user_id: str, record_id: str, meaning: str) -> str:
            """署名ハッシュの生成"""
            data = f"{user_id}:{record_id}:{meaning}:{datetime.utcnow().isoformat()}"
            return hashlib.sha256(data.encode()).hexdigest()
    
        def verify_signatures(self, record_id: str) -> List[Dict]:
            """特定レコードの署名検証"""
            return self.signatures.get(record_id, [])
    
    
    # 実行例
    print("=" * 60)
    print("監査証跡・電子署名システム（21 CFR Part 11準拠）")
    print("=" * 60)
    
    # 監査証跡システムの初期化
    audit_system = AuditTrailSystem(system_name="Manufacturing Execution System")
    
    # バッチレコードの作成
    audit_system.log_event(
        event_type='CREATE',
        user='user001',
        action='バッチレコード作成',
        record_type='BATCH_RECORD',
        record_id='BATCH-2025-0001',
        new_value={'batch_id': 'BATCH-2025-0001', 'product': 'アスピリン錠', 'quantity': 10000}
    )
    
    # バッチレコードの更新
    audit_system.log_event(
        event_type='UPDATE',
        user='user001',
        action='反応温度記録',
        record_type='BATCH_RECORD',
        record_id='BATCH-2025-0001',
        old_value={'reaction_temp': None},
        new_value={'reaction_temp': 80.5},
        reason='プロセスパラメータ入力'
    )
    
    # 逸脱記録の作成
    audit_system.log_event(
        event_type='CREATE',
        user='user002',
        action='逸脱記録作成',
        record_type='DEVIATION',
        record_id='DEV-2025-001',
        new_value={'description': '反応温度が一時的に上限超過', 'severity': 'Minor'},
        reason='温度スパイク検出'
    )
    
    # 電子署名システムの初期化
    esign_system = ElectronicSignatureSystem(audit_system)
    
    # バッチレコードへの署名
    try:
        signature1 = esign_system.sign_record(
            user_id='user001',
            password='password123',
            record_id='BATCH-2025-0001',
            record_type='BATCH_RECORD',
            meaning='Reviewed',
            reason='バッチ記録のレビュー完了'
        )
        print(f"\n✅ 電子署名成功: {signature1['signature_id']}")
    
        signature2 = esign_system.sign_record(
            user_id='user002',
            password='securepass456',
            record_id='BATCH-2025-0001',
            record_type='BATCH_RECORD',
            meaning='Approved',
            reason='バッチリリース承認'
        )
        print(f"✅ 電子署名成功: {signature2['signature_id']}")
    
    except ValueError as e:
        print(f"❌ エラー: {e}")
    
    # 監査証跡レポートの生成
    audit_system.generate_audit_report()
    
    # 監査証跡のエクスポート
    audit_system.export_audit_trail('audit_trail_part11.json', format='json')
    
    # 特定レコードの署名検証
    print(f"\n" + "=" * 60)
    print("電子署名検証")
    print("=" * 60)
    signatures = esign_system.verify_signatures('BATCH-2025-0001')
    for sig in signatures:
        print(f"署名ID: {sig['signature_id']}")
        print(f"  署名者: {sig['user_id']}")
        print(f"  日時: {sig['timestamp']}")
        print(f"  意味: {sig['meaning']}")
        print(f"  理由: {sig['reason']}")
        print()
    

**実装のポイント:**

  * 21 CFR Part 11準拠の監査証跡機能（タイムスタンプ、ユーザーID、変更履歴）
  * SHA-256ハッシュによる改ざん検出機能
  * ブロックチェーン的アプローチによるレコード連鎖
  * 電子署名の2要素認証（ユーザーID + パスワード）
  * 署名の意味（Reviewed, Approved）の明確化
  * 完全性検証機能とエクスポート機能

## 🔒 5.2 データインテグリティとALCOA+原則

### ALCOA+原則

データインテグリティの基本原則（FDA/MHRA要件）：

  * **A** ttributable（帰属性）: データの作成者・変更者が特定可能
  * **L** egible（判読性）: データが読みやすく理解可能
  * **C** ontemporaneous（同時性）: データが作成時点で記録される
  * **O** riginal（原本性）: オリジナルデータまたは真正なコピー
  * **A** ccurate（正確性）: データが正確で誤りがない
  * **+** Complete（完全性）: すべての関連データが含まれる
  * **+** Consistent（一貫性）: データに矛盾がない
  * **+** Enduring（耐久性）: データが長期間保存される
  * **+** Available（利用可能性）: 必要時にアクセス可能

**⚠️ データインテグリティ違反の事例**  
・監査証跡の無効化や削除  
・電子記録の無許可変更  
・バックデート（日時の遡及変更）  
・データの選択的報告  
・複数ユーザーでの単一アカウント共有 

### 💻 コード例5.2: データインテグリティチェッカー
    
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import json
    import warnings
    warnings.filterwarnings('ignore')
    
    class DataIntegrityChecker:
        """データインテグリティ検証システム（ALCOA+準拠）"""
    
        def __init__(self):
            self.violations = []
    
        def check_attributable(self, df: pd.DataFrame, required_columns: list) -> dict:
            """
            帰属性チェック: 作成者・変更者情報の存在確認
    
            Args:
                df: 検証対象データフレーム
                required_columns: 必須カラム（例: ['created_by', 'modified_by']）
    
            Returns:
                検証結果
            """
            missing_columns = [col for col in required_columns if col not in df.columns]
    
            if missing_columns:
                violation = {
                    'principle': 'Attributable',
                    'severity': 'Critical',
                    'description': f"必須カラムが欠落: {missing_columns}"
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            # NULL値のチェック
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                violation = {
                    'principle': 'Attributable',
                    'severity': 'Critical',
                    'description': f"NULL値を検出: {null_counts[null_counts > 0].to_dict()}"
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_contemporaneous(self, df: pd.DataFrame, timestamp_col: str,
                                  event_col: str, max_delay_minutes: int = 5) -> dict:
            """
            同時性チェック: イベント発生と記録時刻の乖離確認
    
            Args:
                df: 検証対象データフレーム
                timestamp_col: タイムスタンプカラム
                event_col: イベント発生時刻カラム
                max_delay_minutes: 許容遅延時間（分）
    
            Returns:
                検証結果
            """
            if timestamp_col not in df.columns or event_col not in df.columns:
                violation = {
                    'principle': 'Contemporaneous',
                    'severity': 'Critical',
                    'description': f"タイムスタンプカラムが存在しません"
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            # タイムスタンプの変換
            df_temp = df.copy()
            df_temp[timestamp_col] = pd.to_datetime(df_temp[timestamp_col])
            df_temp[event_col] = pd.to_datetime(df_temp[event_col])
    
            # 遅延時間の計算
            df_temp['delay_minutes'] = (df_temp[timestamp_col] - df_temp[event_col]).dt.total_seconds() / 60
    
            # 許容範囲外のレコード
            delayed_records = df_temp[df_temp['delay_minutes'] > max_delay_minutes]
    
            if len(delayed_records) > 0:
                violation = {
                    'principle': 'Contemporaneous',
                    'severity': 'Major',
                    'description': f"{len(delayed_records)}件のレコードが遅延記録（>{max_delay_minutes}分）",
                    'details': delayed_records[['delay_minutes']].to_dict()
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_complete(self, df: pd.DataFrame, mandatory_columns: list) -> dict:
            """
            完全性チェック: 必須データの欠損確認
    
            Args:
                df: 検証対象データフレーム
                mandatory_columns: 必須カラムリスト
    
            Returns:
                検証結果
            """
            missing_data = {}
    
            for col in mandatory_columns:
                if col not in df.columns:
                    missing_data[col] = 'カラム自体が存在しない'
                else:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        missing_data[col] = f"{null_count}件の欠損値"
    
            if missing_data:
                violation = {
                    'principle': 'Complete',
                    'severity': 'Critical',
                    'description': "必須データの欠損を検出",
                    'details': missing_data
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_consistent(self, df: pd.DataFrame, validation_rules: dict) -> dict:
            """
            一貫性チェック: ビジネスルールの検証
    
            Args:
                df: 検証対象データフレーム
                validation_rules: 検証ルール辞書
    
            Returns:
                検証結果
            """
            inconsistencies = []
    
            for rule_name, rule_func in validation_rules.items():
                violations_found = rule_func(df)
                if violations_found:
                    inconsistencies.append({
                        'rule': rule_name,
                        'violations': violations_found
                    })
    
            if inconsistencies:
                violation = {
                    'principle': 'Consistent',
                    'severity': 'Major',
                    'description': "データの一貫性違反を検出",
                    'details': inconsistencies
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_accurate(self, df: pd.DataFrame, value_ranges: dict) -> dict:
            """
            正確性チェック: データ値の範囲検証
    
            Args:
                df: 検証対象データフレーム
                value_ranges: 値範囲辞書 {'column': (min, max)}
    
            Returns:
                検証結果
            """
            out_of_range = {}
    
            for col, (min_val, max_val) in value_ranges.items():
                if col in df.columns:
                    violations = df[(df[col] < min_val) | (df[col] > max_val)]
                    if len(violations) > 0:
                        out_of_range[col] = {
                            'count': len(violations),
                            'range': f"{min_val}-{max_val}",
                            'invalid_values': violations[col].tolist()[:10]  # 最初の10件
                        }
    
            if out_of_range:
                violation = {
                    'principle': 'Accurate',
                    'severity': 'Critical',
                    'description': "範囲外の値を検出",
                    'details': out_of_range
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def generate_integrity_report(self) -> dict:
            """データインテグリティレポートの生成"""
            total_checks = len(self.violations) if self.violations else 0
    
            critical_violations = [v for v in self.violations if v['severity'] == 'Critical']
            major_violations = [v for v in self.violations if v['severity'] == 'Major']
            minor_violations = [v for v in self.violations if v['severity'] == 'Minor']
    
            report = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'total_violations': len(self.violations),
                'critical': len(critical_violations),
                'major': len(major_violations),
                'minor': len(minor_violations),
                'violations': self.violations,
                'passed': len(self.violations) == 0
            }
    
            return report
    
        def print_report(self):
            """レポートの出力"""
            report = self.generate_integrity_report()
    
            print("=" * 60)
            print("データインテグリティ検証レポート（ALCOA+）")
            print("=" * 60)
            print(f"検証日時: {report['timestamp']}")
            print(f"\n総違反件数: {report['total_violations']}")
            print(f"  Critical: {report['critical']}")
            print(f"  Major: {report['major']}")
            print(f"  Minor: {report['minor']}")
    
            if report['passed']:
                print(f"\n✅ データインテグリティ検証: 合格")
            else:
                print(f"\n❌ データインテグリティ検証: 不合格")
    
                print(f"\n違反詳細:")
                for i, violation in enumerate(self.violations, 1):
                    print(f"\n{i}. [{violation['severity']}] {violation['principle']}")
                    print(f"   {violation['description']}")
                    if 'details' in violation:
                        print(f"   詳細: {json.dumps(violation['details'], indent=2, ensure_ascii=False)}")
    
    
    # 実行例
    print("=" * 60)
    print("データインテグリティチェッカー（ALCOA+準拠）")
    print("=" * 60)
    
    # サンプルデータの生成（バッチ記録を想定）
    np.random.seed(42)
    n_records = 50
    
    df_batch = pd.DataFrame({
        'batch_id': [f'BATCH-{i+1:04d}' for i in range(n_records)],
        'created_by': ['user001'] * 30 + ['user002'] * 15 + [None] * 5,  # 一部NULL
        'created_at': [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n_records)],
        'event_time': [datetime(2025, 1, 1) + timedelta(hours=i, minutes=np.random.randint(0, 10)) for i in range(n_records)],
        'reaction_temp': np.random.normal(80, 5, n_records),
        'reaction_time': np.random.normal(120, 10, n_records),
        'yield': np.random.normal(95, 3, n_records),
        'purity': np.random.normal(99.5, 0.5, n_records)
    })
    
    # 意図的に異常値を追加
    df_batch.loc[10, 'reaction_temp'] = 150  # 範囲外
    df_batch.loc[20, 'yield'] = 110  # 100%超過
    df_batch.loc[30, 'created_at'] = df_batch.loc[30, 'event_time'] + timedelta(hours=1)  # 遅延記録
    
    # データインテグリティチェッカーの初期化
    checker = DataIntegrityChecker()
    
    # 帰属性チェック
    print("\n【帰属性チェック】")
    result = checker.check_attributable(df_batch, required_columns=['created_by', 'created_at'])
    print(f"結果: {'✅ 合格' if result['passed'] else '❌ 不合格'}")
    
    # 同時性チェック
    print("\n【同時性チェック】")
    result = checker.check_contemporaneous(df_batch, timestamp_col='created_at',
                                           event_col='event_time', max_delay_minutes=5)
    print(f"結果: {'✅ 合格' if result['passed'] else '❌ 不合格'}")
    
    # 完全性チェック
    print("\n【完全性チェック】")
    result = checker.check_complete(df_batch, mandatory_columns=['batch_id', 'created_by',
                                                                  'reaction_temp', 'yield'])
    print(f"結果: {'✅ 合格' if result['passed'] else '❌ 不合格'}")
    
    # 正確性チェック
    print("\n【正確性チェック】")
    value_ranges = {
        'reaction_temp': (70, 90),
        'reaction_time': (100, 140),
        'yield': (0, 100),
        'purity': (95, 100)
    }
    result = checker.check_accurate(df_batch, value_ranges)
    print(f"結果: {'✅ 合格' if result['passed'] else '❌ 不合格'}")
    
    # 総合レポート
    checker.print_report()
    

**実装のポイント:**

  * ALCOA+原則に基づく包括的なデータインテグリティ検証
  * 帰属性、同時性、完全性、一貫性、正確性の自動チェック
  * 違反の重要度分類（Critical、Major、Minor）
  * 詳細なレポート生成と監査対応
  * リアルタイムデータ検証の実装例

## 📚 まとめ

本章および本シリーズでは、医薬品製造プロセスへのAI応用を包括的に学びました。

### 第5章の主要なポイント

  * CSV（Computer System Validation）のライフサイクルアプローチ
  * 21 CFR Part 11準拠の監査証跡と電子署名の実装
  * ALCOA+原則に基づくデータインテグリティの確保
  * 改ざん検出機能とブロックチェーン的アプローチ
  * 規制当局の査察に対応可能なシステム設計

### シリーズ全体の振り返り

  * **第1章** : GMP準拠の統計的品質管理（SPC、工程能力）
  * **第2章** : 電子バッチ記録解析と逸脱管理
  * **第3章** : PATとリアルタイム品質管理（NIR、MSPC）
  * **第4章** : 連続生産とQbD実装（DoE、デザインスペース）
  * **第5章** : 規制対応とCSV実装戦略

**🎓 シリーズ完了**  
本シリーズ「医薬品製造プロセスへのAI応用」を完了しました。 GMP準拠の品質管理から規制対応まで、医薬品製造現場で即活用できる実践知識を習得しました。  
  
継続学習のリソース:  
・プロセス・インフォマティクス道場の他シリーズ（化学プラント、半導体製造など）  
・FDA PATガイダンス、ICH Q8-Q11ガイドライン  
・ISPE（国際製薬工学会）のGAMP 5ガイド  
・データインテグリティに関するFDA/MHRA規制文書 

← 第4章: 連続生産とQbD実装（準備中） [シリーズ目次へ →](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
