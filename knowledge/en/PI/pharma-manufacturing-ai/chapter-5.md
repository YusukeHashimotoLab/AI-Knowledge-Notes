---
title: Chapter 5 Regulatory Compliance and CSV Implementation Strategy
chapter_title: Chapter 5 Regulatory Compliance and CSV Implementation Strategy
subtitle: Regulatory Compliance and Computer System Validation Strategy
---

[AI Terakoya Top](<../index.html>)›[Process Informatics](<../../index.html>)›[Pharma Manufacturing AI](<../../PI/pharma-manufacturing-ai/index.html>)›Chapter 5

🌐 EN | [🇯🇵 JP](<../../../jp/PI/pharma-manufacturing-ai/chapter-5.html>) | Last sync: 2025-11-16

[← Back to Series Index](<index.html>)

## 📖 Chapter Overview

The implementation of AI systems in pharmaceutical manufacturing requires strict compliance with regulatory requirements. In this chapter, we will learn implementation strategies for AI systems that meet regulatory requirements such as Computer System Validation (CSV), 21 CFR Part 11, and EU Annex 11. We will acquire practical skills in ensuring audit trails, electronic signatures, and data integrity. 

### 🎯 Learning Objectives

  * CSV (Computer System Validation) basic concepts and lifecycle approach
  * 21 CFR Part 11 (Electronic Records and Electronic Signatures) requirements and implementation
  * Ensuring data integrity based on ALCOA+ principles
  * Design and implementation of Audit Trail
  * Qualification assessment through risk-based approach
  * AI model validation strategy
  * Implementation of change control and version management

## 📋 5.1 Fundamentals of CSV (Computer System Validation)

### CSV Lifecycle Approach

CSV process based on GAMP 5 (Good Automated Manufacturing Practice):

  1. **Planning** : Validation Master Plan (VMP)
  2. **Specification** : URS, FS, DS creation
  3. **Configuration/Development** : System construction
  4. **Verification** : IQ, OQ, PQ execution
  5. **Reporting/Release** : Validation report
  6. **Operation/Maintenance** : Change control, periodic review
  7. **Retirement** : Data migration, archiving

**🏭 GAMP 5 Category Classification**  
**Category 1** : Infrastructure Software (OS)  
**Category 3** : Non-configured Products (Standard Packages)  
**Category 4** : Configured Products (Customizable)  
**Category 5** : Custom Software (In-house Development)  
*AI systems are typically Category 4 or 5 

### 💻 Code Example 5.1: Audit Trail System Implementation
    
    
    import json
    import hashlib
    from datetime import datetime
    from typing import Dict, List, Any
    import warnings
    warnings.filterwarnings('ignore')
    
    class AuditTrailSystem:
        """GMP-compliant audit trail system"""
    
        def __init__(self, system_name: str):
            """
            Args:
                system_name: System name
            """
            self.system_name = system_name
            self.audit_records: List[Dict] = []
            self.sequence_number = 0
    
        def log_event(self, event_type: str, user: str, action: str,
                      record_type: str, record_id: str, old_value: Any = None,
                      new_value: Any = None, reason: str = None) -> Dict:
            """
            Event logging (21 CFR Part 11 compliant)
    
            Args:
                event_type: Event type (CREATE, READ, UPDATE, DELETE, SIGN)
                user: User ID
                action: Action description
                record_type: Record type (BATCH_RECORD, DEVIATION, SOP, etc.)
                record_id: Record ID
                old_value: Old value before change
                new_value: New value after change
                reason: Reason for change
    
            Returns:
                Audit trail record
            """
            self.sequence_number += 1
    
            # Timestamp (UTC)
            timestamp = datetime.utcnow().isoformat() + 'Z'
    
            # Audit trail record
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
                'ip_address': '192.168.1.100',  # Actually retrieved from request
                'session_id': 'SESSION-' + hashlib.md5(user.encode()).hexdigest()[:8]
            }
    
            # Hash calculation (for tamper detection)
            audit_record['hash'] = self._calculate_hash(audit_record)
    
            # Chain to previous record (blockchain-like approach)
            if len(self.audit_records) > 0:
                audit_record['previous_hash'] = self.audit_records[-1]['hash']
            else:
                audit_record['previous_hash'] = None
    
            self.audit_records.append(audit_record)
    
            return audit_record
    
        def _calculate_hash(self, record: Dict) -> str:
            """Calculate record hash value (SHA-256)"""
            # Data to hash
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
            Verify audit trail integrity
    
            Returns:
                (verification result, error list)
            """
            errors = []
    
            for i, record in enumerate(self.audit_records):
                # Recalculate and verify hash value
                calculated_hash = self._calculate_hash(record)
                if calculated_hash != record['hash']:
                    errors.append(f"Sequence {record['sequence_number']}: Hash mismatch")
    
                # Chain verification
                if i > 0:
                    expected_prev_hash = self.audit_records[i-1]['hash']
                    if record['previous_hash'] != expected_prev_hash:
                        errors.append(f"Sequence {record['sequence_number']}: Chain broken")
    
            is_valid = len(errors) == 0
            return is_valid, errors
    
        def search_audit_trail(self, user: str = None, record_id: str = None,
                               event_type: str = None, start_date: str = None,
                               end_date: str = None) -> List[Dict]:
            """
            Search audit trail
    
            Args:
                user: User ID
                record_id: Record ID
                event_type: Event type
                start_date: Start date/time (ISO8601 format)
                end_date: End date/time (ISO8601 format)
    
            Returns:
                List of search results
            """
            results = self.audit_records.copy()
    
            # Filtering
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
            Export audit trail
    
            Args:
                filename: Output filename
                format: Output format (json, csv)
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
    
            print(f"Audit trail exported to {filename}")
    
        def generate_audit_report(self):
            """Generate audit trail report"""
            print("=" * 60)
            print("Audit Trail Report")
            print("=" * 60)
            print(f"System Name: {self.system_name}")
            print(f"Total Records: {len(self.audit_records)}")
    
            if len(self.audit_records) > 0:
                print(f"First Record: {self.audit_records[0]['timestamp']}")
                print(f"Last Record: {self.audit_records[-1]['timestamp']}")
    
                # Event type aggregation
                event_counts = {}
                for record in self.audit_records:
                    event_type = record['event_type']
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
                print(f"\nEvent Type Summary:")
                for event_type, count in sorted(event_counts.items()):
                    print(f"  {event_type}: {count} records")
    
                # User activity aggregation
                user_counts = {}
                for record in self.audit_records:
                    user = record['user_id']
                    user_counts[user] = user_counts.get(user, 0) + 1
    
                print(f"\nUser Activity:")
                for user, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {user}: {count} records")
    
            # Integrity verification
            is_valid, errors = self.verify_integrity()
            print(f"\nAudit Trail Integrity: {'✅ Verified OK' if is_valid else '❌ Errors Detected'}")
            if errors:
                print(f"Error Details:")
                for error in errors:
                    print(f"  - {error}")
    
    
    class ElectronicSignatureSystem:
        """Electronic signature system (21 CFR Part 11 compliant)"""
    
        def __init__(self, audit_trail: AuditTrailSystem):
            """
            Args:
                audit_trail: Audit trail system
            """
            self.audit_trail = audit_trail
            self.signatures: Dict[str, List[Dict]] = {}
            self.user_credentials = {
                'user001': hashlib.sha256('password123'.encode()).hexdigest(),
                'user002': hashlib.sha256('securepass456'.encode()).hexdigest()
            }
    
        def authenticate_user(self, user_id: str, password: str) -> bool:
            """User authentication"""
            if user_id not in self.user_credentials:
                return False
    
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return password_hash == self.user_credentials[user_id]
    
        def sign_record(self, user_id: str, password: str, record_id: str,
                        record_type: str, meaning: str, reason: str = None) -> Dict:
            """
            Execute electronic signature
    
            Args:
                user_id: User ID
                password: Password
                record_id: Record ID to be signed
                record_type: Record type
                meaning: Signature meaning (Reviewed, Approved, etc.)
                reason: Reason for signature
    
            Returns:
                Signature record
            """
            # User authentication
            if not self.authenticate_user(user_id, password):
                raise ValueError("Authentication failed: Invalid user ID or password")
    
            # Create signature record
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
    
            # Save signature
            if record_id not in self.signatures:
                self.signatures[record_id] = []
            self.signatures[record_id].append(signature)
    
            # Record in audit trail
            self.audit_trail.log_event(
                event_type='SIGN',
                user=user_id,
                action=f"Electronic signature executed: {meaning}",
                record_type=record_type,
                record_id=record_id,
                new_value=meaning,
                reason=reason
            )
    
            return signature
    
        def _create_signature_hash(self, user_id: str, record_id: str, meaning: str) -> str:
            """Generate signature hash"""
            data = f"{user_id}:{record_id}:{meaning}:{datetime.utcnow().isoformat()}"
            return hashlib.sha256(data.encode()).hexdigest()
    
        def verify_signatures(self, record_id: str) -> List[Dict]:
            """Verify signatures for a specific record"""
            return self.signatures.get(record_id, [])
    
    
    # Execution example
    print("=" * 60)
    print("Audit Trail & Electronic Signature System (21 CFR Part 11 Compliant)")
    print("=" * 60)
    
    # Initialize audit trail system
    audit_system = AuditTrailSystem(system_name="Manufacturing Execution System")
    
    # Create batch record
    audit_system.log_event(
        event_type='CREATE',
        user='user001',
        action='Batch record created',
        record_type='BATCH_RECORD',
        record_id='BATCH-2025-0001',
        new_value={'batch_id': 'BATCH-2025-0001', 'product': 'Aspirin Tablets', 'quantity': 10000}
    )
    
    # Update batch record
    audit_system.log_event(
        event_type='UPDATE',
        user='user001',
        action='Reaction temperature recorded',
        record_type='BATCH_RECORD',
        record_id='BATCH-2025-0001',
        old_value={'reaction_temp': None},
        new_value={'reaction_temp': 80.5},
        reason='Process parameter input'
    )
    
    # Create deviation record
    audit_system.log_event(
        event_type='CREATE',
        user='user002',
        action='Deviation record created',
        record_type='DEVIATION',
        record_id='DEV-2025-001',
        new_value={'description': 'Reaction temperature temporarily exceeded upper limit', 'severity': 'Minor'},
        reason='Temperature spike detected'
    )
    
    # Initialize electronic signature system
    esign_system = ElectronicSignatureSystem(audit_system)
    
    # Sign batch record
    try:
        signature1 = esign_system.sign_record(
            user_id='user001',
            password='password123',
            record_id='BATCH-2025-0001',
            record_type='BATCH_RECORD',
            meaning='Reviewed',
            reason='Batch record review completed'
        )
        print(f"\n✅ Electronic signature successful: {signature1['signature_id']}")
    
        signature2 = esign_system.sign_record(
            user_id='user002',
            password='securepass456',
            record_id='BATCH-2025-0001',
            record_type='BATCH_RECORD',
            meaning='Approved',
            reason='Batch release approval'
        )
        print(f"✅ Electronic signature successful: {signature2['signature_id']}")
    
    except ValueError as e:
        print(f"❌ Error: {e}")
    
    # Generate audit trail report
    audit_system.generate_audit_report()
    
    # Export audit trail
    audit_system.export_audit_trail('audit_trail_part11.json', format='json')
    
    # Verify signatures for specific record
    print(f"\n" + "=" * 60)
    print("Electronic Signature Verification")
    print("=" * 60)
    signatures = esign_system.verify_signatures('BATCH-2025-0001')
    for sig in signatures:
        print(f"Signature ID: {sig['signature_id']}")
        print(f"  Signer: {sig['user_id']}")
        print(f"  Date/Time: {sig['timestamp']}")
        print(f"  Meaning: {sig['meaning']}")
        print(f"  Reason: {sig['reason']}")
        print()
    

**Implementation Key Points:**

  * 21 CFR Part 11 compliant audit trail function (timestamp, user ID, change history)
  * Tamper detection function using SHA-256 hash
  * Record chaining using blockchain-like approach
  * Two-factor authentication for electronic signatures (User ID + Password)
  * Clear specification of signature meaning (Reviewed, Approved)
  * Integrity verification and export functions

## 🔒 5.2 Data Integrity and ALCOA+ Principles

### ALCOA+ Principles

Basic principles of data integrity (FDA/MHRA requirements):

  * **A** ttributable: Creator and modifier of data can be identified
  * **L** egible: Data is readable and understandable
  * **C** ontemporaneous: Data is recorded at the time of creation
  * **O** riginal: Original data or authentic copy
  * **A** ccurate: Data is accurate without errors
  * **+** Complete: All relevant data is included
  * **+** Consistent: Data is without contradictions
  * **+** Enduring: Data is preserved for long periods
  * **+** Available: Accessible when needed

**⚠️ Examples of Data Integrity Violations**  
\- Disabling or deleting audit trails  
\- Unauthorized modification of electronic records  
\- Backdating (retrospective time/date changes)  
\- Selective data reporting  
\- Multiple users sharing a single account 

### 💻 Code Example 5.2: Data Integrity Checker
    
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import json
    import warnings
    warnings.filterwarnings('ignore')
    
    class DataIntegrityChecker:
        """Data integrity verification system (ALCOA+ compliant)"""
    
        def __init__(self):
            self.violations = []
    
        def check_attributable(self, df: pd.DataFrame, required_columns: list) -> dict:
            """
            Attributability check: Verify existence of creator/modifier information
    
            Args:
                df: DataFrame to verify
                required_columns: Required columns (e.g., ['created_by', 'modified_by'])
    
            Returns:
                Verification result
            """
            missing_columns = [col for col in required_columns if col not in df.columns]
    
            if missing_columns:
                violation = {
                    'principle': 'Attributable',
                    'severity': 'Critical',
                    'description': f"Required columns missing: {missing_columns}"
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            # Check for NULL values
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                violation = {
                    'principle': 'Attributable',
                    'severity': 'Critical',
                    'description': f"NULL values detected: {null_counts[null_counts > 0].to_dict()}"
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_contemporaneous(self, df: pd.DataFrame, timestamp_col: str,
                                  event_col: str, max_delay_minutes: int = 5) -> dict:
            """
            Contemporaneous check: Verify discrepancy between event occurrence and record time
    
            Args:
                df: DataFrame to verify
                timestamp_col: Timestamp column
                event_col: Event occurrence time column
                max_delay_minutes: Allowable delay time (minutes)
    
            Returns:
                Verification result
            """
            if timestamp_col not in df.columns or event_col not in df.columns:
                violation = {
                    'principle': 'Contemporaneous',
                    'severity': 'Critical',
                    'description': f"Timestamp columns do not exist"
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            # Convert timestamps
            df_temp = df.copy()
            df_temp[timestamp_col] = pd.to_datetime(df_temp[timestamp_col])
            df_temp[event_col] = pd.to_datetime(df_temp[event_col])
    
            # Calculate delay time
            df_temp['delay_minutes'] = (df_temp[timestamp_col] - df_temp[event_col]).dt.total_seconds() / 60
    
            # Records outside allowable range
            delayed_records = df_temp[df_temp['delay_minutes'] > max_delay_minutes]
    
            if len(delayed_records) > 0:
                violation = {
                    'principle': 'Contemporaneous',
                    'severity': 'Major',
                    'description': f"{len(delayed_records)} records with delayed recording (>{max_delay_minutes} min)",
                    'details': delayed_records[['delay_minutes']].to_dict()
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_complete(self, df: pd.DataFrame, mandatory_columns: list) -> dict:
            """
            Completeness check: Verify missing required data
    
            Args:
                df: DataFrame to verify
                mandatory_columns: Mandatory column list
    
            Returns:
                Verification result
            """
            missing_data = {}
    
            for col in mandatory_columns:
                if col not in df.columns:
                    missing_data[col] = 'Column itself does not exist'
                else:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        missing_data[col] = f"{null_count} missing values"
    
            if missing_data:
                violation = {
                    'principle': 'Complete',
                    'severity': 'Critical',
                    'description': "Missing required data detected",
                    'details': missing_data
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_consistent(self, df: pd.DataFrame, validation_rules: dict) -> dict:
            """
            Consistency check: Validate business rules
    
            Args:
                df: DataFrame to verify
                validation_rules: Validation rule dictionary
    
            Returns:
                Verification result
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
                    'description': "Data consistency violations detected",
                    'details': inconsistencies
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def check_accurate(self, df: pd.DataFrame, value_ranges: dict) -> dict:
            """
            Accuracy check: Validate data value ranges
    
            Args:
                df: DataFrame to verify
                value_ranges: Value range dictionary {'column': (min, max)}
    
            Returns:
                Verification result
            """
            out_of_range = {}
    
            for col, (min_val, max_val) in value_ranges.items():
                if col in df.columns:
                    violations = df[(df[col] < min_val) | (df[col] > max_val)]
                    if len(violations) > 0:
                        out_of_range[col] = {
                            'count': len(violations),
                            'range': f"{min_val}-{max_val}",
                            'invalid_values': violations[col].tolist()[:10]  # First 10 items
                        }
    
            if out_of_range:
                violation = {
                    'principle': 'Accurate',
                    'severity': 'Critical',
                    'description': "Out-of-range values detected",
                    'details': out_of_range
                }
                self.violations.append(violation)
                return {'passed': False, 'violation': violation}
    
            return {'passed': True}
    
        def generate_integrity_report(self) -> dict:
            """Generate data integrity report"""
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
            """Print report"""
            report = self.generate_integrity_report()
    
            print("=" * 60)
            print("Data Integrity Verification Report (ALCOA+)")
            print("=" * 60)
            print(f"Verification Date/Time: {report['timestamp']}")
            print(f"\nTotal Violations: {report['total_violations']}")
            print(f"  Critical: {report['critical']}")
            print(f"  Major: {report['major']}")
            print(f"  Minor: {report['minor']}")
    
            if report['passed']:
                print(f"\n✅ Data Integrity Verification: Passed")
            else:
                print(f"\n❌ Data Integrity Verification: Failed")
    
                print(f"\nViolation Details:")
                for i, violation in enumerate(self.violations, 1):
                    print(f"\n{i}. [{violation['severity']}] {violation['principle']}")
                    print(f"   {violation['description']}")
                    if 'details' in violation:
                        print(f"   Details: {json.dumps(violation['details'], indent=2, ensure_ascii=False)}")
    
    
    # Execution example
    print("=" * 60)
    print("Data Integrity Checker (ALCOA+ Compliant)")
    print("=" * 60)
    
    # Generate sample data (assuming batch records)
    np.random.seed(42)
    n_records = 50
    
    df_batch = pd.DataFrame({
        'batch_id': [f'BATCH-{i+1:04d}' for i in range(n_records)],
        'created_by': ['user001'] * 30 + ['user002'] * 15 + [None] * 5,  # Some NULL
        'created_at': [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n_records)],
        'event_time': [datetime(2025, 1, 1) + timedelta(hours=i, minutes=np.random.randint(0, 10)) for i in range(n_records)],
        'reaction_temp': np.random.normal(80, 5, n_records),
        'reaction_time': np.random.normal(120, 10, n_records),
        'yield': np.random.normal(95, 3, n_records),
        'purity': np.random.normal(99.5, 0.5, n_records)
    })
    
    # Intentionally add abnormal values
    df_batch.loc[10, 'reaction_temp'] = 150  # Out of range
    df_batch.loc[20, 'yield'] = 110  # Over 100%
    df_batch.loc[30, 'created_at'] = df_batch.loc[30, 'event_time'] + timedelta(hours=1)  # Delayed recording
    
    # Initialize data integrity checker
    checker = DataIntegrityChecker()
    
    # Attributability check
    print("\n[Attributability Check]")
    result = checker.check_attributable(df_batch, required_columns=['created_by', 'created_at'])
    print(f"Result: {'✅ Passed' if result['passed'] else '❌ Failed'}")
    
    # Contemporaneous check
    print("\n[Contemporaneous Check]")
    result = checker.check_contemporaneous(df_batch, timestamp_col='created_at',
                                           event_col='event_time', max_delay_minutes=5)
    print(f"Result: {'✅ Passed' if result['passed'] else '❌ Failed'}")
    
    # Completeness check
    print("\n[Completeness Check]")
    result = checker.check_complete(df_batch, mandatory_columns=['batch_id', 'created_by',
                                                                  'reaction_temp', 'yield'])
    print(f"Result: {'✅ Passed' if result['passed'] else '❌ Failed'}")
    
    # Accuracy check
    print("\n[Accuracy Check]")
    value_ranges = {
        'reaction_temp': (70, 90),
        'reaction_time': (100, 140),
        'yield': (0, 100),
        'purity': (95, 100)
    }
    result = checker.check_accurate(df_batch, value_ranges)
    print(f"Result: {'✅ Passed' if result['passed'] else '❌ Failed'}")
    
    # Comprehensive report
    checker.print_report()
    

**Implementation Key Points:**

  * Comprehensive data integrity verification based on ALCOA+ principles
  * Automatic checks for attributability, contemporaneousness, completeness, consistency, and accuracy
  * Violation severity classification (Critical, Major, Minor)
  * Detailed report generation and audit support
  * Real-time data verification implementation example

## 📚 Summary

In this chapter and series, we have comprehensively learned AI applications to pharmaceutical manufacturing processes.

### Key Points of Chapter 5

  * CSV (Computer System Validation) lifecycle approach
  * Implementation of 21 CFR Part 11 compliant audit trail and electronic signatures
  * Ensuring data integrity based on ALCOA+ principles
  * Tamper detection function and blockchain-like approach
  * System design capable of responding to regulatory authority inspections

### Series Overview

  * **Chapter 1** : GMP-compliant statistical quality control (SPC, process capability)
  * **Chapter 2** : Electronic batch record analysis and deviation management
  * **Chapter 3** : PAT and real-time quality control (NIR, MSPC)
  * **Chapter 4** : Continuous manufacturing and QbD implementation (DoE, design space)
  * **Chapter 5** : Regulatory compliance and CSV implementation strategy

**🎓 Series Completed**  
You have completed the series "AI Applications to Pharmaceutical Manufacturing Processes." You have acquired practical knowledge that can be immediately applied in pharmaceutical manufacturing settings, from GMP-compliant quality control to regulatory compliance.  
  
Resources for Continued Learning:  
\- Other series in Process Informatics Dojo (chemical plants, semiconductor manufacturing, etc.)  
\- FDA PAT Guidance, ICH Q8-Q11 Guidelines  
\- ISPE (International Society for Pharmaceutical Engineering) GAMP 5 Guide  
\- FDA/MHRA regulatory documents on data integrity 

[← Chapter 4: Continuous Manufacturing and QbD Implementation](<chapter-4.html>) [Back to Series Index →](<index.html>)

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
