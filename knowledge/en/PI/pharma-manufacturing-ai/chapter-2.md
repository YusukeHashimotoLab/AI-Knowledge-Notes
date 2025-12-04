---
title: "Chapter 2: Electronic Batch Record Analysis and Deviation Management"
chapter_title: "Chapter 2: Electronic Batch Record Analysis and Deviation Management"
subtitle: Electronic Batch Record Analysis and Deviation Management
---

[AI Terakoya Top](<../index.html>)›[Process Informatics](<../../index.html>)›[Pharma Manufacturing AI](<../../PI/pharma-manufacturing-ai/index.html>)›Chapter 2

🌐 EN | [🇯🇵 JP](<../../../jp/PI/pharma-manufacturing-ai/chapter-2.html>) | Last sync: 2025-11-16

[← Back to Series Index](<index.html>)

## 📖 Chapter Overview

Electronic Batch Records (EBR) in pharmaceutical manufacturing are critical systems that ensure transparency and traceability of manufacturing processes. In this chapter, you will learn how to build a comprehensive deviation management system utilizing AI, from automated EBR data analysis, anomaly detection, root cause analysis (RCA), to proposing corrective and preventive actions (CAPA). 

### 🎯 Learning Objectives

  * Structure of Electronic Batch Records (EBR) and data analysis techniques
  * Visualization of process variation through batch trend analysis
  * Automated detection of anomalous batches using machine learning
  * Data mining for root cause analysis (RCA)
  * Automated generation of CAPA (Corrective and Preventive Action) proposals
  * Automation of deviation management workflows
  * GMP-compliant document management and version control

## 📋 2.1 Fundamentals of Electronic Batch Records (EBR)

### Components of EBR

Electronic batch records consist of the following key elements:

  * **Batch Header** : Batch number, product name, manufacturing date, lot number
  * **Raw Material Records** : Materials used, quantities, lot numbers, expiration dates
  * **Process Parameters** : Temperature, pressure, time, pH, flow rate, etc.
  * **In-Process Testing** : Quality verification results at each process step
  * **Final Product Testing** : Assay, dissolution, purity, microbial testing
  * **Deviation Records** : Anomaly occurrence, cause, countermeasures, approval
  * **Electronic Signatures** : Signatures of operators, reviewers, approvers

**🏭 GMP Requirements (21 CFR Part 11)**  
・Authenticity of electronic records: Prevention of tampering  
・Integrity: Consistency and accuracy of data  
・Reliability: Stable system operation  
・Availability: Guaranteed access when needed  
・Audit Trail: Recording of all change history 

### 💻 Code Example 2.1: EBR Data Model and Batch Trend Analysis
    
    
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
        """Electronic Batch Record (EBR) management class"""
    
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
            """Record process parameters"""
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
            self._log_audit(f"Process parameter recorded: {step_name} - {parameter_name}")
    
        def _check_tolerance(self, target, actual, tolerance):
            """Check tolerance range"""
            lower = target - tolerance
            upper = target + tolerance
            return lower <= actual <= upper
    
        def add_deviation(self, description, severity, root_cause=None, capa=None):
            """Record deviation"""
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
            self._log_audit(f"Deviation recorded: {deviation['id']}")
    
        def add_signature(self, role, user_name):
            """Record electronic signature"""
            signature = {
                'role': role,
                'user': user_name,
                'timestamp': datetime.now().isoformat()
            }
            self.signatures.append(signature)
            self._log_audit(f"Electronic signature: {role} by {user_name}")
    
        def _log_audit(self, action):
            """Record audit trail"""
            entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action
            }
            self.audit_trail.append(entry)
    
        def to_dict(self):
            """Convert to dictionary format"""
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
            """Export to JSON format"""
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"EBR exported to {filename}")
    
    
    class BatchTrendAnalyzer:
        """Batch trend analysis class"""
    
        def __init__(self):
            self.batches = []
    
        def generate_batch_data(self, n_batches=50):
            """Generate sample batch data"""
            np.random.seed(42)
            start_date = datetime(2025, 1, 1)
    
            for i in range(n_batches):
                batch_id = f"B-2025-{i+1:04d}"
                mfg_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    
                # Normal batches (1-35)
                if i < 35:
                    reaction_temp = np.random.normal(80, 1, 1)[0]
                    reaction_time = np.random.normal(120, 5, 1)[0]
                    yield_value = np.random.normal(95, 2, 1)[0]
                    purity = np.random.normal(99.5, 0.3, 1)[0]
    
                # Temperature anomaly batches (36-40)
                elif 35 <= i < 40:
                    reaction_temp = np.random.normal(85, 2, 1)[0]  # Temperature rise
                    reaction_time = np.random.normal(120, 5, 1)[0]
                    yield_value = np.random.normal(92, 3, 1)[0]  # Yield decrease
                    purity = np.random.normal(99.2, 0.5, 1)[0]
    
                # Time anomaly batches (41-45)
                elif 40 <= i < 45:
                    reaction_temp = np.random.normal(80, 1, 1)[0]
                    reaction_time = np.random.normal(140, 10, 1)[0]  # Time extension
                    yield_value = np.random.normal(93, 2, 1)[0]
                    purity = np.random.normal(99.3, 0.4, 1)[0]
    
                # Combined anomaly batches (46-50)
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
            """Visualize batch trends"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            batch_indices = range(len(df))
    
            # Reaction temperature trend
            axes[0, 0].plot(batch_indices, df['reaction_temp'], marker='o', color='#11998e',
                            linewidth=1.5, markersize=4)
            axes[0, 0].axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target (80°C)')
            axes[0, 0].axhline(y=82, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Warning limit (±2°C)')
            axes[0, 0].axhline(y=78, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            axes[0, 0].set_xlabel('Batch Number')
            axes[0, 0].set_ylabel('Reaction Temperature (°C)')
            axes[0, 0].set_title('Reaction Temperature Trend', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
    
            # Reaction time trend
            axes[0, 1].plot(batch_indices, df['reaction_time'], marker='s', color='#38ef7d',
                            linewidth=1.5, markersize=4)
            axes[0, 1].axhline(y=120, color='green', linestyle='--', linewidth=2, label='Target (120 min)')
            axes[0, 1].axhline(y=130, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Warning limit (±10 min)')
            axes[0, 1].axhline(y=110, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            axes[0, 1].set_xlabel('Batch Number')
            axes[0, 1].set_ylabel('Reaction Time (min)')
            axes[0, 1].set_title('Reaction Time Trend', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
    
            # Yield trend
            axes[1, 0].plot(batch_indices, df['yield'], marker='^', color='#4ecdc4',
                            linewidth=1.5, markersize=4)
            axes[1, 0].axhline(y=95, color='green', linestyle='--', linewidth=2, label='Target (95%)')
            axes[1, 0].axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Lower limit (90%)')
            axes[1, 0].set_xlabel('Batch Number')
            axes[1, 0].set_ylabel('Yield (%)')
            axes[1, 0].set_title('Yield Trend', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
    
            # Purity trend
            axes[1, 1].plot(batch_indices, df['purity'], marker='D', color='#f38181',
                            linewidth=1.5, markersize=4)
            axes[1, 1].axhline(y=99.5, color='green', linestyle='--', linewidth=2, label='Target (99.5%)')
            axes[1, 1].axhline(y=99.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Specification lower limit (99.0%)')
            axes[1, 1].set_xlabel('Batch Number')
            axes[1, 1].set_ylabel('Purity (%)')
            axes[1, 1].set_title('Purity Trend', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('batch_trend_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Example execution
    print("=" * 60)
    print("Electronic Batch Record (EBR) Management System")
    print("=" * 60)
    
    # Create EBR instance
    ebr = ElectronicBatchRecord(
        batch_id="B-2025-0042",
        product_name="Aspirin Tablet 100mg",
        manufacturing_date="2025-10-27"
    )
    
    # Record process parameters
    ebr.add_process_parameter("Reaction", "Temperature", target=80, actual=80.5, unit="°C", tolerance=2)
    ebr.add_process_parameter("Reaction", "Time", target=120, actual=118, unit="min", tolerance=10)
    ebr.add_process_parameter("Drying", "Temperature", target=60, actual=61, unit="°C", tolerance=3)
    
    # Record deviation (example)
    ebr.add_deviation(
        description="Reaction temperature temporarily rose to 82°C",
        severity="Minor",
        root_cause="Inappropriate PID parameters in temperature control system",
        capa="Re-adjust PID parameters and set alarm"
    )
    
    # Electronic signatures
    ebr.add_signature("Manufacturing", "Taro Tanaka")
    ebr.add_signature("Quality Assurance", "Hanako Suzuki")
    
    # Export
    ebr.export_json("ebr_sample.json")
    
    print(f"\nBatch ID: {ebr.batch_id}")
    print(f"Product Name: {ebr.product_name}")
    print(f"Process parameters recorded: {sum(len(params) for params in ebr.process_parameters.values())}")
    print(f"Deviation count: {len(ebr.deviations)}")
    print(f"Electronic signature count: {len(ebr.signatures)}")
    
    # Batch trend analysis
    print("\n" + "=" * 60)
    print("Batch Trend Analysis")
    print("=" * 60)
    
    analyzer = BatchTrendAnalyzer()
    df_batches = analyzer.generate_batch_data(n_batches=50)
    
    print(f"\nAnalyzed batch count: {len(df_batches)}")
    print(f"Period: {df_batches['date'].min()} ~ {df_batches['date'].max()}")
    
    # Statistical summary
    print(f"\nReaction temperature: Mean {df_batches['reaction_temp'].mean():.2f}°C, Std {df_batches['reaction_temp'].std():.2f}°C")
    print(f"Reaction time: Mean {df_batches['reaction_time'].mean():.1f} min, Std {df_batches['reaction_time'].std():.1f} min")
    print(f"Yield: Mean {df_batches['yield'].mean():.2f}%, Std {df_batches['yield'].std():.2f}%")
    print(f"Purity: Mean {df_batches['purity'].mean():.2f}%, Std {df_batches['purity'].std():.2f}%")
    
    # Trend visualization
    analyzer.plot_batch_trends(df_batches)
    

**Implementation Points:**

  * GMP-compliant EBR data model (process parameters, deviations, electronic signatures)
  * Automatic audit trail recording functionality
  * Automated tolerance range checking
  * Anomaly detection visualization through batch trend analysis
  * Data export in JSON format (interoperability)

## 🔍 2.2 Anomalous Batch Detection Using Machine Learning

### Approaches to Anomaly Detection

The following machine learning techniques are effective for anomaly detection in batch manufacturing:

  * **Isolation Forest** : Unsupervised anomaly detection, effective for multivariate data
  * **One-Class SVM** : Learning from normal data only, boundary determination
  * **Autoencoder** : Deep learning-based detection using reconstruction error
  * **Statistical Process Control** : Hotelling's T², MEWMA

### 💻 Code Example 2.2: Anomalous Batch Detection with Isolation Forest
    
    
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
        """Anomalous batch detection system"""
    
        def __init__(self, contamination=0.1):
            """
            Args:
                contamination: Estimated proportion of anomalous data (0.1 = 10%)
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
            Train the model
    
            Args:
                df: Batch dataframe
                feature_columns: List of feature columns
            """
            X = df[feature_columns].values
            X_scaled = self.scaler.fit_transform(X)
    
            self.model.fit(X_scaled)
    
            # Calculate anomaly scores
            anomaly_scores = self.model.score_samples(X_scaled)
            predictions = self.model.predict(X_scaled)
    
            # Transform to 2D using PCA (for visualization)
            X_pca = self.pca.fit_transform(X_scaled)
    
            return anomaly_scores, predictions, X_pca
    
        def detect_anomalies(self, df, feature_columns, anomaly_scores, predictions):
            """
            Identify anomalous batches
    
            Args:
                df: Batch dataframe
                feature_columns: Feature columns
                anomaly_scores: Anomaly scores
                predictions: Prediction results (-1: anomaly, 1: normal)
    
            Returns:
                Dataframe of anomalous batches
            """
            df_result = df.copy()
            df_result['anomaly_score'] = anomaly_scores
            df_result['is_anomaly'] = predictions == -1
    
            # Extract anomalous batches
            anomalies = df_result[df_result['is_anomaly']].copy()
    
            # Anomaly importance ranking (lower score = more anomalous)
            anomalies = anomalies.sort_values('anomaly_score')
    
            return df_result, anomalies
    
        def plot_anomaly_detection(self, df_result, X_pca, feature_columns):
            """Visualize anomaly detection results"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # Plot in PCA space
            normal_mask = ~df_result['is_anomaly']
            anomaly_mask = df_result['is_anomaly']
    
            axes[0, 0].scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                               c='green', s=30, alpha=0.6, label='Normal batches')
            axes[0, 0].scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
                               c='red', s=100, alpha=0.8, marker='X', label='Anomalous batches')
            axes[0, 0].set_xlabel(f'1st Principal Component (variance: {self.pca.explained_variance_ratio_[0]:.1%})')
            axes[0, 0].set_ylabel(f'2nd Principal Component (variance: {self.pca.explained_variance_ratio_[1]:.1%})')
            axes[0, 0].set_title('Anomaly Detection in PCA Space', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
    
            # Anomaly score distribution
            axes[0, 1].hist(df_result[normal_mask]['anomaly_score'], bins=30,
                            alpha=0.6, label='Normal', color='green')
            axes[0, 1].hist(df_result[anomaly_mask]['anomaly_score'], bins=30,
                            alpha=0.6, label='Anomaly', color='red')
            axes[0, 1].set_xlabel('Anomaly Score (lower = more anomalous)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
    
            # Time-series anomaly detection
            batch_indices = range(len(df_result))
            colors = ['red' if x else 'green' for x in df_result['is_anomaly']]
    
            axes[1, 0].scatter(batch_indices, df_result['anomaly_score'],
                               c=colors, s=50, alpha=0.7)
            axes[1, 0].axhline(y=df_result['anomaly_score'].quantile(0.1), color='orange',
                               linestyle='--', linewidth=2, label='Anomaly threshold')
            axes[1, 0].set_xlabel('Batch Number')
            axes[1, 0].set_ylabel('Anomaly Score')
            axes[1, 0].set_title('Time-Series Anomaly Detection', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
    
            # Anomalous batch distribution by features
            if len(feature_columns) >= 2:
                feat1, feat2 = feature_columns[0], feature_columns[1]
    
                axes[1, 1].scatter(df_result[normal_mask][feat1], df_result[normal_mask][feat2],
                                   c='green', s=30, alpha=0.6, label='Normal batches')
                axes[1, 1].scatter(df_result[anomaly_mask][feat1], df_result[anomaly_mask][feat2],
                                   c='red', s=100, alpha=0.8, marker='X', label='Anomalous batches')
                axes[1, 1].set_xlabel(feat1)
                axes[1, 1].set_ylabel(feat2)
                axes[1, 1].set_title(f'{feat1} vs {feat2}', fontsize=12, fontweight='bold')
                axes[1, 1].legend()
                axes[1, 1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('anomaly_batch_detection.png', dpi=300, bbox_inches='tight')
            plt.show()
    
        def generate_anomaly_report(self, anomalies, feature_columns):
            """Generate anomaly report"""
            print("\n" + "=" * 60)
            print("Anomalous Batch Detection Report")
            print("=" * 60)
    
            for idx, row in anomalies.iterrows():
                print(f"\n🚨 Batch ID: {row['batch_id']}")
                print(f"   Manufacturing date: {row['date']}")
                print(f"   Anomaly score: {row['anomaly_score']:.4f}")
                print(f"   Process parameters:")
    
                for feat in feature_columns:
                    print(f"     - {feat}: {row[feat]:.2f}")
    
    # Example execution
    print("=" * 60)
    print("Anomalous Batch Detection System (Isolation Forest)")
    print("=" * 60)
    
    # Generate batch data (reuse from previous code example)
    analyzer = BatchTrendAnalyzer()
    df_batches = analyzer.generate_batch_data(n_batches=50)
    
    # Define features
    feature_columns = ['reaction_temp', 'reaction_time', 'yield', 'purity']
    
    # Train anomaly detection model
    detector = AnomalyBatchDetector(contamination=0.15)  # Assume 15% anomalies
    anomaly_scores, predictions, X_pca = detector.train(df_batches, feature_columns)
    
    # Detect anomalous batches
    df_result, anomalies = detector.detect_anomalies(df_batches, feature_columns, anomaly_scores, predictions)
    
    print(f"\nTotal batch count: {len(df_batches)}")
    print(f"Detected anomalous batches: {len(anomalies)} ({len(anomalies)/len(df_batches)*100:.1f}%)")
    
    # Visualization
    detector.plot_anomaly_detection(df_result, X_pca, feature_columns)
    
    # Generate report
    detector.generate_anomaly_report(anomalies.head(5), feature_columns)
    

**Implementation Points:**

  * Unsupervised learning for detecting unknown anomaly patterns
  * Integrated evaluation of multivariate data (temperature, time, yield, purity)
  * Visualization of high-dimensional data using PCA
  * Prioritization using anomaly scores
  * Automated report generation functionality

## 📚 Summary

In this chapter, we learned about electronic batch record analysis and deviation management.

### Key Points

  * Implementation of GMP-compliant EBR data model and audit trail
  * Visualization of process variation through batch trend analysis
  * Automated detection of anomalous batches using machine learning (Isolation Forest)
  * Comprehensive quality evaluation through multivariate data analysis
  * Work efficiency improvement through automated report generation

**🎯 Next Chapter Preview**  
In Chapter 3, you will learn about Process Analytical Technology (PAT) and real-time quality management. You will master more advanced process management techniques including NIR/Raman spectroscopy analysis, Multivariate Statistical Process Control (MSPC), and Real-Time Release Testing (RTRT). 

[← Chapter 1: GMP Statistical Quality Control](<chapter-1.html>) [Chapter 3: PAT and Real-Time Quality Management →](<chapter-3.html>)

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
