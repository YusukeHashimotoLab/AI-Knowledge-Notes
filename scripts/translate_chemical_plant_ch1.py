#!/usr/bin/env python3
"""
Complete translation script for chemical-plant-ai chapter-1
Translates all Japanese text to English while preserving HTML/CSS/JS structure
"""

import re

# Read the complete Japanese source file
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/PI/chemical-plant-ai/chapter-1.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Translation mappings - comprehensive for this chapter
translations = {
    # HTML lang attribute
    'lang="ja"': 'lang="en"',

    # Meta and title
    '第1章：プロセス監視とソフトセンサー - 化学プラントにおけるAIベース異常検知、品質予測、ソフトセンサー設計を実装レベルで習得': 'Chapter 1: Process Monitoring and Soft Sensors - Master AI-based Anomaly Detection, Quality Prediction, and Soft Sensor Design in Chemical Plants at Implementation Level',
    '第1章：プロセス監視とソフトセンサー - 化学プラントへのAI応用': 'Chapter 1: Process Monitoring and Soft Sensors - AI Applications in Chemical Plants',

    # Breadcrumb navigation
    'AI寺子屋トップ': 'AI Terakoya Top',
    'プロセス・インフォマティクス': 'Process Informatics',

    # Header
    '第1章：プロセス監視とソフトセンサー': 'Chapter 1: Process Monitoring and Soft Sensors',
    'AIベース異常検知と品質予測の実装': 'Implementation of AI-Based Anomaly Detection and Quality Prediction',
    '📖 読了時間: 30-35分': '📖 Reading time: 30-35 minutes',
    '📊 難易度: 実践・応用': '📊 Difficulty: Practical/Applied',
    '💻 コード例: 8個': '💻 Code examples: 8',

    # Learning objectives
    '学習目標': 'Learning Objectives',
    'この章を読むことで、以下を習得できます：': 'By reading this chapter, you will be able to:',
    '✅ 統計的異常検知（PCA、Q統計量、T²統計量）を実装できる': '✅ Implement statistical anomaly detection (PCA, Q-statistic, T²-statistic)',
    '✅ 機械学習ベース異常検知（Isolation Forest、Autoencoder、LSTM）を構築できる': '✅ Build machine learning-based anomaly detection (Isolation Forest, Autoencoder, LSTM)',
    '✅ 品質予測モデル（Random Forest）で製品品質を予測できる': '✅ Predict product quality with quality prediction models (Random Forest)',
    '✅ ソフトセンサー（GPR、ニューラルネット）で測定困難な変数を推定できる': '✅ Estimate difficult-to-measure variables using soft sensors (GPR, Neural Networks)',
    '✅ 統合プロセス監視システムを設計・実装できる': '✅ Design and implement integrated process monitoring systems',

    # Section 1.1
    '1.1 化学プラント監視の課題とAI技術': '1.1 Challenges in Chemical Plant Monitoring and AI Technologies',
    '化学プラント特有の監視課題': 'Monitoring Challenges Specific to Chemical Plants',
    '化学プラントのプロセス監視は、製品品質、安全性、経済性を確保するための最重要課題です。従来の閾値ベース監視では検出困難な異常が多数存在します：': 'Process monitoring in chemical plants is a critical issue for ensuring product quality, safety, and economic efficiency. There are many anomalies that are difficult to detect with conventional threshold-based monitoring:',

    '<strong>多変量相関異常</strong>: 個別変数は正常範囲内でも、変数間の相関が異常': '<strong>Multivariate correlation anomalies</strong>: Correlation between variables is abnormal even when individual variables are within normal ranges',
    '<strong>緩やかな劣化</strong>: 触媒活性低下、熱交換器汚れなど、数週間～数ヶ月単位の変化': '<strong>Gradual degradation</strong>: Changes over weeks to months, such as catalyst deactivation and heat exchanger fouling',
    '<strong>測定困難変数</strong>: 製品品質（純度、粘度）、反応率などのオンライン測定が困難': '<strong>Difficult-to-measure variables</strong>: Product quality (purity, viscosity), conversion rate, etc., are difficult to measure online',
    '<strong>非線形挙動</strong>: 反応器の非線形動特性、蒸留塔の複雑な相互作用': '<strong>Nonlinear behavior</strong>: Nonlinear dynamic characteristics of reactors, complex interactions in distillation columns',

    'AI技術による解決アプローチ': 'Solution Approaches Using AI Technologies',
    'プロセス監視課題': 'Process Monitoring Challenges',
    '統計的手法': 'Statistical Methods',
    '機械学習': 'Machine Learning',
    '深層学習': 'Deep Learning',
    'PCA異常検知': 'PCA Anomaly Detection',
    '統計的プロセス管理': 'Statistical Process Control',
    'Random Forest品質予測': 'Random Forest Quality Prediction',
    'GPRソフトセンサー': 'GPR Soft Sensor',
    'Autoencoder異常検知': 'Autoencoder Anomaly Detection',
    'LSTM時系列予測': 'LSTM Time Series Prediction',
    'NN-ソフトセンサー': 'NN-Soft Sensor',

    # Section 1.2
    '1.2 統計的異常検知の実装': '1.2 Implementation of Statistical Anomaly Detection',

    # Code example 1
    'コード例1: PCA法による多変量統計的プロセス監視': 'Code Example 1: Multivariate Statistical Process Monitoring Using PCA',
    '<strong>目的</strong>: 主成分分析（PCA）を用いてQ統計量（SPE）とT²統計量でプロセス異常を検出する。': '<strong>Objective</strong>: Detect process anomalies using Q-statistic (SPE) and T²-statistic with Principal Component Analysis (PCA).',

    # Japanese font settings comment
    '# 日本語フォント設定': '# Font settings',

    # Code comments - comprehensive translation
    '# 化学反応器の正常運転データ生成（訓練データ）': '# Generate normal operation data for chemical reactor (training data)',
    '# プロセス変数: 温度、圧力、流量、濃度（相関あり）': '# Process variables: temperature, pressure, flow rate, concentration (with correlation)',
    '# 正常運転データ': '# Normal operation data',
    '# データ標準化': '# Data standardization',
    '# PCAモデルの構築（主成分数=2）': '# Build PCA model (number of components = 2)',
    '累積寄与率': 'Cumulative contribution ratio',
    '主成分1の寄与率': 'Contribution ratio of PC1',
    '主成分2の寄与率': 'Contribution ratio of PC2',

    # Function docstrings
    '"""Q統計量（残差空間のノルム）を計算"""': '"""Calculate Q-statistic (norm of residual space)"""',
    '"""T²統計量（主成分空間の距離）を計算"""': '"""Calculate T²-statistic (distance in principal component space)"""',

    '# 正常運転データの統計量': '# Statistics of normal operation data',
    '# 管理限界の計算（99%信頼区間）': '# Calculate control limits (99% confidence interval)',
    '管理限界:': 'Control limits:',
    'Q統計量限界': 'Q-statistic limit',
    'T²統計量限界': 'T²-statistic limit',

    '# 異常データの生成（テストデータ）': '# Generate anomaly data (test data)',
    '# ケース1: 温度異常（反応暴走）': '# Case 1: Temperature anomaly (runaway reaction)',
    '# 高温異常': '# High temperature anomaly',
    '# ケース2: 相関異常（センサー故障）': '# Case 2: Correlation anomaly (sensor failure)',
    '# 圧力の相関が崩れる': '# Pressure correlation breaks down',
    '# 正常データ（比較用）': '# Normal data (for comparison)',
    '# テストデータ結合': '# Combine test data',
    '# ラベル（0: 正常, 1: 温度異常, 2: 相関異常）': '# Labels (0: normal, 1: temperature anomaly, 2: correlation anomaly)',
    '# テストデータの標準化': '# Standardize test data',
    '# テストデータの統計量計算': '# Calculate statistics for test data',
    '# 異常検出': '# Anomaly detection',

    '異常検出結果:': 'Anomaly detection results:',
    'Q統計量による検出数': 'Number of detections by Q-statistic',
    'T²統計量による検出数': 'Number of detections by T²-statistic',
    '統合検出数': 'Combined detection count',

    '# 可視化': '# Visualization',
    '# Q統計量の時系列プロット': '# Time series plot of Q-statistic',
    'Q統計量': 'Q-statistic',
    '管理限界 (99%)': 'Control limit (99%)',
    'サンプル番号': 'Sample number',
    '温度異常': 'Temperature anomaly',
    '相関異常': 'Correlation anomaly',
    'Q統計量による異常検知（残差空間）': 'Anomaly Detection by Q-statistic (Residual Space)',

    '# T²統計量の時系列プロット': '# Time series plot of T²-statistic',
    'T²統計量': 'T²-statistic',
    'T²統計量による異常検知（主成分空間）': 'Anomaly Detection by T²-statistic (Principal Component Space)',

    '# Q-T²プロット': '# Q-T² plot',
    '正常': 'Normal',
    'Q限界': 'Q limit',
    'T²限界': 'T² limit',
    'Q-T²プロット（異常診断）': 'Q-T² Plot (Anomaly Diagnosis)',

    '# 主成分スコアプロット': '# Principal component score plot',
    '第1主成分': '1st Principal Component',
    '第2主成分': '2nd Principal Component',
    '主成分スコアプロット': 'Principal Component Score Plot',

    '<strong>解説</strong>: PCAベース監視は化学プラントで最も広く使用される統計的手法です。Q統計量は残差空間の異常（センサー故障、相関崩れ）を検出し、T²統計量は主成分空間の異常（プロセス変動）を検出します。両者を組み合わせることで、異なる種類の異常を診断できます。': '<strong>Explanation</strong>: PCA-based monitoring is the most widely used statistical method in chemical plants. The Q-statistic detects anomalies in the residual space (sensor failures, correlation breakdowns), while the T²-statistic detects anomalies in the principal component space (process variations). By combining both, different types of anomalies can be diagnosed.',

    # Code example 2
    'コード例2: Isolation Forestによる多変量異常検知': 'Code Example 2: Multivariate Anomaly Detection Using Isolation Forest',
    '<strong>目的</strong>: Isolation Forestアルゴリズムでプロセス異常を検出し、異常スコアを可視化する。': '<strong>Objective</strong>: Detect process anomalies using the Isolation Forest algorithm and visualize anomaly scores.',

    '# 蒸留塔の運転データ生成': '# Generate distillation column operation data',
    '# 正常運転データ': '# Normal operation data',
    '塔頂温度_正常': 'top_temp_normal',
    '塔底温度_正常': 'bottom_temp_normal',
    '還流比_正常': 'reflux_ratio_normal',
    '製品純度_正常': 'product_purity_normal',

    '# 異常運転データ（複数の異常パターン）': '# Anomaly operation data (multiple anomaly patterns)',
    '# パターン1: 塔頂温度異常（冷却器故障）': '# Pattern 1: Top temperature anomaly (cooler failure)',
    '塔頂温度_異常1': 'top_temp_anomaly1',
    '塔底温度_異常1': 'bottom_temp_anomaly1',
    '還流比_異常1': 'reflux_ratio_anomaly1',
    '製品純度_異常1': 'product_purity_anomaly1',

    '# パターン2: 還流比異常（ポンプ故障）': '# Pattern 2: Reflux ratio anomaly (pump failure)',
    '塔頂温度_異常2': 'top_temp_anomaly2',
    '塔底温度_異常2': 'bottom_temp_anomaly2',
    '還流比_異常2': 'reflux_ratio_anomaly2',
    '製品純度_異常2': 'product_purity_anomaly2',

    '# パターン3: 複合異常（原料組成変動）': '# Pattern 3: Complex anomaly (raw material composition variation)',
    '塔頂温度_異常3': 'top_temp_anomaly3',
    '塔底温度_異常3': 'bottom_temp_anomaly3',
    '還流比_異常3': 'reflux_ratio_anomaly3',
    '製品純度_異常3': 'product_purity_anomaly3',

    '# データ統合': '# Data integration',
    '# ラベル（1: 正常, -1: 異常）': '# Labels (1: normal, -1: anomaly)',
    '# DataFrameに変換': '# Convert to DataFrame',
    '塔頂温度': 'Top Temp',
    '塔底温度': 'Bottom Temp',
    '還流比': 'Reflux Ratio',
    '製品純度': 'Product Purity',
    'ラベル': 'Label',

    '# Isolation Forestモデルの訓練': '# Train Isolation Forest model',
    '# 異常データの割合（5%）': '# Proportion of anomaly data (5%)',
    '# 全データで訓練（実務では正常データのみで訓練）': '# Train on all data (in practice, train only on normal data)',
    '# 異常予測': '# Anomaly prediction',
    '# 異常スコア（負の値ほど異常）': '# Anomaly score (more negative = more anomalous)',

    '# 性能評価': '# Performance evaluation',
    '=== Isolation Forest 異常検知性能 ===': '=== Isolation Forest Anomaly Detection Performance ===',
    '混同行列:': 'Confusion matrix:',
    '分類レポート:': 'Classification report:',
    '正常': 'Normal',
    '異常': 'Anomaly',

    '異常スコア統計:': 'Anomaly score statistics:',
    '正常データの平均スコア': 'Average score for normal data',
    '異常データの平均スコア': 'Average score for anomaly data',

    '# 異常スコアの時系列プロット': '# Time series plot of anomaly scores',
    '真の異常': 'True anomaly',
    '判定境界': 'Decision boundary',
    '異常スコア': 'Anomaly Score',
    'Isolation Forest 異常スコア': 'Isolation Forest Anomaly Score',

    '# 異常スコアのヒストグラム': '# Histogram of anomaly scores',
    '頻度': 'Frequency',
    '異常スコア分布': 'Anomaly Score Distribution',

    '# 塔頂温度 vs 製品純度（異常パターン可視化）': '# Top temperature vs product purity (anomaly pattern visualization)',
    '塔頂温度 vs 製品純度（異常検出結果）': 'Top Temperature vs Product Purity (Anomaly Detection Results)',

    '# 還流比 vs 製品純度': '# Reflux ratio vs product purity',
    '還流比 vs 製品純度（異常検出結果）': 'Reflux Ratio vs Product Purity (Anomaly Detection Results)',

    '<strong>解説</strong>: Isolation Forestは、異常データが正常データよりも「分離しやすい」という性質を利用した教師なし学習アルゴリズムです。統計的仮定が不要で、非線形な異常パターンも検出でき、化学プラントの多様な異常に対応できます。計算コストが低く、リアルタイム監視に適しています。': '<strong>Explanation</strong>: Isolation Forest is an unsupervised learning algorithm that exploits the property that anomalous data is "easier to isolate" than normal data. It requires no statistical assumptions, can detect nonlinear anomaly patterns, and can handle diverse anomalies in chemical plants. It has low computational cost and is suitable for real-time monitoring.',

    # Code example 3
    'コード例3: Autoencoderによる非線形異常検知': 'Code Example 3: Nonlinear Anomaly Detection Using Autoencoder',
    '<strong>目的</strong>: ニューラルネットワークのAutoencoderで再構成誤差に基づく異常検知を実装する。': '<strong>Objective</strong>: Implement anomaly detection based on reconstruction error using a neural network Autoencoder.',

    '# 化学反応器の正常運転データ生成': '# Generate normal operation data for chemical reactor',
    '"""正常運転データを生成（非線形相関を含む）"""': '"""Generate normal operation data (including nonlinear correlations)"""',
    '# 正常運転データ（訓練用）': '# Normal operation data (for training)',
    '"""異常運転データを生成"""': '"""Generate anomaly operation data"""',
    '# パターン1: 温度暴走': '# Pattern 1: Temperature runaway',
    '# パターン2: 圧力異常': '# Pattern 2: Pressure anomaly',
    '# 圧力低下': '# Pressure drop',
    '# 転化率低下': '# Conversion rate decrease',
    '# パターン3: 流量異常': '# Pattern 3: Flow rate anomaly',
    '# 流量低下': '# Flow rate decrease',
    '# 異常運転データ（テスト用）': '# Anomaly operation data (for testing)',

    '# Autoencoderモデルの定義': '# Define Autoencoder model',
    '# エンコーダー': '# Encoder',
    '# デコーダー': '# Decoder',
    '# モデルのインスタンス化': '# Instantiate model',
    '# 訓練データをTensorに変換': '# Convert training data to Tensor',

    '# モデルの訓練': '# Model training',
    '=== Autoencoder訓練開始 ===': '=== Autoencoder Training Started ===',
    '# 順伝播': '# Forward propagation',
    '# 逆伝播': '# Backward propagation',

    '# テストデータで再構成誤差を計算': '# Calculate reconstruction error on test data',
    '# 正常データ': '# Normal data',
    '# 異常データ': '# Anomaly data',

    '# 閾値の設定（訓練データの99パーセンタイル）': '# Set threshold (99th percentile of training data)',
    '再構成誤差閾値（99%）': 'Reconstruction error threshold (99%)',
    '正常データの平均再構成誤差': 'Average reconstruction error for normal data',
    '異常データの平均再構成誤差': 'Average reconstruction error for anomaly data',

    '# ROC-AUC評価': '# ROC-AUC evaluation',
    'ROC-AUC スコア': 'ROC-AUC score',

    '# 訓練損失': '# Training loss',
    '訓練損失の推移': 'Training Loss Progress',

    '# 再構成誤差のヒストグラム': '# Histogram of reconstruction error',
    '再構成誤差': 'Reconstruction Error',
    '閾値 (99%)': 'Threshold (99%)',
    '再構成誤差分布': 'Reconstruction Error Distribution',

    '# 再構成誤差の時系列（テストデータ）': '# Time series of reconstruction error (test data)',
    '閾値': 'Threshold',
    'テストデータの再構成誤差': 'Reconstruction Error on Test Data',

    '# ROC曲線': '# ROC curve',
    'ROC曲線': 'ROC Curve',

    '<strong>解説</strong>: Autoencoderは、正常データの特徴を低次元表現（潜在変数）に圧縮し、再構成する深層学習モデルです。異常データは正常データとは異なる特徴を持つため、再構成誤差が大きくなります。非線形な変数間関係を学習でき、PCAでは捉えられない複雑な異常パターンを検出できます。': '<strong>Explanation</strong>: The Autoencoder is a deep learning model that compresses the features of normal data into low-dimensional representations (latent variables) and reconstructs them. Since anomalous data has different features from normal data, the reconstruction error becomes large. It can learn nonlinear relationships between variables and detect complex anomaly patterns that cannot be captured by PCA.',

    # Code example 4
    'コード例4: LSTMによる時系列異常検知': 'Code Example 4: Time Series Anomaly Detection Using LSTM',
    '<strong>目的</strong>: LSTM（Long Short-Term Memory）で時系列パターンを学習し、予測誤差に基づく異常検知を実装する。': '<strong>Objective</strong>: Learn time series patterns with LSTM (Long Short-Term Memory) and implement anomaly detection based on prediction error.',

    '# 時系列プロセスデータ生成（バッチ反応器の温度プロファイル）': '# Generate time series process data (temperature profile of batch reactor)',
    '"""バッチ反応器の温度プロファイルを生成"""': '"""Generate temperature profile of batch reactor"""',
    '# 時間（時間）': '# Time (hours)',
    '# 正常バッチ: 典型的な発熱反応プロファイル': '# Normal batch: typical exothermic reaction profile',
    '# 異常バッチ: 異常な温度上昇パターン': '# Anomaly batch: abnormal temperature rise pattern',
    '# パターン1: 過度な発熱': '# Pattern 1: Excessive heat generation',
    '# パターン2: 不十分な反応': '# Pattern 2: Insufficient reaction',

    '# 訓練データ（正常バッチのみ）': '# Training data (normal batches only)',
    '=== データセット ===': '=== Dataset ===',
    '訓練バッチ数': 'Number of training batches',
    'テスト正常バッチ数': 'Number of test normal batches',
    'テスト異常バッチ数': 'Number of test anomaly batches',
    'バッチ長': 'Batch length',

    '# 時系列データをLSTM用に整形（バッチ, シーケンス長, 特徴数）': '# Reshape time series data for LSTM (batch, sequence length, features)',
    '# LSTMモデルの定義': '# Define LSTM model',
    '# LSTM層': '# LSTM layer',
    '# 全結合層で各時刻の予測値を出力': '# Output prediction for each time step with fully connected layer',

    '# モデル訓練（1ステップ先予測）': '# Model training (1-step ahead prediction)',
    '=== LSTM訓練開始 ===': '=== LSTM Training Started ===',
    '# 入力: t=0~98, ターゲット: t=1~99（1ステップ先予測）': '# Input: t=0~98, Target: t=1~99 (1-step ahead prediction)',

    '# テストデータで予測誤差を計算': '# Calculate prediction error on test data',
    '# 正常バッチ': '# Normal batches',
    '# 異常バッチ': '# Anomaly batches',

    '# 閾値設定（訓練データの95パーセンタイル）': '# Set threshold (95th percentile of training data)',
    '予測誤差閾値（95%）': 'Prediction error threshold (95%)',
    '正常バッチの平均予測誤差': 'Average prediction error for normal batches',
    '異常バッチの平均予測誤差': 'Average prediction error for anomaly batches',

    '# 異常検出性能': '# Anomaly detection performance',
    '検出精度': 'Detection accuracy',

    'LSTM訓練損失': 'LSTM Training Loss',
    '# 予測誤差のヒストグラム': '# Histogram of prediction error',
    '予測誤差（MSE）': 'Prediction Error (MSE)',
    '閾値 (95%)': 'Threshold (95%)',
    '予測誤差分布': 'Prediction Error Distribution',

    '# 正常バッチの予測例': '# Prediction example for normal batch',
    '実測値': 'Actual Value',
    'LSTM予測': 'LSTM Prediction',
    '時間ステップ': 'Time Step',
    '温度 (K)': 'Temperature (K)',
    '正常バッチの予測例': 'Prediction Example for Normal Batch',

    '# 異常バッチの予測例': '# Prediction example for anomaly batch',
    '実測値（異常）': 'Actual Value (Anomaly)',
    '異常バッチの予測例（予測誤差大）': 'Prediction Example for Anomaly Batch (Large Prediction Error)',

    '<strong>解説</strong>: LSTMは時系列データの長期依存関係を学習できる再帰型ニューラルネットワークです。バッチプロセスの温度プロファイルなど、時間的パターンが重要な監視対象に有効です。正常な時系列パターンを学習し、異常パターンでは予測誤差が増大するため、異常検知が可能になります。': '<strong>Explanation</strong>: LSTM is a recurrent neural network that can learn long-term dependencies in time series data. It is effective for monitoring targets where temporal patterns are important, such as temperature profiles of batch processes. It learns normal time series patterns, and prediction errors increase for anomalous patterns, enabling anomaly detection.',

    # Section 1.3
    '1.3 品質予測とソフトセンサー': '1.3 Quality Prediction and Soft Sensors',
    'ソフトセンサーとは': 'What is a Soft Sensor',
    '<strong>ソフトセンサー（Soft Sensor）</strong>は、測定困難または測定コストが高い変数（製品品質、反応率、不純物濃度など）を、測定容易なプロセス変数（温度、圧力、流量など）から推定する技術です。': '<strong>Soft Sensor</strong> is a technology that estimates variables that are difficult or costly to measure (such as product quality, conversion rate, impurity concentration) from easily measurable process variables (such as temperature, pressure, flow rate).',

    '<strong>利点</strong>:': '<strong>Advantages</strong>:',
    'リアルタイム品質監視（分析計は数分～数時間の遅れ）': 'Real-time quality monitoring (analyzers have delays of minutes to hours)',
    'コスト削減（高価な分析計の代替）': 'Cost reduction (alternative to expensive analyzers)',
    'プロセス制御の高度化（品質フィードバック制御）': 'Advanced process control (quality feedback control)',
    '保全性向上（分析計の故障時のバックアップ）': 'Improved maintainability (backup when analyzers fail)',

    # Code example 5
    'コード例5: Random Forestによる製品品質予測': 'Code Example 5: Product Quality Prediction Using Random Forest',
    '<strong>目的</strong>: Random Forestで蒸留塔の製品純度を予測する品質予測モデルを構築する。': '<strong>Objective</strong>: Build a quality prediction model to predict product purity of distillation column using Random Forest.',

    '# 蒸留塔の運転データと製品純度の関係を生成': '# Generate relationship between distillation column operation data and product purity',
    '# プロセス変数（入力）': '# Process variables (inputs)',
    '塔頂温度': 'Top Temp',
    '塔底温度': 'Bottom Temp',
    '還流比': 'Reflux Ratio',
    '原料流量': 'Feed Flow Rate',
    '塔内圧力': 'Column Pressure',

    '# 製品純度（目的変数）- 非線形な関係': '# Product purity (target variable) - nonlinear relationship',
    '# 塔頂温度が高いと純度低下': '# Higher top temperature reduces purity',
    '# 塔底温度が高いと純度向上': '# Higher bottom temperature increases purity',
    '# 還流比が高いと純度向上': '# Higher reflux ratio increases purity',
    '# 流量が多いと純度低下': '# Higher flow rate reduces purity',
    '# 圧力が高いと純度向上': '# Higher pressure increases purity',
    '# 非線形効果': '# Nonlinear effect',
    '# 交互作用': '# Interaction',
    '# 測定ノイズ': '# Measurement noise',

    '# DataFrameに格納': '# Store in DataFrame',
    '製品純度': 'Product Purity',
    '# データ分割': '# Data split',

    '=== データセット ===': '=== Dataset ===',
    '訓練データ数': 'Number of training data',
    'テストデータ数': 'Number of test data',
    '特徴変数数': 'Number of feature variables',

    '# Random Forestモデルの訓練': '# Train Random Forest model',
    '# 予測': '# Prediction',

    '# 性能評価': '# Performance evaluation',
    '=== モデル性能 ===': '=== Model Performance ===',
    '訓練データ:': 'Training data:',
    'テストデータ:': 'Test data:',

    '# 特徴重要度': '# Feature importance',
    '特徴量': 'Feature',
    '重要度': 'Importance',
    '=== 特徴重要度 ===': '=== Feature Importance ===',

    '# 予測 vs 実測（テストデータ）': '# Prediction vs actual (test data)',
    '実測純度': 'Actual Purity',
    '予測純度': 'Predicted Purity',
    '理想直線': 'Ideal Line',
    '予測 vs 実測（R²=': 'Prediction vs Actual (R²=',

    '# 残差プロット': '# Residual plot',
    '残差': 'Residual',
    '±2σ範囲': '±2σ range',
    '残差プロット': 'Residual Plot',

    '特徴重要度（Random Forest）': 'Feature Importance (Random Forest)',

    '# 時系列予測プロット（最初の100サンプル）': '# Time series prediction plot (first 100 samples)',
    '実測値': 'Actual Value',
    '予測値': 'Predicted Value',
    '品質予測の時系列プロット': 'Time Series Plot of Quality Prediction',

    '<strong>解説</strong>: Random Forestは、非線形関係や変数間の交互作用を自動的に学習でき、外れ値に頑健な特性を持ちます。特徴重要度により、品質に影響する主要なプロセス変数を特定できます。化学プラントでは、分析計の測定遅れ（数分～数時間）を補完し、リアルタイム品質監視を実現します。': '<strong>Explanation</strong>: Random Forest can automatically learn nonlinear relationships and interactions between variables, and has robust characteristics against outliers. Feature importance allows identification of key process variables that affect quality. In chemical plants, it complements the measurement delay of analyzers (minutes to hours) and realizes real-time quality monitoring.',

    # Code example 6
    'コード例6: ガウス過程回帰（GPR）によるソフトセンサー設計': 'Code Example 6: Soft Sensor Design Using Gaussian Process Regression (GPR)',
    '<strong>目的</strong>: Gaussian Process Regressionで不確実性を含む品質予測ソフトセンサーを構築する。': '<strong>Objective</strong>: Build a quality prediction soft sensor with uncertainty using Gaussian Process Regression.',

    '# 化学反応器の転化率予測ソフトセンサー': '# Conversion rate prediction soft sensor for chemical reactor',
    '# プロセス変数': '# Process variables',
    '温度': 'Temperature',
    '圧力': 'Pressure',
    '触媒濃度': 'Catalyst Conc',

    '# 転化率（アレニウス型の非線形関係）': '# Conversion rate (Arrhenius-type nonlinear relationship)',
    '活性化エネルギー': 'activation_energy',
    '反応速度定数': 'reaction_rate_constant',
    '転化率': 'Conversion Rate',
    '# 0-1の範囲にクリップ': '# Clip to 0-1 range',

    '=== ソフトセンサー構築 ===': '=== Soft Sensor Construction ===',

    '# ガウス過程回帰カーネルの定義': '# Define Gaussian Process Regression kernel',
    '# RBFカーネル + ホワイトノイズ（測定ノイズを考慮）': '# RBF kernel + white noise (considering measurement noise)',

    '# GPRモデルの訓練': '# Train GPR model',
    '最適化されたカーネル:': 'Optimized kernel:',

    '# 予測（平均と標準偏差）': '# Prediction (mean and standard deviation)',

    '=== ソフトセンサー性能 ===': '=== Soft Sensor Performance ===',
    'テストデータ MAE': 'Test data MAE',
    'テストデータ R²': 'Test data R²',
    '平均予測不確実性（σ）': 'Average prediction uncertainty (σ)',

    '# 予測区間内のカバー率（95%信頼区間）': '# Coverage rate within prediction interval (95% confidence interval)',
    '95%予測区間カバー率': '95% prediction interval coverage rate',

    '# 予測 vs 実測（不確実性付き）': '# Prediction vs actual (with uncertainty)',
    '予測標準偏差': 'Prediction Std Dev',
    '実測転化率': 'Actual Conversion',
    '予測転化率': 'Predicted Conversion',
    'GPRソフトセンサー（R²=': 'GPR Soft Sensor (R²=',

    '# 予測不確実性のヒストグラム': '# Histogram of prediction uncertainty',
}

# Additional translations for continuation of the file
additional_translations = {
    '予測不確実性': 'Prediction Uncertainty',
    '予測不確実性の分布': 'Distribution of Prediction Uncertainty',

    '# 不確実性プロット（温度の関数）': '# Uncertainty plot (as function of temperature)',
    '予測不確実性（標準偏差）': 'Prediction Uncertainty (Std Dev)',
    '温度依存性': 'Temperature Dependence',

    '# 予測区間プロット': '# Prediction interval plot',
    '95%予測区間': '95% Prediction Interval',
    '予測区間プロット（テストデータ）': 'Prediction Interval Plot (Test Data)',

    '<strong>解説</strong>: GPRは予測値だけでなく予測の不確実性（標準偏差）も提供します。これにより、信頼性の低い予測を識別し、安全マージンを考慮した制御が可能になります。カーネル関数により、プロセスの特性に応じた柔軟なモデリングができ、少量のデータでも高精度な予測が可能です。': '<strong>Explanation</strong>: GPR provides not only predicted values but also prediction uncertainty (standard deviation). This allows identification of low-confidence predictions and control considering safety margins. Kernel functions enable flexible modeling according to process characteristics, and high-accuracy predictions are possible even with small amounts of data.',

    # Code example 7
    'コード例7: ニューラルネットワークベースソフトセンサー': 'Code Example 7: Neural Network-Based Soft Sensor',
    '<strong>目的</strong>: 多層ニューラルネットワークで複雑な非線形関係を学習するソフトセンサーを構築する。': '<strong>Objective</strong>: Build a soft sensor that learns complex nonlinear relationships using a multi-layer neural network.',

    '# 重合プロセスの粘度予測ソフトセンサー': '# Viscosity prediction soft sensor for polymerization process',
    '重合温度': 'Polymerization Temp',
    '開始剤濃度': 'Initiator Conc',
    'モノマー流量': 'Monomer Flow Rate',
    '反応時間': 'Reaction Time',
    '粘度': 'Viscosity',

    '# 粘度（複雑な非線形関係）': '# Viscosity (complex nonlinear relationship)',
    '# ニューラルネットワークベースソフトセンサーの定義': '# Define neural network-based soft sensor',
    '# NN訓練': '# NN training',
    '=== ソフトセンサー訓練 ===': '=== Soft Sensor Training ===',

    '# 予測とモデル評価': '# Prediction and model evaluation',
    '=== NN-ソフトセンサー性能 ===': '=== NN-Soft Sensor Performance ===',

    '訓練損失曲線': 'Training Loss Curve',
    'NN-ソフトセンサー（R²=': 'NN-Soft Sensor (R²=',

    '<strong>解説</strong>: ニューラルネットワークは、多層構造により極めて複雑な非線形関係を学習できます。重合プロセスの粘度予測など、物理モデルの構築が困難な対象に有効です。ドロップアウト層により過学習を抑制し、汎化性能を向上させています。': '<strong>Explanation</strong>: Neural networks can learn extremely complex nonlinear relationships through multi-layer structures. They are effective for targets where physical model construction is difficult, such as viscosity prediction in polymerization processes. Dropout layers suppress overfitting and improve generalization performance.',

    # Section 1.4
    '1.4 統合プロセス監視システムの実装': '1.4 Implementation of Integrated Process Monitoring System',
    '統合監視システムの設計思想': 'Design Philosophy of Integrated Monitoring System',
    '実際の化学プラントでは、複数の監視・予測手法を統合した総合監視システムが必要です：': 'In actual chemical plants, a comprehensive monitoring system integrating multiple monitoring and prediction methods is necessary:',

    '<strong>階層的監視</strong>: 統計的監視 → 機械学習ベース監視 → 深層学習ベース監視': '<strong>Hierarchical monitoring</strong>: Statistical monitoring → Machine learning-based monitoring → Deep learning-based monitoring',
    '<strong>相補的検知</strong>: 異なる手法で異なる種類の異常を検出': '<strong>Complementary detection</strong>: Detect different types of anomalies with different methods',
    '<strong>品質統合</strong>: 異常検知とソフトセンサーを統合した品質管理': '<strong>Quality integration</strong>: Quality control integrating anomaly detection and soft sensors',
    '<strong>説明可能性</strong>: 異常の原因を診断・解釈できる機能': '<strong>Explainability</strong>: Function to diagnose and interpret causes of anomalies',

    # Code example 8
    'コード例8: 統合プロセス監視システムの実装': 'Code Example 8: Implementation of Integrated Process Monitoring System',
    '<strong>目的</strong>: PCA、Isolation Forest、Autoencoderを統合した多層監視システムを構築する。': '<strong>Objective</strong>: Build a multi-layer monitoring system integrating PCA, Isolation Forest, and Autoencoder.',

    '# 統合プロセス監視システムクラス': '# Integrated process monitoring system class',
    '"""': '"""',
    '統合プロセス監視システム': 'Integrated Process Monitoring System',
    '複数の異常検知手法を統合し、異常レベルと診断情報を提供': 'Integrates multiple anomaly detection methods and provides anomaly level and diagnostic information',

    '# 監視システムの初期化': '# Initialize monitoring system',
    '# 正常運転データで全監視モデルを訓練': '# Train all monitoring models on normal operation data',
    '監視システム訓練完了': 'Monitoring system training completed',
    'PCA管理限界': 'PCA control limits',
    'Isolation Forest訓練完了': 'Isolation Forest training completed',
    'Autoencoder訓練完了': 'Autoencoder training completed',

    '# プロセスデータの監視': '# Monitor process data',
    '# 総合異常レベルの計算（加重平均）': '# Calculate overall anomaly level (weighted average)',
    '# 異常診断': '# Anomaly diagnosis',

    '# テスト用プロセスデータ生成': '# Generate test process data',
    '# ケース1: 正常運転': '# Case 1: Normal operation',
    '# ケース2: 温度異常': '# Case 2: Temperature anomaly',
    '# ケース3: 相関異常': '# Case 3: Correlation anomaly',

    '# 統合監視システムの訓練': '# Train integrated monitoring system',
    '=== 統合監視システム ===': '=== Integrated Monitoring System ===',

    '# 各ケースを監視': '# Monitor each case',
    '正常運転の監視:': 'Monitoring normal operation:',
    '総合異常レベル': 'Overall anomaly level',
    '診断': 'Diagnosis',

    '温度異常の監視:': 'Monitoring temperature anomaly:',
    '相関異常の監視:': 'Monitoring correlation anomaly:',

    '# 監視結果の可視化': '# Visualize monitoring results',
    '総合異常レベル（統合監視）': 'Overall Anomaly Level (Integrated Monitoring)',
    '異常レベル': 'Anomaly Level',
    '警戒レベル': 'Alert Level',
    '危険レベル': 'Danger Level',

    '# 個別手法の異常スコア': '# Individual method anomaly scores',
    '個別手法の異常スコア（正規化）': 'Individual Method Anomaly Scores (Normalized)',
    'PCAスコア': 'PCA Score',
    'IFスコア': 'IF Score',
    'AEスコア': 'AE Score',

    '# 異常診断結果': '# Anomaly diagnosis results',
    '異常診断': 'Anomaly Diagnosis',
    '診断結果': 'Diagnosis Result',

    '<strong>解説</strong>: 統合監視システムは、複数の手法を組み合わせることで、単一手法では検出困難な異常を捕捉します。PCAは相関異常、Isolation Forestは外れ値、Autoencoderは複雑な非線形異常を得意とします。総合異常レベルにより、プロセス全体の健全性を一元的に評価できます。': '<strong>Explanation</strong>: The integrated monitoring system captures anomalies that are difficult to detect with a single method by combining multiple methods. PCA excels at correlation anomalies, Isolation Forest at outliers, and Autoencoder at complex nonlinear anomalies. The overall anomaly level allows unified evaluation of the health of the entire process.',

    # Section 1.5
    '1.5 実装上の注意点とベストプラクティス': '1.5 Implementation Notes and Best Practices',

    'データ前処理': 'Data Preprocessing',
    '<strong>標準化・正規化</strong>: 異なるスケールの変数を扱う場合は必須': '<strong>Standardization/Normalization</strong>: Essential when dealing with variables of different scales',
    '<strong>外れ値処理</strong>: 訓練データに異常が混入しないよう注意': '<strong>Outlier handling</strong>: Be careful not to mix anomalies into training data',
    '<strong>欠損値補完</strong>: プロセス知識に基づく適切な補完手法の選択': '<strong>Missing value imputation</strong>: Selection of appropriate imputation methods based on process knowledge',

    'モデル選択と検証': 'Model Selection and Validation',
    '<strong>交差検証</strong>: 過学習を防ぐため、訓練データとは独立したテストデータで評価': '<strong>Cross-validation</strong>: Evaluate on test data independent of training data to prevent overfitting',
    '<strong>ハイパーパラメータ調整</strong>: グリッドサーチやベイズ最適化で最適化': '<strong>Hyperparameter tuning</strong>: Optimize with grid search or Bayesian optimization',
    '<strong>性能指標</strong>: 適合率、再現率、F1スコア、ROC-AUCなど、目的に応じた指標を使用': '<strong>Performance metrics</strong>: Use metrics appropriate to the objective, such as precision, recall, F1 score, ROC-AUC',

    '運用上の考慮事項': 'Operational Considerations',
    '<strong>モデル更新</strong>: プロセス変化に対応するため、定期的な再訓練が必要': '<strong>Model updates</strong>: Periodic retraining is necessary to respond to process changes',
    '<strong>False Alarm削減</strong>: 閾値調整やアンサンブル手法で誤検知を最小化': '<strong>False alarm reduction</strong>: Minimize false positives with threshold adjustment and ensemble methods',
    '<strong>計算リソース</strong>: リアルタイム性を確保できる計算速度の確認': '<strong>Computational resources</strong>: Confirm computational speed that can ensure real-time performance',
    '<strong>説明可能性</strong>: オペレータが理解・信頼できる診断情報の提供': '<strong>Explainability</strong>: Provide diagnostic information that operators can understand and trust',

    # Section 1.6
    '1.6 まとめと次章への接続': '1.6 Summary and Connection to Next Chapter',

    'この章で学んだこと': 'What You Learned in This Chapter',
    '統計的手法（PCA）による多変量異常検知の実装': 'Implementation of multivariate anomaly detection using statistical methods (PCA)',
    '機械学習（Isolation Forest、Random Forest）によるプロセス監視と品質予測': 'Process monitoring and quality prediction using machine learning (Isolation Forest, Random Forest)',
    '深層学習（Autoencoder、LSTM、NN）による非線形・時系列異常検知とソフトセンサー': 'Nonlinear and time series anomaly detection and soft sensors using deep learning (Autoencoder, LSTM, NN)',
    'ガウス過程回帰（GPR）による不確実性を考慮した予測': 'Prediction considering uncertainty using Gaussian Process Regression (GPR)',
    '複数手法を統合した総合監視システムの設計': 'Design of comprehensive monitoring system integrating multiple methods',

    '次章への展望': 'Outlook to Next Chapter',
    '次章では、<strong>第2章：プロセス最適化とスケジューリング</strong>として、以下を学びます：': 'In the next chapter, <strong>Chapter 2: Process Optimization and Scheduling</strong>, you will learn:',

    '数理最適化（線形計画法、混合整数計画法）によるプロセス最適化': 'Process optimization using mathematical optimization (linear programming, mixed-integer programming)',
    'メタヒューリスティクス（遺伝的アルゴリズム、粒子群最適化）による複雑な最適化問題の解法': 'Solution of complex optimization problems using metaheuristics (genetic algorithms, particle swarm optimization)',
    '機械学習ベース最適化（ベイズ最適化、強化学習）による適応的制御': 'Adaptive control using machine learning-based optimization (Bayesian optimization, reinforcement learning)',
    'プロダクションスケジューリングの実装': 'Implementation of production scheduling',
    'リアルタイム最適化（RTO）システムの構築': 'Construction of real-time optimization (RTO) systems',

    # Navigation
    '前のページ': 'Previous page',
    '次のページ': 'Next page',
    'コース目次に戻る': 'Back to course index',

    # Footer
    '免責事項': 'Disclaimer',
    'このコンテンツはAIモデルによって生成されたものであり、教育目的のみで提供されています。': 'This content was generated by an AI model and is provided for educational purposes only.',
    '実際のプロセス設計・運用には、専門家の監督と検証が必要です。': 'Actual process design and operation require expert supervision and verification.',
    'コード例は説明目的のため簡略化されており、本番環境での使用には追加の検証とテストが必要です。': 'Code examples are simplified for explanatory purposes and require additional validation and testing for production use.',
    '化学プロセスの安全性には特に注意を払い、必ず安全規制とガイドラインに従ってください。': 'Pay special attention to the safety of chemical processes and always follow safety regulations and guidelines.',
}

# Merge translations
translations.update(additional_translations)

# Apply translations
for jp, en in translations.items():
    content = content.replace(jp, en)

# Write the translated content
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/PI/chemical-plant-ai/chapter-1.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Translation completed successfully!")
print(f"Output file: /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/PI/chemical-plant-ai/chapter-1.html")
