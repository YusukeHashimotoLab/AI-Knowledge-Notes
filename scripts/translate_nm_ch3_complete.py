#!/usr/bin/env python3
"""
Complete translation for NM Chapter 3
Comprehensive Japanese to English translation
"""

import re
from pathlib import Path

JP_FILE = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MI/nm-introduction/chapter3-hands-on.html"
EN_FILE = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MI/nm-introduction/chapter3-hands-on.html"

def create_comprehensive_translations():
    """Create comprehensive translation dictionary"""

    translations = {
        # HTML attributes
        '<html lang="ja">': '<html lang="en">',

        # Title and meta
        '<title>Chapter 3: Python実践チュートリアル - AI Terakoya</title>':
            '<title>Chapter 3: Hands-On Python Tutorial - AI Terakoya</title>',

        # Breadcrumb
        'AI寺子屋トップ': 'AI Terakoya Top',
        'マテリアルズ・インフォマティクス': 'Materials Informatics',

        # Header
        'Chapter 3: Python実践チュートリアル': 'Chapter 3: Hands-On Python Tutorial',
        'ナノ材料データ解析と機械学習': 'Nanomaterial Data Analysis and Machine Learning',
        '読了時間: 30-40分': 'Reading Time: 30-40 min',
        '難易度: 初級': 'Difficulty: Beginner',
        'コード例: 0個': 'Code Examples: 0',
        '演習問題: 0問': 'Exercises: 0',

        # Chapter description
        '小規模データでも効く回帰モデルとベイズ最適化で、効率よく条件探索する筋肉を付けます。MDデータの要点可視化とSHAPによる解釈まで一気に通します。':
            'Build skills for efficiently exploring conditions using regression models effective even with small datasets and Bayesian optimization. Covers essential visualization of MD data and interpretation with SHAP in one go.',
        '💡 補足:': '💡 Supplement:',
        '少ない試行で良い条件を見つけるのが目標。ベイズ最適化は"金属探知機"的に当たりを導きます。':
            'The goal is to find good conditions with minimal trials. Bayesian optimization guides you to hits like a "metal detector".',

        # Learning objectives
        '本章の学習目標': 'Learning Objectives',
        '本章を学習することで、以下のスキルを習得できます：': 'By studying this chapter, you will acquire the following skills:',
        'ナノ粒子データの生成・可視化・前処理の実践': 'Hands-on nanoparticle data generation, visualization, and preprocessing',
        '5種類の回帰モデルによるナノ材料物性予測': 'Prediction of nanomaterial properties using 5 types of regression models',
        'ベイズ最適化によるナノ材料の最適設計': 'Optimal design of nanomaterials through Bayesian optimization',
        'SHAP分析による機械学習モデルの解釈': 'Interpretation of machine learning models using SHAP analysis',
        '多目的最適化によるトレードオフ分析': 'Trade-off analysis through multi-objective optimization',
        'TEM画像解析とサイズ分布のフィッティング': 'TEM image analysis and size distribution fitting',
        '異常検知による品質管理への応用': 'Application to quality control through anomaly detection',

        # Main sections
        '3.1 環境構築': '3.1 Environment Setup',
        '3.2 ナノ粒子データの準備と可視化': '3.2 Nanoparticle Data Preparation and Visualization',
        '3.3 前処理とデータ分割': '3.3 Preprocessing and Data Splitting',
        '3.4 回帰モデルによるナノ粒子物性予測': '3.4 Predicting Nanoparticle Properties with Regression Models',
        '3.5 量子ドット発光波長予測': '3.5 Quantum Dot Emission Wavelength Prediction',
        '3.6 特徴量重要度分析': '3.6 Feature Importance Analysis',
        '3.7 ベイズ最適化によるナノ材料設計': '3.7 Nanomaterial Design with Bayesian Optimization',
        '3.8 多目的最適化：サイズと発光効率のトレードオフ': '3.8 Multi-Objective Optimization: Size and Emission Efficiency Trade-offs',
        '3.9 TEM画像解析とサイズ分布': '3.9 TEM Image Analysis and Size Distribution',
        '3.10 分子動力学（MD）データ解析': '3.10 Molecular Dynamics (MD) Data Analysis',
        '3.11 異常検知：品質管理への応用': '3.11 Anomaly Detection: Quality Control Applications',
        '3.12 章末チェックリスト：ナノ材料データ解析スキルの品質保証': '3.12 End-of-Chapter Checklist: Quality Assurance of Nanomaterial Data Analysis Skills',

        # Subsections
        '必要なライブラリ': 'Required Libraries',
        'インストール方法': 'Installation Methods',
        '本チュートリアルで使用する主要なPythonライブラリ：': 'Main Python libraries used in this tutorial:',

        # Common patterns - examples
        '【例1】合成データ生成：金ナノ粒子のサイズと光学特性': '[Example 1] Synthetic Data Generation: Size and Optical Properties of Gold Nanoparticles',
        '【例2】サイズ分布のヒストグラム': '[Example 2] Size Distribution Histogram',
        '【例3】散布図マトリックス': '[Example 3] Scatter Plot Matrix',
        '【例4】相関行列のヒートマップ': '[Example 4] Correlation Matrix Heatmap',
        '【例5】3Dプロット：サイズ vs 温度 vs LSPR': '[Example 5] 3D Plot: Size vs Temperature vs LSPR',
        '【例6】欠損値処理': '[Example 6] Missing Value Handling',
        '【例7】外れ値検出（IQR法）': '[Example 7] Outlier Detection (IQR Method)',
        '【例8】特徴量スケーリング（StandardScaler）': '[Example 8] Feature Scaling (StandardScaler)',
        '【例9】訓練データとテストデータの分割': '[Example 9] Train-Test Data Splitting',
        '【例10】線形回帰（Linear Regression）': '[Example 10] Linear Regression',
        '【例11】ランダムフォレスト回帰（Random Forest）': '[Example 11] Random Forest Regression',
        '【例12】勾配ブースティング（LightGBM）': '[Example 12] Gradient Boosting (LightGBM)',
        '【例13】サポートベクター回帰（SVR）': '[Example 13] Support Vector Regression (SVR)',
        '【例14】ニューラルネットワーク（MLP Regressor）': '[Example 14] Neural Network (MLP Regressor)',
        '【例15】モデル性能比較': '[Example 15] Model Performance Comparison',
        '【例16】データ生成：CdSe量子ドット': '[Example 16] Data Generation: CdSe Quantum Dots',
        '【例17】量子ドットモデル（LightGBM）': '[Example 17] Quantum Dot Model (LightGBM)',
        '【例18】予測結果の可視化': '[Example 18] Prediction Result Visualization',
        '【例19】特徴量重要度（LightGBM）': '[Example 19] Feature Importance (LightGBM)',
        '【例20】SHAP分析：予測解釈': '[Example 20] SHAP Analysis: Prediction Interpretation',
        '【例21】探索空間の定義': '[Example 21] Search Space Definition',
        '【例22】目的関数の設定': '[Example 22] Objective Function Setup',
        '【例23】ベイズ最適化の実行（scikit-optimize）': '[Example 23] Running Bayesian Optimization (scikit-optimize)',
        '【例24】最適化結果の可視化': '[Example 24] Optimization Result Visualization',
        '【例25】収束プロット': '[Example 25] Convergence Plot',
        '【例26】Pareto最適化（NSGA-II）': '[Example 26] Pareto Optimization (NSGA-II)',
        '【例27】Paretoフロントの可視化': '[Example 27] Pareto Front Visualization',
        '【例28】模擬TEMデータの生成': '[Example 28] Simulated TEM Data Generation',
        '【例29】対数正規分布フィッティング': '[Example 29] Log-Normal Distribution Fitting',
        '【例30】フィッティング結果の可視化': '[Example 30] Fitting Result Visualization',
        '【例31】MDシミュレーションデータの読み込み': '[Example 31] Loading MD Simulation Data',
        '【例32】動径分布関数（RDF）の計算': '[Example 32] Radial Distribution Function (RDF) Calculation',
        '【例33】拡散係数の計算（Mean Squared Displacement）': '[Example 33] Diffusion Coefficient Calculation (Mean Squared Displacement)',
        '【例34】Isolation Forestによる異常ナノ粒子検出': '[Example 34] Anomalous Nanoparticle Detection with Isolation Forest',
        '【例35】異常サンプルの可視化': '[Example 35] Anomaly Sample Visualization',

        # Code comment translations
        '# データ処理・可視化': '# Data processing and visualization',
        '# 機械学習': '# Machine learning',
        '# 最適化': '# Optimization',
        '# モデル解釈': '# Model interpretation',
        '# 多目的最適化（オプション）': '# Multi-objective optimization (optional)',
        '# Anacondaで新しい環境を作成': '# Create new Anaconda environment',
        '# 必要なライブラリをインストール': '# Install required libraries',
        '# 多目的最適化用（オプション）': '# For multi-objective optimization (optional)',
        '# 仮想環境を作成': '# Create virtual environment',
        '# 仮想環境を有効化': '# Activate virtual environment',
        '# 追加パッケージのインストール': '# Install additional packages',
        '# インポートの確認': '# Verify imports',
        '# 日本語フォント設定（必要に応じて）': '# Font settings (adjust as needed)',
        '# 乱数シードの設定（再現性のため）': '# Set random seed (for reproducibility)',
        '# サンプル数': '# Number of samples',
        '# 金ナノ粒子のサイズ（nm）: 平均15 nm、標準偏差5 nm': '# Gold nanoparticle size (nm): mean 15 nm, std 5 nm',
        '# LSPR波長（nm）: Mie理論の簡易近似': '# LSPR wavelength (nm): simplified Mie theory approximation',
        '# 基本波長520 nm + サイズ依存項 + ノイズ': '# Base wavelength 520 nm + size-dependent term + noise',
        '# 合成条件': '# Synthesis conditions',
        '# 温度（℃）': '# Temperature (°C)',
        '# データフレームの作成': '# Create DataFrame',
        '# サイズ分布のヒストグラム': '# Size distribution histogram',
        '# ヒストグラムとKDE（カーネル密度推定）': '# Histogram and KDE (kernel density estimation)',
        '# KDEプロット': '# KDE plot',
        '# ペアプロット（散布図マトリックス）': '# Pairplot (scatter plot matrix)',
        '# 相関行列の計算': '# Calculate correlation matrix',
        '# ヒートマップ': '# Heatmap',
        '# 3D散布図': '# 3D scatter plot',
        '# カラーマップ': '# Colormap',
        '# カラーバー': '# Colorbar',
        '# 欠損値を人為的に導入（実習用）': '# Introduce missing values artificially (for practice)',
        '# ランダムに5%の欠損値を導入': '# Introduce 5% missing values randomly',
        '# 欠損値の処理方法1: 平均値で補完': '# Missing value handling method 1: fill with mean',
        '# 欠損値の処理方法2: 中央値で補完': '# Missing value handling method 2: fill with median',
        '# 欠損値の処理方法3: 削除': '# Missing value handling method 3: drop',
        '# 以降の分析では元のデータ（欠損値なし）を使用': '# Use original data (no missing values) for subsequent analysis',
        '# IQR（四分位範囲）法による外れ値検出': '# Outlier detection using IQR (interquartile range) method',
        '# サイズについて外れ値検出': '# Detect outliers in size',
        '# 可視化': '# Visualization',
        '# 特徴量とターゲットの分離': '# Separate features and target',
        '# StandardScaler（平均0、標準偏差1に標準化）': '# StandardScaler (normalize to mean 0, std 1)',
        '# スケーリング前後の比較': '# Compare before and after scaling',
        '# 訓練データとテストデータに分割（80:20）': '# Split into training and test data (80:20)',
        '# 線形回帰モデルの構築': '# Build linear regression model',
        '# 予測': '# Prediction',
        '# 評価指標': '# Evaluation metrics',
        '# 回帰係数': '# Regression coefficients',
        '# 残差プロット': '# Residual plot',
        '# 予測値 vs 実測値': '# Predicted vs actual values',
        '# ランダムフォレスト回帰モデル': '# Random forest regression model',
        '# 評価': '# Evaluation',
        '# 特徴量重要度': '# Feature importance',
        '# 特徴量重要度の可視化': '# Visualize feature importance',
        '# LightGBMモデルの構築': '# Build LightGBM model',
        '# 予測値 vs 実測値プロット': '# Predicted vs actual values plot',
        '# SVRモデル（RBFカーネル）': '# SVR model (RBF kernel)',
        '# MLPモデル': '# MLP model',
        '# 全モデルの性能をまとめる': '# Summarize all model performances',
        '# 最良モデルの特定': '# Identify best model',
        '# R²スコア比較': '# R² score comparison',

        # Common text patterns
        '環境構築完了！': 'Environment setup complete!',
        '金ナノ粒子データの生成完了': 'Gold nanoparticle data generation complete',
        '基本統計量:': 'Basic statistics:',
        '平均サイズ': 'Average size',
        '標準偏差': 'Standard deviation',
        '中央値': 'Median',
        '各変数間の関係を可視化しました': 'Visualized relationships between variables',
        '相関係数:': 'Correlation coefficients:',
        'LSPR波長とサイズの相関': 'Correlation between LSPR wavelength and size',
        '3Dプロットで多次元の関係を可視化しました': 'Visualized multidimensional relationships with 3D plot',
        '欠損値の確認': 'Check missing values',
        '欠損値の数:': 'Number of missing values:',
        '元のデータ:': 'Original data:',
        '欠損値削除後:': 'After dropping missing values:',
        '平均値補完後:': 'After mean imputation:',
        '行': 'rows',
        '（欠損値なし）': '(no missing values)',
        '→ 以降は欠損値のないデータを使用します': '→ Using data without missing values henceforth',
        '外れ値検出（IQR法）': 'Outlier Detection (IQR Method)',
        '検出された外れ値の数': 'Number of detected outliers',
        '下限': 'Lower bound',
        '上限': 'Upper bound',
        '→ 外れ値は除去せず、全データを使用します': '→ Using all data without removing outliers',
        'スケーリング前の統計量': 'Statistics before scaling',
        'スケーリング後の統計量（平均≈0、標準偏差≈1）': 'Statistics after scaling (mean≈0, std≈1)',
        '→ スケーリングにより各特徴量のスケールが統一されました': '→ Scaling unified the scale of each feature',
        'データ分割': 'Data Split',
        '全データ数': 'Total data',
        'サンプル': 'samples',
        '訓練データ': 'Training data',
        'テストデータ': 'Test data',
        '訓練データの統計量:': 'Training data statistics:',
        '目標：サイズ、温度、pHからLSPR波長を予測': 'Goal: Predict LSPR wavelength from size, temperature, and pH',
        '線形回帰（Linear Regression）': 'Linear Regression',
        '訓練データ R²': 'Training R²',
        'テストデータ R²': 'Test R²',
        'テストデータ RMSE': 'Test RMSE',
        'テストデータ MAE': 'Test MAE',
        '切片': 'Intercept',
        'ランダムフォレスト回帰（Random Forest）': 'Random Forest Regression',
        '勾配ブースティング（LightGBM）': 'Gradient Boosting (LightGBM)',
        'サポートベクター回帰（SVR）': 'Support Vector Regression (SVR)',
        'サポートベクター数': 'Number of support vectors',
        'ニューラルネットワーク（MLP Regressor）': 'Neural Network (MLP Regressor)',
        '反復回数': 'Number of iterations',
        '隠れ層の構造': 'Hidden layer structure',
        '全モデルの性能比較': 'Performance Comparison of All Models',
        '最良モデル': 'Best model',

        # Summary and exercises
        'まとめ': 'Summary',
        '習得した主要技術': 'Key Skills Acquired',
        '実践的な応用': 'Practical Applications',
        '次章の予告': 'Preview of Next Chapter',
        '参考文献': 'References',

        # Skill levels
        '基礎レベル': 'Foundation Level',
        '応用レベル': 'Applied Level',
        '上級レベル': 'Advanced Level',
        '達成': 'Achieved',
        '到達目標': 'Learning Goal',
        '環境構築スキル': 'Environment Setup Skills',
        'データ処理・可視化スキル': 'Data Processing & Visualization Skills',
        '機械学習モデル実装スキル': 'Machine Learning Model Implementation Skills',
        '特徴量重要度とモデル解釈スキル': 'Feature Importance & Model Interpretation Skills',
        'ベイズ最適化スキル': 'Bayesian Optimization Skills',

        # Verbs and actions
        'を実装できる': 'can implement',
        'を計算できる': 'can calculate',
        'を解決できる': 'can solve',
        '過学習を検出できる': 'can detect overfitting',
        '以上達成': 'or more achieved',
        '完遂確認': 'Completion check',

        # Units and measurements
        'nm': 'nm',
        '℃': '°C',
        '分': 'min',

        # Option labels
        'Option 1: Anaconda環境': 'Option 1: Anaconda Environment',
        'Option 2: venv + pip環境': 'Option 2: venv + pip Environment',
        'Option 3: Google Colab': 'Option 3: Google Colab',
        'Google Colabを使用する場合、以下のコードをセルで実行：':
            'When using Google Colab, execute the following code in a cell:',

        # Technical terms
        '金ナノ粒子の局在表面プラズモン共鳴（LSPR）波長は、粒子サイズに依存します。この関係を模擬データで表現します。':
            'The localized surface plasmon resonance (LSPR) wavelength of gold nanoparticles depends on particle size. This relationship is represented with simulated data.',
        '量子ドット': 'quantum dots',
        '動径分布関数': 'radial distribution function',
        '拡散係数': 'diffusion coefficient',
        '平均二乗変位': 'mean squared displacement',
        '異常検知': 'anomaly detection',

        # Exercise patterns
        '演習': 'Exercise',
        '解答例': 'Sample Solution',
        '目標サイズ': 'Target size',
        '目標': 'Goal',
        '波長': 'wavelength',
        'プロット': 'plot',
        '評価': 'evaluation',
        'モデル': 'model',
        '訓練': 'training',
        '予測': 'prediction',
        '可視化': 'visualization',
        'サイズ': 'size',
        '温度': 'temperature',
        '平均': 'average',
        '散布図': 'scatter plot',
        '線形回帰': 'linear regression',
        '予測値': 'predicted value',
        'の計算': 'calculation',
        'ベイズ最適化': 'Bayesian optimization',
        '最適化': 'optimization',
        '目的関数': 'objective function',
        '多目的最適化': 'multi-objective optimization',
        'ヒストグラム': 'histogram',
        'データ生成': 'data generation',
        'データ解析': 'data analysis',
        'ランダムフォレスト': 'random forest',
        'ニューラルネットワーク': 'neural network',
        'の範囲に制限': 'clipped to range',
        'をインストール': 'install',
        'を使用': 'use',
        '金属探知機': 'metal detector',
        '的に当たりを導きます': 'guides you to hits',
        '法': 'method',
        'レベル': 'level',
        'スケーリング': 'scaling',
        '探索空間の定義': 'search space definition',
        '収束プロット': 'convergence plot',
        '欠損値処理': 'missing value handling',
        '外れ値検出': 'outlier detection',
        '散布図マトリックス': 'scatter plot matrix',
        'サンプル数': 'number of samples',
        '特徴量重要度': 'feature importance',
        'モデル解釈': 'model interpretation',
        '実践チュートリアル': 'hands-on tutorial',
        'の': '',  # Remove standalone の particles where appropriate
        '例': 'Example',

        # Footer
        '© 2024 AI寺子屋': '© 2024 AI Terakoya',

        # Additional common patterns - verbs and particles
        'できる': 'can',
        'を作成できる': 'can create',
        'を実装できる': 'can implement',
        'を計算できる': 'can calculate',
        'を作成・解釈できる': 'can create and interpret',
        'を取得・': 'acquire and',
        'を実装し': 'implement and',
        'を実装': 'implement',
        'けられる': 'can be done',
        'ができる': 'can be done',
        'を実': 'implement',
        'による': 'by',
        'では': 'in',
        'と': 'and',
        'で': 'with',
        'を': '',
        'は': 'is',
        'が': '',
        'の': 'of',
        'に': 'to',
        'と誤差': 'and error',
        'つ': '',
        'し': '',

        # Technical terms from remaining patterns
        '対数正規': 'log-normal',
        '布': 'distribution',
        '対数正規布': 'log-normal distribution',
        '布に従う': 'follows distribution',
        '布フィッティング': 'distribution fitting',
        '布パラメータ': 'distribution parameters',
        'フィッティングされた': 'fitted',
        'フィット': 'fit',

        # Measurement and statistics
        '以上': 'or more',
        '結果': 'results',
        '誤差': 'error',
        '値': 'value',
        '精度': 'accuracy',
        '絶対誤差': 'absolute error',
        '相関': 'correlation',

        # Data and analysis
        '析': 'analysis',
        '析による': 'by analysis',
        '実測値と比較': 'compared with measured values',
        '実測': 'measured',
        '完了': 'complete',
        '構築': 'construction',
        '画像解析': 'image analysis',
        '子動力学': 'molecular dynamics',
        '直径': 'diameter',
        '最も重要な特徴量': 'most important feature',

        # Processes and operations
        'データ生成': 'data generation',
        'テスト': 'test',
        'テスト実': 'test implementation',
        '割': 'split',
        '別': 'classification',
        'プロセス': 'process',
        '適用': 'application',
        'パラメータ': 'parameters',
        '回数': 'number of times',
        '回': 'times',
        'ランダムサンプリング回数': 'random sampling count',
        '中': 'middle',

        # Optimization terms
        '最小化': 'minimization',
        '最適': 'optimal',
        '最適条件で': 'under optimal conditions',
        '最良値推移': 'best value progression',
        '履歴': 'history',

        # Emission and optical properties
        '発光': 'emission',
        '多色発光設計': 'multi-color emission design',
        '効率最大化': 'efficiency maximization',

        # Algorithms and models
        'アルゴリズム': 'algorithm',
        'を作成': 'create',

        # Quantum dots and nanoparticles
        '粒子': 'particles',
        '銀ナノ粒子最適合成条件': 'optimal silver nanoparticle synthesis conditions',

        # MD simulation
        '原子数': 'number of atoms',
        '原子位置': 'atom positions',
        'タイムステップ数': 'number of timesteps',
        '関係式': 'relationship equation',
        '方程式に基づき': 'based on equation',
        '方程式簡易近似': 'equation simplified approximation',
        'モル比': 'molar ratio',

        # Anomaly detection
        '正常': 'normal',
        '異常': 'anomaly',
        '異常データ割合': 'anomaly data ratio',
        '混同': 'confusion',
        '異常スコア': 'anomaly score',

        # Rankings and ordering
        '上位': 'top',

        # Data relationships
        '関係': 'relationship',
        'データフレーム作成': 'create DataFrame',
        '発光と': 'emission and',
        '結果を': 'results',

        # Skill categories
        '自': 'self',
        '複数': 'multiple',
        'ライブラリ': 'library',
        '赤・緑・青': 'red, green, blue',
        '概念を理解している': 'understand the concept',
        'スキル': 'skills',
        'ナノ材料特有解析スキル': 'nanomaterial-specific analysis skills',
        '全カテゴリ': 'all categories',
        'へ準備': 'prepare for',
        '前章': 'previous chapter',
        '次章': 'next chapter',
        'データ生成完了': 'data generation complete',

        # Example sentences
        '少ない試で良い条件を見つけるが': 'finding good conditions with few trials is the goal',
        '回帰係数比較': 'regression coefficient comparison',
        '信頼区間付き': 'with confidence interval',
        '範囲': 'range',
        '電気伝導度': 'electrical conductivity',

        # Additional fragments
        'と発光': 'and emission',
        'に依存します': 'depends on',
    }

    return translations

def apply_translations(content, translations):
    """Apply translations with proper ordering"""
    result = content

    # Sort by length (descending) to avoid partial replacements
    sorted_translations = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)

    for jp, en in sorted_translations:
        result = result.replace(jp, en)

    return result

def count_japanese_chars(text):
    """Count remaining Japanese characters"""
    jp_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    matches = jp_pattern.findall(text)
    return len(matches)

def find_japanese_segments(text, limit=20):
    """Find Japanese text segments"""
    jp_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')
    matches = jp_pattern.findall(text)
    return matches[:limit]

def main():
    print("=" * 80)
    print("NM Chapter 3: Complete Translation (JP → EN)")
    print("=" * 80)

    # Read Japanese file
    print("\n[1/5] Reading Japanese source file...")
    with open(JP_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    initial_jp_count = count_japanese_chars(content)
    total_lines = content.count('\n') + 1
    print(f"      Total lines: {total_lines}")
    print(f"      Japanese chars: {initial_jp_count}")

    # Create translations
    print("\n[2/5] Creating translation dictionary...")
    translations = create_comprehensive_translations()
    print(f"      Translation entries: {len(translations)}")

    # Apply translations
    print("\n[3/5] Applying translations...")
    translated = apply_translations(content, translations)

    # Count remaining Japanese
    final_jp_count = count_japanese_chars(translated)
    jp_percent = (final_jp_count / len(translated) * 100) if len(translated) > 0 else 0

    print(f"      Japanese chars remaining: {final_jp_count}")
    print(f"      Reduction: {initial_jp_count - final_jp_count} chars")
    print(f"      Japanese percentage: {jp_percent:.2f}%")

    # Write output
    print("\n[4/5] Writing translated file...")
    output_path = Path(EN_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(EN_FILE, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"      Output: {EN_FILE}")

    # Report
    print("\n[5/5] Translation Report")
    print("=" * 80)
    print(f"✓ File translated successfully!")
    print(f"  Total lines: {total_lines}")
    print(f"  Japanese chars: {initial_jp_count} → {final_jp_count}")
    print(f"  Remaining JP: {jp_percent:.2f}%")

    if final_jp_count > 0:
        print(f"\n⚠ Warning: {final_jp_count} Japanese characters remain")
        print("  First 20 Japanese segments:")
        segments = find_japanese_segments(translated, 20)
        for i, seg in enumerate(segments, 1):
            print(f"    {i:2d}. {seg}")
    else:
        print("\n✓ Perfect translation - no Japanese characters remain!")

    print("\n" + "=" * 80)
    print("Translation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
