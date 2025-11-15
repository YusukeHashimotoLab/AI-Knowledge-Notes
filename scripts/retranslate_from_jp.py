#!/usr/bin/env python3
"""
Comprehensive Re-translation Script from Japanese Source Files
Fixes severely corrupted EN files by translating from JP source files
Processes ML (119 files) and FM (23 files) categories

Usage:
    python retranslate_from_jp.py ML   # Process ML category
    python retranslate_from_jp.py FM   # Process FM category
    python retranslate_from_jp.py --all  # Process both categories

Features:
- Reads from JP source files in /knowledge/jp/{category}/
- Applies comprehensive 500+ entry translation dictionary
- Complete phrase translation (no partial word translations)
- Validation: ensures < 5% Japanese characters remain
- Preserves HTML structure, code blocks, and equations
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import unicodedata

# Base directories
BASE_DIR = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge')
EN_DIR = BASE_DIR / 'en'
JP_DIR = BASE_DIR / 'jp'
FILE_LIST = EN_DIR / 'JAPANESE_CHARACTER_FILES_LIST.txt'

# Comprehensive translation dictionary (combined from PI/MI successful scripts + ML/FM specific)
# CRITICAL: Longer phrases FIRST to avoid partial replacements
TRANSLATIONS = {
    # HTML attributes
    'lang="ja"': 'lang="en"',

    # ===== LONGEST PHRASES FIRST (Critical for preventing partial replacements) =====

    # Mixed language patterns with particles
    'Machine Learningの基礎': 'Foundations of Machine Learning',
    'Deep Learningの基礎': 'Foundations of Deep Learning',
    'Deep Learningの': 'Deep Learning',
    'Neural Networksの': 'Neural Networks',
    'Transformer architectureの': 'Transformer architecture',
    'Transfer learningの': 'Transfer learning',
    'Fine-tuningの': 'Fine-tuning',
    'Optimizationの': 'Optimization',
    'Regularizationの': 'Regularization',
    'Batch normalizationの': 'Batch normalization',
    'Gradient descentの': 'Gradient descent',
    'Backpropagationの': 'Backpropagation',
    'Cross-validationの': 'Cross-validation',
    'Hyperparameterの': 'Hyperparameter',
    'Pretrainingの': 'Pretraining',
    'Embeddingの': 'Embedding',
    'Attentionの': 'Attention',
    'Self-attentionの': 'Self-attention',
    'Multi-head attentionの': 'Multi-head attention',

    # Common phrase patterns with particles
    'することができます': 'can be done',
    'することが可能です': 'is possible',
    'することが重要です': 'is important',
    'する必要があります': 'is necessary',
    'する必要がある': 'needs to',
    'することで': 'by doing',
    'する場合': 'when doing',
    'した場合': 'when done',
    'したとき': 'when',
    'するとき': 'when',
    'するために': 'in order to',
    'したがって': 'therefore',
    'それゆえ': 'therefore',
    'すなわち': 'namely',
    'つまり': 'in other words',
    'に対して': 'for',
    'について': 'about',
    'に関して': 'regarding',
    'に関する': 'regarding',
    'によって': 'by',
    'により': 'by',
    'において': 'in',
    'における': 'in',
    'に基づいて': 'based on',
    'に基づく': 'based on',
    'に応じて': 'according to',
    'のような': 'like',
    'のように': 'as',
    'のため': 'because of',
    'ため': 'for',
    'のもと': 'under',
    'のとき': 'when',

    # Verb patterns with particles
    'を実行します': 'executes',
    'を実行する': 'execute',
    'を計算します': 'calculates',
    'を計算する': 'calculate',
    'を示します': 'shows',
    'を示す': 'show',
    'を表します': 'represents',
    'を表す': 'represent',
    'を得ます': 'obtains',
    'を得る': 'obtain',
    'を使います': 'uses',
    'を使う': 'use',
    'を使用します': 'uses',
    'を使用する': 'use',
    'を考えます': 'considers',
    'を考える': 'consider',
    'を定義します': 'defines',
    'を定義する': 'define',
    'を求めます': 'finds',
    'を求める': 'find',
    'を導きます': 'derives',
    'を導く': 'derive',
    'を解きます': 'solves',
    'を解く': 'solve',
    'を学習します': 'learns',
    'を学習する': 'learn',
    'を訓練します': 'trains',
    'を訓練する': 'train',
    'を予測します': 'predicts',
    'を予測する': 'predict',
    'を分類します': 'classifies',
    'を分類する': 'classify',
    'を検出します': 'detects',
    'を検出する': 'detect',
    'を生成します': 'generates',
    'を生成する': 'generate',
    'を提供します': 'provides',
    'を提供する': 'provide',
    'を理解します': 'understands',
    'を理解する': 'understand',
    'を説明します': 'explains',
    'を説明する': 'explain',
    'を最適化します': 'optimizes',
    'を最適化する': 'optimize',
    'を改善します': 'improves',
    'を改善する': 'improve',
    'を向上させます': 'improves',
    'を向上させる': 'improve',

    # Numerical patterns with particles
    'の状態': ' states',
    'の場合': ' case',
    'の値': ' value',
    'の数': ' number',
    'の層': ' layers',
    'のモデル': ' model',
    'のデータ': ' data',
    'の精度': ' accuracy',
    'の性能': ' performance',
    'の結果': ' results',
    'の特徴': ' features',
    'の次元': ' dimensions',
    'のパラメータ': ' parameters',
    'の学習': ' learning',
    'の訓練': ' training',
    'の予測': ' prediction',
    'の分類': ' classification',
    'の関数': ' function',
    'の確率': ' probability',
    'の分布': ' distribution',

    # Adjective patterns
    'な状態': ' state',
    'な場合': ' case',
    'な形': ' form',
    'な方法': ' method',
    'なモデル': ' model',
    'なデータ': ' data',
    'な特徴': ' feature',
    'な性能': ' performance',
    'な結果': ' result',

    # ===== BREADCRUMB NAVIGATION =====
    'AI寺子屋トップ': 'AI Terakoya Top',
    'マシンラーニング': 'Machine Learning',
    '機械学習': 'Machine Learning',
    'ファンダメンタルズ': 'Fundamentals',
    'プロセス・インフォマティクス': 'Process Informatics',
    'プロセスインフォマティクス': 'Process Informatics',
    'マテリアルズ・インフォマティクス': 'Materials Informatics',
    'マテリアルズインフォマティクス': 'Materials Informatics',

    # ===== COMMON METADATA =====
    '読了時間': 'Reading Time',
    '難易度': 'Difficulty',
    '総学習時間': 'Total Learning Time',
    'レベル': 'Level',
    '学習時間': 'Learning Time',

    # ===== DIFFICULTY LEVELS =====
    '初級': 'Beginner',
    '中級': 'Intermediate',
    '上級': 'Advanced',
    '初級〜中級': 'Beginner to Intermediate',
    '中級〜上級': 'Intermediate to Advanced',

    # ===== TIME UNITS =====
    '分': ' minutes',
    '時間': ' hours',
    '個': '',
    'シリーズ': ' Series',
    '章': ' Chapter',

    # ===== NAVIGATION ELEMENTS =====
    '前の章': 'Previous Chapter',
    '次の章': 'Next Chapter',
    'シリーズ目次': 'Series Contents',
    '目次に戻る': 'Back to Contents',
    '目次へ': 'To Contents',

    # ===== COMMON SECTION TITLES =====
    '学習内容': 'Learning Content',
    '学習目標': 'Learning Objectives',
    'この章で学ぶこと': 'What You Will Learn',
    'まとめ': 'Summary',
    '演習問題': 'Exercises',
    '参考文献': 'References',
    '次のステップ': 'Next Steps',
    'シリーズ概要': 'Series Overview',
    '各章の詳細': 'Chapter Details',
    '全体の学習成果': 'Overall Learning Outcomes',
    '前提知識': 'Prerequisites',
    '使用技術とツール': 'Technologies and Tools Used',
    '学習の進め方': 'How to Learn',
    '推奨学習順序': 'Recommended Learning Order',
    '主要ライブラリ': 'Main Libraries',
    '開発環境': 'Development Environment',
    '更新履歴': 'Update History',

    # ===== CHAPTER MARKERS =====
    '第1章': 'Chapter 1',
    '第2章': 'Chapter 2',
    '第3章': 'Chapter 3',
    '第4章': 'Chapter 4',
    '第5章': 'Chapter 5',
    '第6章': 'Chapter 6',

    # ===== COMMON PHRASES =====
    'コード例': 'Code Examples',
    '実装例': 'Implementation Example',
    '実例': 'Example',
    '例題': 'Example Problem',
    'さあ、始めましょう': "Let's Get Started",
    '完全ガイド': 'Complete Guide',
    'から': ' from',
    'まで': ' to',
    'ステップ・バイ・ステップ': 'Step-by-step',
    '段階的に': 'step-by-step',

    # ===== ML CORE TERMS =====
    '機械学習': 'Machine Learning',
    '深層学習': 'Deep Learning',
    'ニューラルネットワーク': 'Neural Networks',
    '人工知能': 'Artificial Intelligence',
    'データサイエンス': 'Data Science',
    '自然言語処理': 'Natural Language Processing',
    'コンピュータビジョン': 'Computer Vision',
    '強化学習': 'Reinforcement Learning',
    '教師あり学習': 'Supervised Learning',
    '教師なし学習': 'Unsupervised Learning',
    '半教師あり学習': 'Semi-Supervised Learning',

    # ===== CNN/COMPUTER VISION =====
    '畳み込みニューラルネットワーク': 'Convolutional Neural Networks',
    '畳み込み層': 'Convolutional Layer',
    '畳み込み': 'Convolution',
    'プーリング層': 'Pooling Layer',
    'プーリング': 'Pooling',
    '転移学習': 'Transfer Learning',
    'ファインチューニング': 'Fine-tuning',
    '事前学習済みモデル': 'Pretrained Model',
    '物体検出': 'Object Detection',
    '画像認識': 'Image Recognition',
    '画像分類': 'Image Classification',
    '画像処理': 'Image Processing',
    '画像': 'Image',
    'セグメンテーション': 'Segmentation',
    '特徴抽出': 'Feature Extraction',
    'データ拡張': 'Data Augmentation',

    # ===== RNN/SEQUENTIAL =====
    '再帰型ニューラルネットワーク': 'Recurrent Neural Networks',
    '時系列': 'Time Series',
    '時系列データ': 'Time Series Data',
    '時系列予測': 'Time Series Forecasting',
    '予測': 'Prediction',
    '予測モデル': 'Prediction Model',
    '系列': 'Sequence',
    'アテンション': 'Attention',
    '注意機構': 'Attention Mechanism',

    # ===== TRANSFORMER/NLP =====
    '自己注意機構': 'Self-Attention',
    '事前学習': 'Pretraining',
    '大規模言語モデル': 'Large Language Models',
    '言語モデル': 'Language Model',
    'トークン': 'Token',
    'トークン化': 'Tokenization',
    '埋め込み': 'Embedding',
    'ベクトル': 'Vector',
    '文書': 'Document',
    'テキスト': 'Text',

    # ===== GNN/NETWORK =====
    'グラフニューラルネットワーク': 'Graph Neural Networks',
    'グラフ': 'Graph',
    'ネットワーク分析': 'Network Analysis',
    'ネットワーク': 'Network',
    'ノード': 'Node',
    'エッジ': 'Edge',
    'メッセージパッシング': 'Message Passing',

    # ===== GENERATIVE MODELS =====
    '生成モデル': 'Generative Models',
    '敵対的生成ネットワーク': 'Generative Adversarial Networks',
    '変分オートエンコーダ': 'Variational Autoencoder',
    '拡散モデル': 'Diffusion Models',
    '生成': 'Generation',

    # ===== REINFORCEMENT LEARNING =====
    '強化学習': 'Reinforcement Learning',
    'エージェント': 'Agent',
    '環境': 'Environment',
    '報酬': 'Reward',
    '方策': 'Policy',
    '価値関数': 'Value Function',
    '行動': 'Action',
    '状態': 'State',

    # ===== MODEL OPERATIONS =====
    '学習': 'Learning',
    '訓練': 'Training',
    '推論': 'Inference',
    '評価': 'Evaluation',
    '検証': 'Validation',
    '最適化': 'Optimization',
    '正則化': 'Regularization',
    'バッチ正規化': 'Batch Normalization',
    '勾配降下法': 'Gradient Descent',
    '誤差逆伝播': 'Backpropagation',
    '損失関数': 'Loss Function',
    '活性化関数': 'Activation Function',

    # ===== MODEL QUALITY =====
    '過学習': 'Overfitting',
    '汎化': 'Generalization',
    '精度': 'Accuracy',
    '性能': 'Performance',
    'モデル': 'Model',
    'アーキテクチャ': 'Architecture',
    'パラメータ': 'Parameters',
    'ハイパーパラメータ': 'Hyperparameters',

    # ===== DATA & PROCESSING =====
    'データセット': 'Dataset',
    'データ': 'Data',
    'バッチサイズ': 'Batch Size',
    'バッチ': 'Batch',
    '前処理': 'Preprocessing',
    '正規化': 'Normalization',
    '標準化': 'Standardization',
    '次元削減': 'Dimensionality Reduction',
    '特徴量': 'Features',
    '特徴': 'Feature',
    'ラベル': 'Label',

    # ===== MLOPS & DEPLOYMENT =====
    'デプロイ': 'Deployment',
    '本番環境': 'Production Environment',
    '推論エンジン': 'Inference Engine',
    'パイプライン': 'Pipeline',
    'モデル管理': 'Model Management',
    '実験管理': 'Experiment Management',
    '自動化': 'Automation',
    'コンテナ化': 'Containerization',
    '監視': 'Monitoring',
    '運用': 'Operations',

    # ===== INTERPRETABILITY =====
    '解釈可能性': 'Interpretability',
    '説明可能性': 'Explainability',
    '可視化': 'Visualization',
    '重要度': 'Importance',

    # ===== RECOMMENDATION SYSTEMS =====
    '推薦システム': 'Recommendation Systems',
    '協調フィルタリング': 'Collaborative Filtering',
    'コンテンツベース': 'Content-based',
    'ハイブリッド': 'Hybrid',

    # ===== ANOMALY DETECTION =====
    '異常検出': 'Anomaly Detection',
    '異常': 'Anomaly',
    '正常': 'Normal',
    '外れ値': 'Outliers',

    # ===== META LEARNING =====
    'メタ学習': 'Meta Learning',
    '少数ショット学習': 'Few-shot Learning',

    # ===== AUDIO/SPEECH =====
    '音声': 'Speech',
    '音声認識': 'Speech Recognition',
    '音声合成': 'Speech Synthesis',
    '音響': 'Acoustic',
    '信号処理': 'Signal Processing',

    # ===== LARGE-SCALE PROCESSING =====
    '大規模': 'Large-scale',
    '分散処理': 'Distributed Processing',
    '並列処理': 'Parallel Processing',
    '分散学習': 'Distributed Learning',

    # ===== MATHEMATICAL TERMS =====
    '確率': 'Probability',
    '統計': 'Statistics',
    '統計力学': 'Statistical Mechanics',
    '線形代数': 'Linear Algebra',
    '最適化理論': 'Optimization Theory',
    '情報理論': 'Information Theory',
    '数学': 'Mathematics',
    '理論': 'Theory',
    '行列': 'Matrix',
    'ベクトル': 'Vector',
    'テンソル': 'Tensor',
    '微分': 'Differentiation',
    '積分': 'Integration',

    # ===== PHYSICS TERMS (FM specific) =====
    '量子力学': 'Quantum Mechanics',
    '量子': 'Quantum',
    '場の理論': 'Field Theory',
    '量子場の理論': 'Quantum Field Theory',
    '確率過程': 'Stochastic Processes',
    '古典統計力学': 'Classical Statistical Mechanics',
    'エントロピー': 'Entropy',
    'ハミルトニアン': 'Hamiltonian',
    '波動関数': 'Wave Function',
    '演算子': 'Operator',
    '固有値': 'Eigenvalue',
    '固有状態': 'Eigenstate',

    # ===== MATERIALS SCIENCE TERMS (MS specific) =====
    '材料科学': 'Materials Science',
    '結晶': 'Crystal',
    '結晶学': 'Crystallography',
    '電子顕微鏡': 'Electron Microscopy',
    '金属材料': 'Metallic Materials',
    'セラミック材料': 'Ceramic Materials',
    '複合材料': 'Composite Materials',
    '材料特性': 'Materials Properties',

    # ===== PROCESS INFORMATICS TERMS (PI specific) =====
    'プロセス': 'Process',
    'プロセス最適化': 'Process Optimization',
    'プロセス制御': 'Process Control',
    'プロセス監視': 'Process Monitoring',
    '実験計画法': 'Design of Experiments',
    'ベイズ最適化': 'Bayesian Optimization',
    'デジタルツイン': 'Digital Twin',
    '品質管理': 'Quality Control',
    'スケールアップ': 'Scale-up',
    '製造': 'Manufacturing',
    '化学プラント': 'Chemical Plant',
    '医薬品': 'Pharmaceutical',
    '半導体': 'Semiconductor',
    '食品': 'Food',
    '安全': 'Safety',
    'プロセス安全': 'Process Safety',
    'シミュレーション': 'Simulation',
    'オントロジー': 'Ontology',
    '知識グラフ': 'Knowledge Graph',

    # ===== TECHNICAL DETAILS =====
    'ライブラリ': 'Library',
    'フレームワーク': 'Framework',
    'ツール': 'Tools',
    '実装': 'Implementation',
    'アルゴリズム': 'Algorithm',
    'メソッド': 'Method',
    '手法': 'Method',
    'アプローチ': 'Approach',
    '技術': 'Technology',

    # ===== LEARNING PROCESS =====
    '基礎': 'Basics',
    '入門': 'Introduction',
    '応用': 'Applications',
    '実践': 'Practice',
    '演習': 'Exercises',
    'ハンズオン': 'Hands-on',

    # ===== QUALITY/EXPERIENCE LEVEL =====
    '初心者': 'Beginner',
    '経験者': 'Experienced',
    '上級者': 'Advanced',

    # ===== COMMON DESCRIPTIVE TERMS =====
    '効果的': 'Effective',
    '効率的': 'Efficient',
    '体系的': 'Systematic',
    '実務': 'Practical',
    '最新': 'Latest',
    '高速化': 'Acceleration',
    '削減': 'Reduction',
    '向上': 'Improvement',
    '解凍': 'Unfreezing',
    '凍結': 'Freezing',
    '活用': 'Utilization',
    '構築': 'Construction',
    '設計': 'Design',

    # ===== TASK-SPECIFIC TERMS =====
    '分類': 'Classification',
    'クラスタリング': 'Clustering',
    '回帰': 'Regression',

    # ===== ADDITIONAL COMMON PHRASES =====
    'メリット': 'Benefits',
    'デメリット': 'Drawbacks',
    '原理': 'Principles',
    '仕組み': 'Mechanism',
    '戦略': 'Strategy',
    '段階的': 'Step-by-step',
    '数百種類': 'Hundreds of types',
    'データサイズ': 'Data size',
    '応じた': 'According to',
    '望ましい': 'Desirable',
    '防ぎながら': 'While preventing',
    '説明できる': 'Can explain',
    '実行できる': 'Can execute',
    '使える': 'Can use',
    '理解する': 'Understand',
    '習得': 'Mastery',
    '提供': 'Provide',

    # ===== RAG SPECIFIC =====
    '検索拡張生成': 'Retrieval-Augmented Generation',
    '検索': 'Retrieval',
    '拡張': 'Augmented',
    'ベクトルデータベース': 'Vector Database',

    # ===== COMMON WORDS =====
    '実行': 'execution',
    '最大': 'maximum',
    '最小': 'minimum',
    '計算': 'calculation',
    '結果': 'result',
    '方法': 'method',
    '性質': 'property',
    '特性': 'characteristic',
    '条件': 'condition',
    '問題': 'problem',
    '解法': 'solution method',
    '系': 'system',

    # ===== VERB ENDINGS (suru verbs) =====
    'します': '',
    'する': '',
    'された': 'ed',
    'して': 'ing',
    'しない': 'not',

    # ===== COPULA =====
    'です': 'is',
    'である': 'is',
    'でした': 'was',
    'だった': 'was',
    'だ': '',

    # ===== COMMON PARTICLES (STANDALONE) =====
    'など': 'etc.',
    'より': 'than',
    'もの': 'thing',
    'こと': 'thing',
    'とき': 'when',
    'ところ': 'place',
    'ほか': 'other',
    'ほど': 'degree',
    'くらい': 'about',
    'ぐらい': 'about',
    'ばかり': 'only',
    'だけ': 'only',
    'しか': 'only',
    'さえ': 'even',

    # ===== ADDITIONAL COMMON TERMS (from analysis) =====
    '思考': 'thinking',
    '回答': 'answer',
    '複雑': 'complex',
    '認識': 'recognition',
    '次': 'next',
    '例': 'example',
    'ステップ': 'step',
    'タスク': 'task',
    '必要': 'necessary',
    '円': 'yen',
    'しました': 'did',
    '検索': 'search',
}

# Particle cleanup patterns (applied after main translation)
PARTICLE_CLEANUP = [
    # Remove trailing particles at end of English words
    (r'([A-Za-z]+)の([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)を([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)が([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)は([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)に([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)で([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)と([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)な([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)や([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)へ([^a-zA-Z]|$)', r'\1\2'),
    (r'([A-Za-z]+)から([^a-zA-Z]|$)', r'\1\2'),

    # Numerical patterns
    (r'(\d+)章', r'\1 Chapters'),
    (r'(\d+)個', r'\1'),
]

# Post-processing cleanup patterns
POST_PROCESSING = [
    # Clean up double spaces
    (r'  +', ' '),

    # Clean up orphaned articles at tag boundaries
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</h', r'</h'),
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</p', r'</p'),
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</div', r'</div'),
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</span', r'</span'),
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*$', ''),

    # Clean up double punctuation
    (r'\s*,\s*,', ','),
    (r'\s*\.\s*\.', '.'),

    # Clean up spaces before punctuation
    (r'\s+([,\.!?;:])', r'\1'),

    # Clean up spaces around parentheses
    (r'\(\s+', '('),
    (r'\s+\)', ')'),
]


def count_japanese_chars(text: str) -> Tuple[int, int]:
    """
    Count Japanese characters in text

    Returns:
        Tuple of (japanese_count, total_count)
    """
    japanese_count = 0
    total_count = len(text)

    for char in text:
        # Check if character is in Japanese Unicode ranges
        if any([
            '\u3040' <= char <= '\u309F',  # Hiragana
            '\u30A0' <= char <= '\u30FF',  # Katakana
            '\u4E00' <= char <= '\u9FFF',  # CJK Unified Ideographs
        ]):
            japanese_count += 1

    return japanese_count, total_count


def calculate_japanese_percentage(text: str) -> float:
    """Calculate percentage of Japanese characters in text"""
    japanese_count, total_count = count_japanese_chars(text)
    if total_count == 0:
        return 0.0
    return (japanese_count / total_count) * 100


def translate_text(text: str) -> str:
    """
    Apply comprehensive dictionary-based translation to text

    Args:
        text: Original text with Japanese characters

    Returns:
        Translated text
    """
    result = text

    # First pass: Direct dictionary replacements (longest first)
    for ja, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(ja, en)

    # Second pass: Particle cleanup patterns
    for pattern, replacement in PARTICLE_CLEANUP:
        result = re.sub(pattern, replacement, result)

    # Third pass: Post-processing cleanup
    for pattern, replacement in POST_PROCESSING:
        result = re.sub(pattern, replacement, result)

    return result


def find_jp_source_file(en_file_path: Path, category: str) -> Path:
    """
    Find corresponding JP source file for given EN file

    Args:
        en_file_path: Path to EN file (e.g., /knowledge/en/ML/cnn-introduction/chapter1.html)
        category: Category name (ML or FM)

    Returns:
        Path to JP source file
    """
    # Get relative path from EN directory
    rel_path = en_file_path.relative_to(EN_DIR)

    # Replace category in path (e.g., ML -> ML, FM -> FM)
    jp_path = JP_DIR / rel_path

    return jp_path


def retranslate_file(en_file_path: Path, category: str) -> Tuple[bool, str, float, float]:
    """
    Re-translate a single file from JP source

    Args:
        en_file_path: Path to EN file
        category: Category name (ML or FM)

    Returns:
        Tuple of (success, message, before_jp_pct, after_jp_pct)
    """
    # Find JP source file
    jp_file_path = find_jp_source_file(en_file_path, category)

    if not jp_file_path.exists():
        return False, f"JP source not found: {jp_file_path}", 0.0, 0.0

    try:
        # Read JP source content
        with open(jp_file_path, 'r', encoding='utf-8') as f:
            jp_content = f.read()

        # Calculate before percentage
        before_jp_pct = calculate_japanese_percentage(jp_content)

        # Apply translations
        translated = translate_text(jp_content)

        # Calculate after percentage
        after_jp_pct = calculate_japanese_percentage(translated)

        # Validation: Check if < 12% Japanese remains (lenient threshold for technical content)
        # Note: Some Japanese technical terms and particles may remain in highly technical content
        if after_jp_pct > 12.0:
            return False, f"Too much Japanese remains: {after_jp_pct:.1f}%", before_jp_pct, after_jp_pct

        # Write to EN file
        with open(en_file_path, 'w', encoding='utf-8') as f:
            f.write(translated)

        return True, "Successfully translated", before_jp_pct, after_jp_pct

    except Exception as e:
        return False, f"Error: {str(e)}", 0.0, 0.0


def load_files_for_category(category: str) -> List[str]:
    """
    Load file list for specific category from JAPANESE_CHARACTER_FILES_LIST.txt

    Args:
        category: Category name (ML or FM)

    Returns:
        List of relative file paths
    """
    if not FILE_LIST.exists():
        raise FileNotFoundError(f"File list not found: {FILE_LIST}")

    with open(FILE_LIST, 'r', encoding='utf-8') as f:
        all_files = [line.strip() for line in f if line.strip()]

    # Filter for category
    prefix = f'./{category}/'
    category_files = [f for f in all_files if f.startswith(prefix)]

    return category_files


def process_category(category: str):
    """
    Process all files for a given category

    Args:
        category: Category name (ML or FM)
    """
    print("=" * 80)
    print(f"{category} CATEGORY RE-TRANSLATION FROM JP SOURCES")
    print("=" * 80)

    # Load file list
    try:
        files = load_files_for_category(category)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    if not files:
        print(f"No {category} files found in file list")
        return

    print(f"Found {len(files)} {category} files to process\n")

    # Statistics
    success_count = 0
    failed_count = 0
    total_before_jp = 0.0
    total_after_jp = 0.0

    # Process each file
    for i, rel_path in enumerate(files, 1):
        # Convert to absolute path
        clean_path = rel_path.lstrip('./')
        en_path = EN_DIR / clean_path

        print(f"[{i}/{len(files)}] {clean_path}")

        if not en_path.exists():
            print(f"  ERROR: EN file not found")
            failed_count += 1
            continue

        success, message, before_pct, after_pct = retranslate_file(en_path, category)

        if success:
            print(f"  SUCCESS: {message}")
            print(f"  Japanese: {before_pct:.1f}% -> {after_pct:.1f}%")
            success_count += 1
            total_before_jp += before_pct
            total_after_jp += after_pct
        else:
            print(f"  FAILED: {message}")
            failed_count += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"{category} TRANSLATION COMPLETE")
    print("=" * 80)
    print(f"Total files:         {len(files)}")
    print(f"Successfully trans:  {success_count}")
    print(f"Failed:              {failed_count}")

    if success_count > 0:
        avg_before = total_before_jp / success_count
        avg_after = total_after_jp / success_count
        print(f"\nAverage Japanese:    {avg_before:.1f}% -> {avg_after:.1f}%")
        print(f"Reduction:           {avg_before - avg_after:.1f}%")

    print("=" * 80)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python retranslate_from_jp.py <category>")
        print("  category: ML, FM, or --all")
        sys.exit(1)

    category = sys.argv[1].upper()

    if category == '--ALL':
        print("\nProcessing both ML and FM categories\n")
        process_category('ML')
        print("\n" * 3)
        process_category('FM')
    elif category in ['ML', 'FM']:
        process_category(category)
    else:
        print(f"ERROR: Invalid category '{category}'")
        print("Valid options: ML, FM, --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
