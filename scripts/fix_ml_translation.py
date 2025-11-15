#!/usr/bin/env python3
"""
Enhanced comprehensive translation script for ML category files (119 files)
Handles Japanese particles, verbs, and mixed-language patterns
Includes post-processing for clean output
"""

import os
import re
from pathlib import Path

BASE_DIR = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en')
FILE_LIST = BASE_DIR / 'JAPANESE_CHARACTER_FILES_LIST.txt'

# Comprehensive translation dictionary for ML content
# IMPORTANT: Longer phrases FIRST to avoid partial replacements
TRANSLATIONS = {
    # HTML attributes
    'lang="ja"': 'lang="en"',

    # ===== LONGER PHRASES FIRST (Mixed Language Patterns) =====

    # Common mixed-language patterns with particles
    'Machine Learningの基礎': 'Foundations of Machine Learning',
    'Deep Learningの': 'Deep Learning',
    'Neural Networksの': 'Neural Networks',
    'CNNの': 'CNN',
    'RNNの': 'RNN',
    'Transformerの': 'Transformer',
    'GNNの': 'GNN',

    # Verb endings in mixed contexts
    'を実行します': 'executes',
    'を計算します': 'calculates',
    'を示します': 'shows',
    'を表します': 'represents',
    'を得ます': 'obtains',
    'を使います': 'uses',
    'を考えます': 'considers',
    'を定義します': 'defines',
    'を求めます': 'finds',
    'を導きます': 'derives',
    'を解きます': 'solves',
    'を学習します': 'learns',
    'を訓練します': 'trains',
    'を予測します': 'predicts',
    'を分類します': 'classifies',
    'を検出します': 'detects',
    'を生成します': 'generates',

    # Common phrase patterns
    'することができます': 'can be done',
    'することが可能です': 'is possible',
    'する必要があります': 'is necessary',
    'する場合': 'when doing',
    'した場合': 'when done',
    'したとき': 'when',
    'するとき': 'when',
    'するために': 'in order to',
    'したがって': 'therefore',
    'に対して': 'for',
    'について': 'about',
    'に関して': 'regarding',
    'によって': 'by',
    'において': 'in',
    'における': 'in',
    'に基づいて': 'based on',
    'に応じて': 'according to',
    'のような': 'like',
    'のように': 'as',
    'のため': 'because of',
    'のもと': 'under',
    'のとき': 'when',

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

    # Adjective patterns
    'な状態': ' state',
    'な場合': ' case',
    'な形': ' form',
    'な方法': ' method',
    'なモデル': ' model',
    'なデータ': ' data',
    'な特徴': ' feature',
    'な性能': ' performance',

    # ===== BREADCRUMB NAVIGATION =====
    'AI寺子屋トップ': 'AI Terakoya Top',
    'マシンラーニング': 'Machine Learning',
    '機械学習': 'Machine Learning',

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
    '画像': 'Image',
    'セグメンテーション': 'Segmentation',
    '特徴抽出': 'Feature Extraction',
    'データ拡張': 'Data Augmentation',

    # ===== RNN/SEQUENTIAL =====
    '再帰型ニューラルネットワーク': 'Recurrent Neural Networks',
    '時系列': 'Time Series',
    '時系列データ': 'Time Series Data',
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
    '線形代数': 'Linear Algebra',
    '最適化理論': 'Optimization Theory',
    '情報理論': 'Information Theory',
    '数学': 'Mathematics',
    '理論': 'Theory',
    '行列': 'Matrix',

    # ===== TECHNICAL DETAILS =====
    'ライブラリ': 'Library',
    'フレームワーク': 'Framework',
    'ツール': 'Tools',
    '実装': 'Implementation',
    'アルゴリズム': 'Algorithm',
    'メソッド': 'Method',
    '手法': 'Method',
    'アプローチ': 'Approach',

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

    # ===== SPECIFIC ML LIBRARIES/FRAMEWORKS =====
    '事前学習': 'Pretraining',
    'モデル集': 'Model Collection',

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

    # ===== SPECIFIC TECHNICAL COMBINATIONS =====
    '局所的特徴抽出': 'Local feature extraction',
    '事前学習済み': 'Pretrained',
    '数百種類の': 'Hundreds of',
    '高速学習': 'Fast learning',

    # ===== RAG SPECIFIC =====
    '検索拡張生成': 'Retrieval-Augmented Generation',
    '検索': 'Retrieval',
    '拡張': 'Augmented',
    'ベクトルデータベース': 'Vector Database',

    # ===== NAVIGATION CONTEXT WORDS =====
    'learnerの方': 'learners',
    'with experience': 'with experience',
    'Focused study': 'Focused study',

    # ===== ADDITIONAL ML SERIES NAMES (30 series) =====
    'ai-agents-introduction': 'AI Agents Introduction',
    'anomaly-detection-introduction': 'Anomaly Detection Introduction',
    'cnn-introduction': 'CNN Introduction',
    'computer-vision-introduction': 'Computer Vision Introduction',
    'generative-models-introduction': 'Generative Models Introduction',
    'gnn-introduction': 'GNN Introduction',
    'large-scale-data-processing-introduction': 'Large-scale Data Processing Introduction',
    'meta-learning-introduction': 'Meta Learning Introduction',
    'ml-mathematics-introduction': 'ML Mathematics Introduction',
    'mlops-introduction': 'MLOps Introduction',
    'model-deployment-introduction': 'Model Deployment Introduction',
    'model-interpretability-introduction': 'Model Interpretability Introduction',
    'network-analysis-introduction': 'Network Analysis Introduction',
    'nlp-introduction': 'NLP Introduction',
    'rag-introduction': 'RAG Introduction',
    'recommendation-systems-introduction': 'Recommendation Systems Introduction',
    'reinforcement-learning-introduction': 'Reinforcement Learning Introduction',
    'rnn-introduction': 'RNN Introduction',
    'speech-audio-introduction': 'Speech and Audio Introduction',
    'time-series-introduction': 'Time Series Introduction',
    'transformer-introduction': 'Transformer Introduction',

    # ===== TECHNICAL TERMS (KATAKANA) =====
    'ミクロ': 'micro',
    'マクロ': 'macro',
    'エネルギー': 'energy',

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
    '手法': 'technique',
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
    'ため': 'for',
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
    'まで': 'until',
}

# Additional particle cleanup patterns (applied after main translation)
PARTICLE_CLEANUP = [
    # Remove trailing particles at end of English words
    (r'([A-Za-z]+)の([^a-zA-Z]|$)', r'\1\2'),  # "Wordの " -> "Word "
    (r'([A-Za-z]+)を([^a-zA-Z]|$)', r'\1\2'),  # "Wordを " -> "Word "
    (r'([A-Za-z]+)が([^a-zA-Z]|$)', r'\1\2'),  # "Wordが " -> "Word "
    (r'([A-Za-z]+)は([^a-zA-Z]|$)', r'\1\2'),  # "Wordは " -> "Word "
    (r'([A-Za-z]+)に([^a-zA-Z]|$)', r'\1\2'),  # "Wordに " -> "Word "
    (r'([A-Za-z]+)で([^a-zA-Z]|$)', r'\1\2'),  # "Wordで " -> "Word "
    (r'([A-Za-z]+)と([^a-zA-Z]|$)', r'\1\2'),  # "Wordと " -> "Word "
    (r'([A-Za-z]+)な([^a-zA-Z]|$)', r'\1\2'),  # "Wordな " -> "Word "
    (r'([A-Za-z]+)や([^a-zA-Z]|$)', r'\1\2'),  # "Wordや " -> "Word "
    (r'([A-Za-z]+)へ([^a-zA-Z]|$)', r'\1\2'),  # "Wordへ " -> "Word "

    # Numerical patterns
    (r'(\d+)章', r'\1 Chapters'),  # N章 -> N Chapters
    (r'(\d+)個', r'\1'),  # N個 -> N (remove counter)
]

# Post-processing cleanup patterns
POST_PROCESSING = [
    # Clean up double spaces
    (r'  +', ' '),  # Multiple spaces -> single space

    # Clean up orphaned articles and prepositions at line/tag boundaries
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</h', r'</h'),  # Remove orphaned articles before closing heading tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</p', r'</p'),  # Remove orphaned articles before closing paragraph tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</div', r'</div'),  # Remove orphaned articles before closing div tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</span', r'</span'),  # Remove orphaned articles before closing span tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*$', ''),  # Remove orphaned articles at end of line

    # Clean up double punctuation
    (r'\s*,\s*,', ','),  # Double comma
    (r'\s*\.\s*\.', '.'),  # Double period (but preserve ellipsis ...)

    # Clean up spaces before punctuation
    (r'\s+([,\.!?;:])', r'\1'),  # Remove space before punctuation

    # Clean up spaces around parentheses
    (r'\(\s+', '('),  # Remove space after opening paren
    (r'\s+\)', ')'),  # Remove space before closing paren
]


def translate_text(text):
    """
    Apply comprehensive dictionary-based translation to text with post-processing.

    Args:
        text: Original text with Japanese characters

    Returns:
        Translated text with Japanese replaced by English
    """
    result = text

    # First pass: Direct dictionary replacements (longest first to avoid partial matches)
    for ja, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(ja, en)

    # Second pass: Particle cleanup patterns
    for pattern, replacement in PARTICLE_CLEANUP:
        result = re.sub(pattern, replacement, result)

    # Third pass: Post-processing cleanup
    for pattern, replacement in POST_PROCESSING:
        result = re.sub(pattern, replacement, result)

    return result


def translate_file(file_path):
    """Translate a single HTML file in-place"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply translations
        translated = translate_text(content)

        # Write back if changes were made
        if translated != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            return True
        return False

    except Exception as e:
        print(f"  ✗ Error translating {file_path}: {e}")
        return False


def main():
    """Main translation process for ML category files"""
    print("\n" + "="*70)
    print("ML Category Translation - Enhanced Comprehensive Fix")
    print("Includes: Particles, Verbs, Mixed Patterns, Post-processing")
    print("="*70)

    # Read file list and filter for ML files
    try:
        with open(FILE_LIST, 'r', encoding='utf-8') as f:
            all_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"✗ File list not found: {FILE_LIST}")
        return

    # Filter for ML files (starting with ./ML/)
    ml_files = [f for f in all_files if f.startswith('./ML/')]

    if not ml_files:
        print("✗ No ML files found in the list")
        return

    print(f"\nFound {len(ml_files)} ML files to process")
    print("="*70)

    success_count = 0
    unchanged_count = 0
    error_count = 0

    for i, rel_path in enumerate(ml_files, 1):
        # Convert relative path to absolute
        # Remove leading './' and convert to absolute path
        clean_path = rel_path.lstrip('./')
        abs_path = BASE_DIR / clean_path

        print(f"\n[{i}/{len(ml_files)}] Processing: {clean_path}")

        if not abs_path.exists():
            print(f"  ⚠ File not found: {abs_path}")
            error_count += 1
            continue

        if translate_file(abs_path):
            print(f"  ✓ Translated successfully")
            success_count += 1
        else:
            print(f"  ⊙ No changes needed")
            unchanged_count += 1

    # Summary
    print("\n" + "="*70)
    print("TRANSLATION COMPLETE")
    print("="*70)
    print(f"Total files processed:  {len(ml_files)}")
    print(f"Successfully translated: {success_count}")
    print(f"No changes needed:      {unchanged_count}")
    print(f"Errors:                 {error_count}")
    print("="*70)

    if success_count > 0:
        print("\n✓ Translation completed successfully!")
        print(f"  {success_count} files were updated with English translations")

    if unchanged_count > 0:
        print(f"\n⊙ {unchanged_count} files had no Japanese characters or were already translated")

    if error_count > 0:
        print(f"\n⚠ {error_count} files encountered errors during processing")


if __name__ == '__main__':
    main()
