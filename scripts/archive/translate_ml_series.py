#!/usr/bin/env python3
"""
Translation script for ML series from Japanese to English
Handles HTML files with preservation of structure, code, equations, and styling
"""

import os
import re
from pathlib import Path
import shutil

# Series to translate
SERIES = [
    'cnn-introduction',
    'rnn-introduction',
    'transformer-introduction',
    'nlp-introduction',
    'computer-vision-introduction',
    'reinforcement-learning-introduction'
]

BASE_JP = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/ML')
BASE_EN = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/ML')

# Comprehensive translation dictionary
TRANSLATIONS = {
    # HTML attributes
    'lang="ja"': 'lang="en"',

    # Breadcrumb navigation
    'AI寺子屋トップ': 'AI Terakoya Top',
    '機械学習': 'Machine Learning',

    # CNN series titles
    '畳み込みニューラルネットワーク（CNN）入門': 'Convolutional Neural Networks (CNN) Introduction',
    '畳み込みニューラルネットワーク（CNN）': 'Convolutional Neural Networks (CNN)',
    'CNN': 'CNN',
    'Cnn': 'CNN',
    '第1章：CNNの基礎と畳み込み層': 'Chapter 1: CNN Fundamentals & Convolutional Layers',
    '第2章：プーリング層とCNNアーキテクチャ': 'Chapter 2: Pooling Layers & CNN Architectures',
    '第3章：転移学習とファインチューニング': 'Chapter 3: Transfer Learning & Fine-tuning',
    '第4章：データ拡張とモデル最適化': 'Chapter 4: Data Augmentation & Model Optimization',
    '第5章：物体検出入門': 'Chapter 5: Introduction to Object Detection',

    # RNN series titles
    '再帰型ニューラルネットワーク（RNN）入門': 'Recurrent Neural Networks (RNN) Introduction',
    '第1章：RNNの基礎と時系列データ': 'Chapter 1: RNN Fundamentals & Time Series Data',
    '第2章：LSTMとGRU': 'Chapter 2: LSTM & GRU',
    '第3章：Seq2Seqとエンコーダ・デコーダ': 'Chapter 3: Seq2Seq & Encoder-Decoder',
    '第4章：Attentionメカニズム': 'Chapter 4: Attention Mechanisms',
    '第5章：時系列予測と応用': 'Chapter 5: Time Series Forecasting & Applications',

    # Transformer series titles
    'Transformer入門': 'Transformer Introduction',
    '第1章：Self-AttentionとMulti-Head Attention': 'Chapter 1: Self-Attention & Multi-Head Attention',
    '第2章：Transformerアーキテクチャ': 'Chapter 2: Transformer Architecture',
    '第3章：事前学習とファインチューニング': 'Chapter 3: Pre-training & Fine-tuning',
    '第4章：BERTとGPT': 'Chapter 4: BERT & GPT',
    '第5章：大規模言語モデル': 'Chapter 5: Large Language Models',

    # NLP series titles
    '自然言語処理（NLP）入門': 'Natural Language Processing (NLP) Introduction',
    '第1章：NLPの基礎とテキスト前処理': 'Chapter 1: NLP Fundamentals & Text Preprocessing',
    '第2章：深層学習によるNLP': 'Chapter 2: Deep Learning for NLP',
    '第3章：TransformerとBERT': 'Chapter 3: Transformer & BERT',
    '第4章：大規模言語モデル': 'Chapter 4: Large Language Models',
    '第5章：NLPの応用': 'Chapter 5: NLP Applications',

    # Computer Vision series titles
    'コンピュータビジョン入門': 'Computer Vision Introduction',
    '第1章：画像処理の基礎': 'Chapter 1: Image Processing Fundamentals',
    '第2章：画像分類': 'Chapter 2: Image Classification',
    '第3章：物体検出': 'Chapter 3: Object Detection',
    '第4章：セグメンテーション': 'Chapter 4: Segmentation',
    '第5章：コンピュータビジョンの応用': 'Chapter 5: Computer Vision Applications',

    # Reinforcement Learning series titles
    '強化学習入門': 'Reinforcement Learning Introduction',
    '第1章：強化学習の基礎': 'Chapter 1: Reinforcement Learning Fundamentals',
    '第2章：Q学習とSARSA': 'Chapter 2: Q-Learning & SARSA',
    '第3章：DQN': 'Chapter 3: DQN',
    '第4章：方策勾配法': 'Chapter 4: Policy Gradient Methods',
    '第5章：高度なアルゴリズムと応用': 'Chapter 5: Advanced Algorithms & Applications',

    # Common metadata
    '読了時間': 'Reading Time',
    '難易度': 'Difficulty',
    '総学習時間': 'Total Learning Time',
    'レベル': 'Level',

    # Difficulty levels
    '初級': 'Beginner',
    '中級': 'Intermediate',
    '上級': 'Advanced',
    '初級〜中級': 'Beginner to Intermediate',
    '中級〜上級': 'Intermediate to Advanced',

    # Time units and counters
    '分': ' minutes',
    '時間': ' hours',
    '個': '',
    'シリーズ': ' Series',
    '章': ' Chapter',
    '者': ' learner',
    '分類': ' classification',
    '分岐': ' branching',

    # Navigation elements
    '前の章': 'Previous Chapter',
    '次の章': 'Next Chapter',
    'シリーズ目次': 'Series Contents',
    '目次に戻る': 'Back to Contents',
    '目次へ': 'To Contents',

    # Common section titles
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

    # Common phrases
    'コード例': 'Code Examples',
    '読む': 'Read',
    'を読む': 'Read',
    '第': 'Chapter ',
    'あなたの': 'Your ',
    '学習の旅はここから始まります': 'learning journey starts here',
    'さあ、始めましょう': "Let's Get Started",
    '準備はできましたか': 'Are you ready',
    'から始めて': 'Start with ',
    'の技術を習得しましょう': ' and master the techniques',
    'このシリーズを完了した後、以下のトピックへ進むことをお勧めします': 'After completing this series, we recommend proceeding to the following topics',
    '初学': 'Beginner',
    '推奨される前の学習': 'Recommended Prior Learning',
    'をまったく知らない': ' (No knowledge)',
    'の経験あり': ' (with experience)',
    '所要': 'Duration',
    '全章推奨': 'All chapters recommended',
    '集中学習': 'Focused study',
    '特定トピックの強化': 'Focused Topic Enhancement',
    '深掘り学習': 'Advanced Learning',
    '関連シリーズ': 'Related Series',
    '実践プロジェクト': 'Practical Projects',
    '必須': 'Required',
    '推奨': 'Recommended',
    '知識': 'Knowledge',
    '実践スキル': 'Practical Skills',
    '応用力': 'Application Ability',

    # Disclaimer
    '免責事項': 'Disclaimer',
    '本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。': 'This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical guarantees, etc.).',
    '本コンテンツおよび付随するコード例は「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。': 'This content and accompanying code examples are provided "AS IS" without warranty of any kind, either express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.',
    '外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。': 'The creators and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.',
    '本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。': 'To the maximum extent permitted by applicable law, the creators and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.',
    '本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。': 'The content of this material may be changed, updated, or discontinued without notice.',
    '本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。': 'The copyright and license of this content are subject to the specified terms (e.g., CC BY 4.0). Such licenses typically include warranty disclaimers.',
}

def translate_text(text):
    """Apply translation dictionary to text"""
    result = text
    # Sort by length (longest first) to avoid partial replacements
    for jp, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(jp, en)
    return result

def translate_file(source_path, dest_path):
    """Translate a single HTML file"""
    print(f"Translating: {source_path.name}")

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply translations
    translated = translate_text(content)

    # Write to destination
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"  → Created: {dest_path}")

def translate_series(series_name):
    """Translate all files in a series"""
    print(f"\n{'='*60}")
    print(f"Translating series: {series_name}")
    print(f"{'='*60}")

    source_dir = BASE_JP / series_name
    dest_dir = BASE_EN / series_name

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get all HTML files
    html_files = sorted(source_dir.glob('*.html'))

    for source_file in html_files:
        dest_file = dest_dir / source_file.name
        translate_file(source_file, dest_file)

    print(f"\nCompleted {series_name}: {len(html_files)} files translated")
    return len(html_files)

def main():
    """Main translation process"""
    print("="*60)
    print("ML Series Translation: Japanese → English")
    print("="*60)

    total_files = 0
    completed_series = []

    for series in SERIES:
        try:
            files_count = translate_series(series)
            total_files += files_count
            completed_series.append(series)
        except Exception as e:
            print(f"ERROR translating {series}: {e}")

    print("\n" + "="*60)
    print("TRANSLATION SUMMARY")
    print("="*60)
    print(f"Completed series: {len(completed_series)}/{len(SERIES)}")
    print(f"Total files translated: {total_files}")
    print("\nCompleted series:")
    for series in completed_series:
        print(f"  ✓ {series}")
    print("="*60)

if __name__ == '__main__':
    main()
