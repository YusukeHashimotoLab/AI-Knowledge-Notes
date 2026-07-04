#!/usr/bin/env python3
"""
Translation script for remaining ML series from Japanese to English
Handles HTML files with preservation of structure, code, equations, and styling
"""

import os
import re
from pathlib import Path
import shutil

# Series to translate (all remaining series)
SERIES = [
    'ai-agents-introduction',
    'anomaly-detection-introduction',
    'generative-models-introduction',
    'gnn-introduction',
    'large-scale-data-processing-introduction',
    'meta-learning-introduction',
    'ml-mathematics-introduction',
    'mlops-introduction',
    'model-deployment-introduction',
    'model-interpretability-introduction',
    'network-analysis-introduction',
    'rag-introduction',
    'recommendation-systems-introduction',
    'speech-audio-introduction',
    'time-series-introduction'
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

    # AI Agents series
    'AIエージェント入門シリーズ': 'AI Agents Introduction Series',
    'AIエージェント入門': 'AI Agents Introduction',
    'AIエージェント': 'AI Agents',
    '第1章：AIエージェント基礎': 'Chapter 1: AI Agent Fundamentals',
    '第2章：ツール使用とFunction Calling': 'Chapter 2: Tool Use & Function Calling',
    '第3章：マルチエージェントシステム': 'Chapter 3: Multi-Agent Systems',
    '第4章：実践的な応用': 'Chapter 4: Practical Applications',

    # Anomaly Detection series
    '異常検知入門シリーズ': 'Anomaly Detection Introduction Series',
    '異常検知入門': 'Anomaly Detection Introduction',
    '異常検知': 'Anomaly Detection',
    '第1章：異常検知の基礎': 'Chapter 1: Anomaly Detection Fundamentals',
    '第2章：統計的手法': 'Chapter 2: Statistical Methods',
    '第3章：機械学習による異常検知': 'Chapter 3: ML-based Anomaly Detection',
    '第4章：深層学習による異常検知': 'Chapter 4: Deep Learning for Anomaly Detection',

    # Generative Models series
    '生成モデル入門シリーズ': 'Generative Models Introduction Series',
    '生成モデル入門': 'Generative Models Introduction',
    '生成モデル': 'Generative Models',
    '第1章：生成モデルの基礎': 'Chapter 1: Generative Model Fundamentals',
    '第2章：VAE（Variational Autoencoder）': 'Chapter 2: VAE (Variational Autoencoder)',
    '第2章：VAE': 'Chapter 2: VAE',
    '第3章：GAN（Generative Adversarial Network）': 'Chapter 3: GAN (Generative Adversarial Network)',
    '第3章：GAN': 'Chapter 3: GAN',
    '第4章：Diffusion Models': 'Chapter 4: Diffusion Models',
    '第5章：生成モデルの応用': 'Chapter 5: Generative Model Applications',

    # GNN series
    'グラフニューラルネットワーク（GNN）入門シリーズ': 'Graph Neural Networks (GNN) Introduction Series',
    'グラフニューラルネットワーク（GNN）入門': 'Graph Neural Networks (GNN) Introduction',
    'グラフニューラルネットワーク': 'Graph Neural Networks',
    'GNN入門': 'GNN Introduction',
    '第1章：グラフデータとGNNの基礎': 'Chapter 1: Graph Data & GNN Fundamentals',
    '第2章：GNNアーキテクチャ': 'Chapter 2: GNN Architectures',
    '第3章：グラフ畳み込み': 'Chapter 3: Graph Convolution',
    '第4章：グラフアテンション': 'Chapter 4: Graph Attention',
    '第5章：GNNの応用': 'Chapter 5: GNN Applications',

    # Large-Scale Data Processing series
    '大規模データ処理入門シリーズ': 'Large-Scale Data Processing Introduction Series',
    '大規模データ処理入門': 'Large-Scale Data Processing Introduction',
    '大規模データ処理': 'Large-Scale Data Processing',
    '第1章：分散処理の基礎': 'Chapter 1: Distributed Processing Fundamentals',
    '第2章：Apache Spark': 'Chapter 2: Apache Spark',
    '第3章：データパイプライン': 'Chapter 3: Data Pipelines',
    '第4章：ストリーミング処理': 'Chapter 4: Stream Processing',
    '第5章：大規模機械学習': 'Chapter 5: Large-Scale Machine Learning',

    # Meta-Learning series
    'メタ学習入門シリーズ': 'Meta-Learning Introduction Series',
    'メタ学習入門': 'Meta-Learning Introduction',
    'メタ学習': 'Meta-Learning',
    '第1章：メタ学習の基礎': 'Chapter 1: Meta-Learning Fundamentals',
    '第2章：MAML': 'Chapter 2: MAML',
    '第3章：Few-Shot Learning': 'Chapter 3: Few-Shot Learning',
    '第4章：メタ学習の応用': 'Chapter 4: Meta-Learning Applications',

    # ML Mathematics series
    '機械学習数学入門シリーズ': 'ML Mathematics Introduction Series',
    '機械学習数学入門': 'ML Mathematics Introduction',
    '機械学習数学': 'ML Mathematics',
    '第1章：線形代数': 'Chapter 1: Linear Algebra',
    '第2章：微積分': 'Chapter 2: Calculus',
    '第3章：確率統計': 'Chapter 3: Probability & Statistics',
    '第4章：最適化': 'Chapter 4: Optimization',
    '第5章：情報理論': 'Chapter 5: Information Theory',

    # MLOps series
    'MLOps入門シリーズ': 'MLOps Introduction Series',
    'MLOps入門': 'MLOps Introduction',
    'MLOps': 'MLOps',
    '第1章：MLOpsの基礎': 'Chapter 1: MLOps Fundamentals',
    '第2章：実験管理とバージョン管理': 'Chapter 2: Experiment Management & Version Control',
    '第3章：パイプライン自動化': 'Chapter 3: Pipeline Automation',
    '第4章：モデル管理': 'Chapter 4: Model Management',
    '第5章：CI/CD for ML': 'Chapter 5: CI/CD for ML',

    # Model Deployment series
    'モデルデプロイメント入門シリーズ': 'Model Deployment Introduction Series',
    'モデルデプロイメント入門': 'Model Deployment Introduction',
    'モデルデプロイメント': 'Model Deployment',
    '第1章：デプロイメントの基礎': 'Chapter 1: Deployment Fundamentals',
    '第2章：モデルサービング': 'Chapter 2: Model Serving',
    '第3章：スケーラブルデプロイメント': 'Chapter 3: Scalable Deployment',
    '第4章：モニタリングと保守': 'Chapter 4: Monitoring & Maintenance',

    # Model Interpretability series
    'モデル解釈性入門シリーズ': 'Model Interpretability Introduction Series',
    'モデル解釈性入門': 'Model Interpretability Introduction',
    'モデル解釈性': 'Model Interpretability',
    '第1章：解釈性の基礎': 'Chapter 1: Interpretability Fundamentals',
    '第2章：特徴量重要度': 'Chapter 2: Feature Importance',
    '第3章：LIME・SHAP': 'Chapter 3: LIME & SHAP',
    '第4章：モデル可視化': 'Chapter 4: Model Visualization',

    # Network Analysis series
    'ネットワーク分析入門シリーズ': 'Network Analysis Introduction Series',
    'ネットワーク分析入門': 'Network Analysis Introduction',
    'ネットワーク分析': 'Network Analysis',
    '第1章：ネットワークの基礎': 'Chapter 1: Network Fundamentals',
    '第2章：ネットワーク指標': 'Chapter 2: Network Metrics',
    '第3章：コミュニティ検出': 'Chapter 3: Community Detection',
    '第4章：ネットワーク予測': 'Chapter 4: Network Prediction',
    '第5章：動的ネットワーク': 'Chapter 5: Dynamic Networks',

    # RAG series
    'RAG入門シリーズ': 'RAG Introduction Series',
    'RAG入門': 'RAG Introduction',
    'RAG（Retrieval-Augmented Generation）入門': 'RAG (Retrieval-Augmented Generation) Introduction',
    '第1章：RAGの基礎': 'Chapter 1: RAG Fundamentals',
    '第2章：検索システム': 'Chapter 2: Retrieval Systems',
    '第3章：RAG実装': 'Chapter 3: RAG Implementation',
    '第4章：RAGの応用': 'Chapter 4: RAG Applications',

    # Recommendation Systems series
    'レコメンドシステム入門シリーズ': 'Recommendation Systems Introduction Series',
    'レコメンドシステム入門': 'Recommendation Systems Introduction',
    'レコメンドシステム': 'Recommendation Systems',
    '第1章：レコメンデーションの基礎': 'Chapter 1: Recommendation Fundamentals',
    '第2章：協調フィルタリング': 'Chapter 2: Collaborative Filtering',
    '第3章：コンテンツベースフィルタリング': 'Chapter 3: Content-Based Filtering',
    '第4章：ハイブリッド手法': 'Chapter 4: Hybrid Methods',

    # Speech/Audio series
    '音声・音響処理入門シリーズ': 'Speech & Audio Processing Introduction Series',
    '音声・音響処理入門': 'Speech & Audio Processing Introduction',
    '音声・音響処理': 'Speech & Audio Processing',
    '第1章：音声処理の基礎': 'Chapter 1: Speech Processing Fundamentals',
    '第2章：音声認識': 'Chapter 2: Speech Recognition',
    '第3章：音声合成': 'Chapter 3: Speech Synthesis',
    '第4章：音響特徴量': 'Chapter 4: Acoustic Features',
    '第5章：音声アプリケーション': 'Chapter 5: Speech Applications',

    # Time Series series
    '時系列分析入門シリーズ': 'Time Series Analysis Introduction Series',
    '時系列分析入門': 'Time Series Analysis Introduction',
    '時系列分析': 'Time Series Analysis',
    '第1章：時系列データの基礎': 'Chapter 1: Time Series Data Fundamentals',
    '第2章：統計的時系列モデル': 'Chapter 2: Statistical Time Series Models',
    '第3章：機械学習による時系列予測': 'Chapter 3: ML-based Time Series Forecasting',
    '第4章：深層学習による時系列分析': 'Chapter 4: Deep Learning for Time Series',
    '第5章：時系列の応用': 'Chapter 5: Time Series Applications',

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
    '分散': ' distributed',

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
    print(f"  Translating: {source_path.name}")

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply translations
    translated = translate_text(content)

    # Write to destination
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"    → Created: {dest_path.name}")

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

    print(f"\n  ✓ Completed {series_name}: {len(html_files)} files translated")
    return len(html_files)

def main():
    """Main translation process"""
    print("="*60)
    print("ML Series Translation: Japanese → English")
    print("Remaining Series Translation")
    print("="*60)

    total_files = 0
    completed_series = []
    series_details = []

    for series in SERIES:
        try:
            files_count = translate_series(series)
            total_files += files_count
            completed_series.append(series)
            series_details.append((series, files_count))
        except Exception as e:
            print(f"  ✗ ERROR translating {series}: {e}")

    print("\n" + "="*60)
    print("TRANSLATION SUMMARY")
    print("="*60)
    print(f"Completed series: {len(completed_series)}/{len(SERIES)}")
    print(f"Total files translated: {total_files}")
    print("\nCompleted series (with file counts):")
    for series, count in series_details:
        print(f"  ✓ {series}: {count} files")
    print("="*60)

if __name__ == '__main__':
    main()
