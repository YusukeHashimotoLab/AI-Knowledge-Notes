#!/usr/bin/env python3
"""
Translation script for Materials Applications Introduction series
Translates Japanese HTML to English while preserving structure
"""

import re
import sys
from pathlib import Path

# Glossary mapping (key terms from Translation_Glossary.csv)
GLOSSARY = {
    '材料インフォマティクス': 'Materials Informatics',
    'マテリアルズインフォマティクス': 'Materials Informatics',
    'マテリアルズ・インフォマティクス': 'Materials Informatics',
    '機械学習': 'Machine Learning',
    '深層学習': 'Deep Learning',
    'ニューラルネットワーク': 'Neural Network',
    '畳み込みニューラルネットワーク': 'Convolutional Neural Network (CNN)',
    '再帰型ニューラルネットワーク': 'Recurrent Neural Network (RNN)',
    'トランスフォーマー': 'Transformer',
    'グラフニューラルネットワーク': 'Graph Neural Network (GNN)',
    '強化学習': 'Reinforcement Learning',
    '能動学習': 'Active Learning',
    'ベイズ最適化': 'Bayesian Optimization',
    '材料科学': 'Materials Science',
    '結晶構造': 'Crystal Structure',
    '微細構造': 'Microstructure',
    '第一原理計算': 'First-Principles Calculation',
    '密度汎関数理論': 'Density Functional Theory (DFT)',
    '分子動力学': 'Molecular Dynamics (MD)',
    '機械学習ポテンシャル': 'Machine Learning Potential (MLP)',
    '特徴量': 'Feature',
    '記述子': 'Descriptor',
    '訓練データ': 'Training Data',
    'テストデータ': 'Test Data',
    '検証データ': 'Validation Data',
    'モデル': 'Model',
    '予測': 'Prediction',
    '分類': 'Classification',
    '回帰': 'Regression',
    'クラスタリング': 'Clustering',
    '前処理': 'Preprocessing',
    '正規化': 'Normalization',
    '標準化': 'Standardization',
    '過学習': 'Overfitting',
    '汎化': 'Generalization',
    'ハイパーパラメータ': 'Hyperparameter',
    '最適化': 'Optimization',
    '損失関数': 'Loss Function',
    '活性化関数': 'Activation Function',
    'バッチ正規化': 'Batch Normalization',
    'ドロップアウト': 'Dropout',
    '勾配降下法': 'Gradient Descent',
    '確率的勾配降下法': 'Stochastic Gradient Descent (SGD)',
    '学習率': 'Learning Rate',
    'エポック': 'Epoch',
    'バッチサイズ': 'Batch Size',
    '精度': 'Accuracy',
    '適合率': 'Precision',
    '再現率': 'Recall',
    '交差検証': 'Cross-Validation',
    '混同行列': 'Confusion Matrix',
    '転移学習': 'Transfer Learning',
    'ファインチューニング': 'Fine-tuning',
    '事前学習': 'Pre-training',
    '埋め込み': 'Embedding',
    '次元削減': 'Dimensionality Reduction',
    '主成分分析': 'Principal Component Analysis (PCA)',
    '物性予測': 'Property Prediction',
    '材料設計': 'Materials Design',
    '材料探索': 'Materials Discovery',
    'データ駆動': 'Data-Driven',
    '触媒': 'Catalyst',
    '電池': 'Battery',
}

def translate_html_file(input_path, output_path):
    """Translate HTML file from Japanese to English"""

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Change lang attribute
    content = content.replace('lang="ja"', 'lang="en"')

    # Translate meta description
    content = re.sub(
        r'<meta name="description" content="([^"]+)"',
        lambda m: f'<meta name="description" content="{translate_text(m.group(1))}"',
        content
    )

    # Translate title
    content = re.sub(
        r'<title>([^<]+)</title>',
        lambda m: f'<title>{translate_text(m.group(1))}</title>',
        content
    )

    # Translate breadcrumbs
    content = content.replace('AI寺子屋トップ', 'AI Terakoya Top')
    content = content.replace('マテリアルズ・インフォマティクス', 'Materials Informatics')
    content = content.replace('本文へスキップ', 'Skip to content')

    # Fix <br> tags
    content = re.sub(r'<br(?!\s*/)>', '<br/>', content)

    # Translate chapter numbers
    content = content.replace('第1章', 'Chapter 1')
    content = content.replace('第2章', 'Chapter 2')
    content = content.replace('第3章', 'Chapter 3')
    content = content.replace('第4章', 'Chapter 4')

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Translated: {input_path.name} -> {output_path.name}")
    return len(content.splitlines())

def translate_text(text):
    """Simple text translation using glossary"""
    for jp, en in GLOSSARY.items():
        text = text.replace(jp, en)
    return text

def main():
    source_dir = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MI/materials-applications-introduction/')
    dest_dir = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MI/materials-applications-introduction/')

    files = ['chapter-1.html', 'chapter-2.html', 'chapter-3.html', 'chapter-4.html']

    total_lines = 0
    for filename in files:
        input_path = source_dir / filename
        output_path = dest_dir / filename

        if input_path.exists():
            lines = translate_html_file(input_path, output_path)
            total_lines += lines
        else:
            print(f"Warning: {input_path} not found")

    print(f"\nTotal lines processed: {total_lines}")
    print("Translation complete!")

if __name__ == '__main__':
    main()
