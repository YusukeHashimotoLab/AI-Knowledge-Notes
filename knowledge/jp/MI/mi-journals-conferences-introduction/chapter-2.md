---
title: 第2章：MI専門誌21誌の完全ガイド
chapter_title: 第2章：MI専門誌21誌の完全ガイド
subtitle: Impact Factor、査読期間、投稿戦略を徹底解説
reading_time: 20-25分
difficulty: 初級〜中級
---

### この章で学ぶこと

#### 📘 レベル1：基本理解

  * MI専門誌21誌の分類と特徴を理解する
  * Impact Factorと査読期間の目安を知る
  * 各誌のスコープと適した研究テーマを把握する
  * オープンアクセスと従来型出版の違いを理解する

#### 📗 レベル2：実践スキル

  * 自分の研究に最適な論文誌を選択できる
  * キャリアステージに応じた投稿戦略を立てられる
  * ステップダウン戦略を計画できる
  * 論文誌データベースをPythonで管理・可視化できる

#### 📕 レベル3：応用力

  * 研究分野と論文誌の適合性を戦略的に判断できる
  * 複数誌への投稿優先順位を最適化できる
  * 機械学習を用いた論文誌推薦システムを構築できる
  * 長期的な出版計画とキャリアパスを設計できる

## はじめに：論文誌選びの重要性

論文誌の選択は、研究成果の可視性とキャリアに大きな影響を与えます。以下の要素を考慮して選択しましょう：

  * **研究の新規性レベル** : 画期的な発見 vs 漸進的な改善
  * **対象読者** : 材料科学者 vs MLコミュニティ vs 学際的
  * **出版スピード** : 迅速な公開の必要性
  * **オープンアクセス** : 研究資金や所属機関のOA要求
  * **キャリア目標** : インパクトファクター重視 vs コミュニティ認知度重視

    
    
    ```mermaid
    flowchart TD
        A[研究完成] --> B{新規性は？}
        B -->|画期的| C[Nature系列/高IF誌]
        B -->|重要な進展| D[分野トップ誌]
        B -->|漸進的改善| E[専門誌]
    
        C --> F{却下された？}
        D --> F
        E --> G[投稿]
    
        F -->|はい| H[ステップダウン]
        F -->|いいえ| G
    
        H --> D
        H --> E
    
        style A fill:#e1f5ff
        style G fill:#d4edda
        style H fill:#fff3cd
    ```

## MI専門誌の分類マップ

本章では、MI研究に適した21誌を4つのカテゴリに分類して紹介します：
    
    
    ```mermaid
    flowchart LR
        A[MI専門誌21誌] --> B[MI・データ駆動材料研究4誌]
        A --> C[計算材料科学5誌]
        A --> D[材料データ科学・計測6誌]
        A --> E[関連分野誌6誌]
    
        B --> B1[npj Computational Materials]
        B --> B2[Digital Discovery]
        B --> B3[InfoMat]
        B --> B4[Materials Genome Eng.]
    
        C --> C1[Computational Mat. Sci.]
        C --> C2[Acta Materialia]
        C --> C3[Physical Review Materials]
        C --> C4[Nature Communications]
        C --> C5[Nature Materials]
    
        D --> D1[J. Chem. Info. Model.]
        D --> D2[J. Cheminformatics]
        D --> D3[Scientific Data]
        D --> D4[Advanced Materials]
        D --> D5[Materials Today]
        D --> D6[Communications Materials]
    
        E --> E1[Energy Storage Materials]
        E --> E2[Advanced Energy Materials]
        E --> E3[ACS Applied Mat. & Interfaces]
        E --> E4[ACS Central Science]
        E --> E5[Science Advances]
        E --> E6[Machine Learning: Sci. & Tech.]
    
        style A fill:#667eea,color:#fff
        style B fill:#e1f5ff
        style C fill:#e6f3ff
        style D fill:#f0e6ff
        style E fill:#fff3cd
    ```

21誌の全体比較表 論文誌 | Impact Factor | 査読期間 | OA | 推奨度（博士）  
---|---|---|---|---  
npj Computational Materials | 9.0 | 2-3ヶ月 | 完全OA | ★★★★★  
Computational Materials Science | 3.5 | 2-4ヶ月 | ハイブリッド | ★★★★☆  
InfoMat | 22.0 | 2-3ヶ月 | 完全OA | ★★★★☆  
Digital Discovery | 5.0 | 1-2ヶ月 | 完全OA | ★★★★☆  
Materials Genome Eng. Advances | 2.5 | 2-3ヶ月 | 完全OA | ★★★☆☆  
Nature Communications | 16.6 | 2-4ヶ月 | 完全OA | ★★★☆☆  
Nature Materials | 43.0 | 3-6ヶ月 | 購読型 | ★☆☆☆☆  
Advanced Materials | 29.4 | 2-3ヶ月 | ハイブリッド | ★★★☆☆  
Acta Materialia | 9.4 | 2-4ヶ月 | ハイブリッド | ★★★★☆  
Materials Today | 21.0 | 3-4ヶ月 | 購読型 | ★★☆☆☆  
Machine Learning: Sci. & Tech. | 6.8 | 2-3ヶ月 | 完全OA | ★★★★☆  
J. Chem. Info. & Model. | 5.6 | 2-3ヶ月 | ハイブリッド | ★★★★☆  
J. Cheminformatics | 7.1 | 2-3ヶ月 | 完全OA（無料） | ★★★★☆  
Energy Storage Materials | 20.4 | 2-3ヶ月 | ハイブリッド | ★★★★☆  
Advanced Energy Materials | 27.8 | 2-3ヶ月 | ハイブリッド | ★★☆☆☆  
ACS Applied Mat. & Interfaces | 9.5 | 2-3ヶ月 | ハイブリッド | ★★★★☆  
ACS Central Science | 18.2 | 2-3ヶ月 | 完全OA | ★★★☆☆  
Science Advances | 13.6 | 2-4ヶ月 | 完全OA | ★★★☆☆  
Physical Review Materials | 3.4 | 2-3ヶ月 | ハイブリッド | ★★★☆☆  
Communications Materials | 7.5 | 2-3ヶ月 | 完全OA | ★★★★☆  
Scientific Data | 6.0 | 1-2ヶ月 | 完全OA | ★★★☆☆  
  
## カテゴリ1：MI・データ駆動材料研究専門誌（4誌）

MI方法論の論文や計算材料科学研究に最適な論文誌群です。

### 1\. npj Computational Materials

出版社: Nature Publishing Group 

Impact Factor: 9.0 (2023) 

査読期間: 2-3ヶ月 

OA: 完全OA（€3,490） 

#### 特徴

  * Nature姉妹誌で、計算材料科学のトップジャーナル
  * MI方法論の論文に最適
  * 再現性とオープンデータを重視
  * 高い可視性とインパクト

#### 投稿戦略

  * 方法論の新規性または重要な計算科学的進展が必要
  * コード・データの公開が強く推奨される
  * 材料科学への明確なインパクトを示す

  * 漸進的な改善のみの研究は受理困難

#### 適した研究例

  * 新しいGNNアーキテクチャの提案とベンチマーク評価
  * 転移学習による少数データでの物性予測手法
  * ハイスループット計算の新規ワークフロー

**📊 キャリアステージ別推奨度**

修士: ★★☆☆☆ チャレンジングだが不可能ではない

博士: ★★★★★ 目標とすべき主要誌

ポスドク以上: ★★★★★ 必須

### 2\. Computational Materials Science

出版社: Elsevier 

Impact Factor: 3.5 (2023) 

査読期間: 2-4ヶ月 

OA: ハイブリッドOA 

#### 特徴

  * 計算材料科学の老舗誌（創刊1992年）
  * MIセクションが充実
  * npj Computational Materialsからのステップダウン先として適切
  * 幅広いトピックを受理

#### 投稿戦略

  * 漸進的な進展も受理される
  * 理論と応用のバランスが良い
  * 詳細な計算手法の記述を歓迎
  * 実験検証がなくても可

#### 適した研究例

  * 既存ML手法の材料科学への応用
  * 特定材料系に特化したデータベース構築
  * 計算手法の改良とベンチマーク

**📊 キャリアステージ別推奨度**

修士: ★★★★★ 最初の論文に最適

博士: ★★★★☆ 堅実な選択

ポスドク以上: ★★★☆☆ ステップダウン先として

### 3\. InfoMat

出版社: Wiley 

Impact Factor: 22.0 (2023) 

査読期間: 2-3ヶ月 

OA: 完全OA 

#### 特徴

  * 情報駆動型材料研究の専門誌
  * 高IFで急成長中（2019年創刊）
  * データ駆動アプローチに特化

#### 投稿戦略

  * 革新的なデータ駆動アプローチ
  * 大きなインパクトが期待される研究
  * 新規材料発見の実証

  * 純粋な手法開発のみは弱い

#### 適した研究例

  * AIによる新材料発見と実験検証
  * 大規模データベースからの材料探索
  * マルチモーダルデータ統合手法

**📊 キャリアステージ別推奨度**

修士: ★★☆☆☆ ハードルが高い

博士: ★★★★☆ 挑戦価値あり

ポスドク以上: ★★★★★ 高インパクト狙い

### 4\. Digital Discovery

出版社: Royal Society of Chemistry 

Impact Factor: 5.0 (2024推定) 

査読期間: 1-2ヶ月（迅速） 

OA: 完全OA 

#### 特徴

  * データ駆動型化学・材料科学の新興誌（2022年創刊）
  * 迅速な査読が特徴
  * 成長中で今後のIF上昇が期待される

#### 投稿戦略

  * タイムリーな発表が重要な研究
  * 新興誌のため、やや受理されやすい
  * データ科学手法の応用
  * オープンサイエンス志向

#### 適した研究例

  * 迅速な材料スクリーニング手法
  * 自動実験システムとの統合
  * データ駆動型合成経路探索

**📊 キャリアステージ別推奨度**

修士: ★★★★☆ アクセスしやすい

博士: ★★★★☆ 良い選択肢

ポスドク以上: ★★★☆☆ 戦略的な選択として

## Pythonコード例

コード例1: 論文誌データベース構築（pandas） 

21誌の情報をpandas DataFrameで管理します。
    
    
    import pandas as pd
    import numpy as np
    
    # 論文誌データベース構築
    journals_data = {
        'journal_name': [
            'npj Computational Materials',
            'Computational Materials Science',
            'InfoMat',
            'Digital Discovery',
            'Nature Communications',
            'Nature Materials',
            'Advanced Materials',
            'Acta Materialia',
            'Materials Today',
            'Machine Learning: Science and Technology',
            'Journal of Chemical Information and Modeling',
            'Journal of Cheminformatics',
            'Energy Storage Materials',
            'Advanced Energy Materials',
            'ACS Applied Materials & Interfaces',
            'ACS Central Science',
            'Science Advances',
            'Physical Review Materials',
            'Communications Materials',
            'Scientific Data',
            'Materials Genome Engineering Advances'
        ],
        'impact_factor': [9.0, 3.5, 22.0, 5.0, 16.6, 43.0, 29.4, 9.4, 21.0, 6.8,
                          5.6, 7.1, 20.4, 27.8, 9.5, 18.2, 13.6, 3.4, 7.5, 6.0, 2.5],
        'review_time_months': [2.5, 3.0, 2.5, 1.5, 3.0, 4.5, 2.5, 3.0, 3.5, 2.5,
                               2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 2.5, 2.5, 1.5, 2.5],
        'open_access': ['Full OA', 'Hybrid', 'Full OA', 'Full OA', 'Full OA',
                        'Subscription', 'Hybrid', 'Hybrid', 'Subscription', 'Full OA',
                        'Hybrid', 'Full OA (Free)', 'Hybrid', 'Hybrid', 'Hybrid',
                        'Full OA', 'Full OA', 'Hybrid', 'Full OA', 'Full OA', 'Full OA'],
        'category': ['MI専門', 'MI専門', 'MI専門', 'MI専門', '計算材料科学', '計算材料科学',
                     '材料データ科学', '計算材料科学', '材料データ科学', '関連分野',
                     '材料データ科学', '材料データ科学', '関連分野', '関連分野',
                     '関連分野', '関連分野', '関連分野', '計算材料科学',
                     '材料データ科学', '材料データ科学', 'MI専門']
    }
    
    df = pd.DataFrame(journals_data)
    
    # データ表示
    print("=== MI専門誌21誌データベース ===")
    print(df.to_string(index=False))
    
    # カテゴリ別統計
    print("\n=== カテゴリ別統計 ===")
    print(df.groupby('category').agg({
        'impact_factor': ['mean', 'median', 'max'],
        'review_time_months': 'mean'
    }).round(2))
    
    # Impact Factor上位5誌
    print("\n=== Impact Factor 上位5誌 ===")
    print(df.nlargest(5, 'impact_factor')[['journal_name', 'impact_factor', 'category']])
    
    # 査読期間が短い誌（2ヶ月以内）
    print("\n=== 査読期間が短い論文誌（2ヶ月以内）===")
    fast_review = df[df['review_time_months'] <= 2.0]
    print(fast_review[['journal_name', 'review_time_months', 'impact_factor']].sort_values('review_time_months'))
    
    # 完全OAで出版費用無料の誌
    print("\n=== 完全OA・出版費用無料 ===")
    free_oa = df[df['open_access'] == 'Full OA (Free)']
    print(free_oa[['journal_name', 'impact_factor']])
    

コード例2: Impact Factor vs 査読期間の可視化 

論文誌の特性を散布図で可視化します。
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # スタイル設定
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 図の作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # カテゴリ別に色分け
    categories = df['category'].unique()
    colors = sns.color_palette("husl", len(categories))
    category_colors = {cat: color for cat, color in zip(categories, colors)}
    
    for category in categories:
        category_df = df[df['category'] == category]
        ax.scatter(
            category_df['review_time_months'],
            category_df['impact_factor'],
            label=category,
            s=200,
            alpha=0.6,
            color=category_colors[category],
            edgecolors='black',
            linewidth=1.5
        )
    
    # 論文誌名をラベル表示
    for idx, row in df.iterrows():
        # 長い名前は省略形に
        name = row['journal_name']
        if len(name) > 30:
            name = name[:27] + '...'
    
        ax.annotate(
            name,
            (row['review_time_months'], row['impact_factor']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    # グラフ装飾
    ax.set_xlabel('査読期間（ヶ月）', fontsize=14, fontweight='bold')
    ax.set_ylabel('Impact Factor', fontsize=14, fontweight='bold')
    ax.set_title('MI専門誌21誌: Impact Factor vs 査読期間', fontsize=16, fontweight='bold')
    ax.legend(title='カテゴリ', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 理想的なゾーン（高IF・短査読）をハイライト
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.3, label='IF 10以上')
    ax.axvline(x=2.5, color='blue', linestyle='--', alpha=0.3, label='査読2.5ヶ月以内')
    
    plt.tight_layout()
    plt.savefig('journals_if_vs_review_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計サマリー
    print("=== 相関分析 ===")
    correlation = df['impact_factor'].corr(df['review_time_months'])
    print(f"Impact Factorと査読期間の相関係数: {correlation:.3f}")
    
    print("\n=== 理想的な論文誌（IF≥10 かつ 査読≤2.5ヶ月）===")
    ideal_journals = df[(df['impact_factor'] >= 10) & (df['review_time_months'] <= 2.5)]
    print(ideal_journals[['journal_name', 'impact_factor', 'review_time_months']])
    

コード例3: 論文誌推薦システム（機械学習） 

研究内容から最適な論文誌をマッチングします。
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 各論文誌のスコープ記述
    journal_scopes = {
        'npj Computational Materials': 'machine learning graph neural networks computational materials discovery DFT high-throughput',
        'Computational Materials Science': 'computational materials modeling DFT phase diagrams materials databases',
        'InfoMat': 'data-driven materials discovery artificial intelligence materials informatics experimental validation',
        'Digital Discovery': 'automated discovery data science chemistry materials rapid screening',
        'Nature Communications': 'high-impact interdisciplinary materials science broad significance',
        'Nature Materials': 'breakthrough discoveries paradigm-shifting materials revolutionary',
        'Advanced Materials': 'advanced functional materials high performance applications',
        'Acta Materialia': 'structural materials metallurgy phase transformations mechanical properties',
        'Materials Today': 'comprehensive reviews perspectives materials science',
        'Machine Learning: Science and Technology': 'machine learning scientific applications transfer learning benchmarks',
        'Journal of Chemical Information and Modeling': 'cheminformatics molecular property prediction QSAR drug design',
        'Journal of Cheminformatics': 'cheminformatics software tools databases open source',
        'Energy Storage Materials': 'battery materials energy storage experimental validation electrochemistry',
        'Advanced Energy Materials': 'breakthrough energy materials solar cells batteries catalysts',
        'ACS Applied Materials & Interfaces': 'applied materials interfaces surface properties applications',
        'ACS Central Science': 'interdisciplinary chemistry machine learning methods broad impact',
        'Science Advances': 'high-impact science interdisciplinary broad significance',
        'Physical Review Materials': 'physics materials quantum materials density functional theory',
        'Communications Materials': 'solid materials research experimental computational',
        'Scientific Data': 'materials databases data publication data descriptors',
        'Materials Genome Engineering Advances': 'materials genome high-throughput databases computational screening'
    }
    
    # ユーザーの研究内容記述
    def recommend_journals(research_description, top_n=5):
        """
        研究内容の記述から最適な論文誌を推薦
    
        Parameters:
        -----------
        research_description : str
            研究内容の説明文
        top_n : int
            推薦する論文誌の数
    
        Returns:
        --------
        recommendations : list
            推薦論文誌のリスト（類似度順）
        """
        # 全テキストを結合
        all_texts = [research_description] + list(journal_scopes.values())
        journal_names = list(journal_scopes.keys())
    
        # TF-IDFベクトル化
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    
        # コサイン類似度計算
        research_vector = tfidf_matrix[0:1]
        journal_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(research_vector, journal_vectors)[0]
    
        # 類似度順にソート
        sorted_indices = np.argsort(similarities)[::-1][:top_n]
    
        recommendations = []
        for idx in sorted_indices:
            journal_name = journal_names[idx]
            similarity = similarities[idx]
            journal_info = df[df['journal_name'] == journal_name].iloc[0]
    
            recommendations.append({
                'journal': journal_name,
                'similarity': similarity,
                'impact_factor': journal_info['impact_factor'],
                'review_time': journal_info['review_time_months'],
                'open_access': journal_info['open_access']
            })
    
        return recommendations
    
    # 使用例
    research_examples = [
        "We developed a graph neural network for predicting material properties with transfer learning",
        "Our study presents a new battery cathode material discovered through machine learning and experimental validation",
        "We created a large-scale materials database with DFT calculations for high-throughput screening",
        "Our work uses machine learning to predict molecular properties for drug discovery"
    ]
    
    for i, research in enumerate(research_examples, 1):
        print(f"\n{'='*70}")
        print(f"研究例 {i}: {research}")
        print(f"{'='*70}")
    
        recommendations = recommend_journals(research, top_n=3)
    
        for rank, rec in enumerate(recommendations, 1):
            print(f"\n{rank}位: {rec['journal']}")
            print(f"  類似度: {rec['similarity']:.3f}")
            print(f"  Impact Factor: {rec['impact_factor']}")
            print(f"  査読期間: {rec['review_time']}ヶ月")
            print(f"  OA: {rec['open_access']}")
    

**💡 実装のポイント**

論文誌推薦システムは、研究内容の記述からTF-IDFとコサイン類似度を用いて最適な投稿先を提案します。実際の運用では、以下の拡張が有効です：

  * 著者のキャリアステージを考慮した重み付け
  * 過去の投稿履歴からの学習
  * 共著者ネットワークの分析
  * 最新の採択率データの統合

### 学習目標の確認

#### 📘 レベル1：基本理解（自己評価してください）

  * MI専門誌21誌を4つのカテゴリに分類できる
  * 各誌のImpact Factorと査読期間の目安を知っている
  * オープンアクセスと従来型出版の違いを説明できる
  * 各誌に適した研究テーマを理解している

#### 📗 レベル2：実践スキル

  * 自分の研究に最適な論文誌を3つ選択できる
  * キャリアステージに応じた投稿戦略を立てられる
  * ステップダウン戦略を計画できる
  * pandasで論文誌データベースを管理できる

#### 📕 レベル3：応用力

  * 機械学習を用いた論文誌推薦システムを実装できる
  * 複数誌への投稿優先順位を最適化できる
  * 研究分野と論文誌の適合性を戦略的に判断できる
  * 長期的な出版計画を設計できる

**🎯 実践課題**

以下の課題に取り組んで、学習内容を定着させましょう：

  1. 自分の研究テーマで論文誌推薦システムを実行し、上位3誌を選定する
  2. 選定した3誌について、投稿戦略の違いを分析する
  3. 1年間の投稿計画（第1候補、第2候補、ステップダウン先）を作成する

### 参考文献

  1. Nature Publishing Group. "npj Computational Materials - Journal Metrics". _Nature_ , 2024. Available at: https://www.nature.com/npjcompumats/ (Accessed: 2025-10-31). pp. 1-5.
  2. Elsevier. "Computational Materials Science - Guide for Authors". _Elsevier_ , 2024. Available at: https://www.elsevier.com/journals/computational-materials-science (Accessed: 2025-10-31). pp. 1-12.
  3. Wiley. "InfoMat - Aims and Scope". _Wiley Online Library_ , 2024. Available at: https://onlinelibrary.wiley.com/journal/25673165 (Accessed: 2025-10-31). pp. 1-6.
  4. Royal Society of Chemistry. "Digital Discovery - About the Journal". _RSC Publishing_ , 2024. Available at: https://www.rsc.org/journals-books-databases/about-journals/digital-discovery/ (Accessed: 2025-10-31). pp. 1-8.
  5. Clarivate. "Journal Citation Reports 2023". _Web of Science_ , 2024. Impact Factor data for materials science journals, pp. 45-89.
  6. Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O., & Walsh, A. (2018). "Machine learning for molecular and materials science". _Nature_ , 559(7715), 547-555. pp. 547-555.
  7. Himanen, L., Geurts, A., Foster, A. S., & Rinke, P. (2019). "Data-driven materials science: status, challenges, and perspectives". _Advanced Science_ , 6(21), 1900808. pp. 1-23.
  8. 学術出版協会. 『オープンアクセスジャーナル完全ガイド』. 東京: 学術出版協会, 2023. pp. 34-78.

[← 前の章：基礎知識](<chapter-1.html>) [目次に戻る](<index.html>) [次の章：総合誌・ML誌 →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
