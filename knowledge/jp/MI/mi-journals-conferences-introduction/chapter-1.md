---
title: 第1章：学会・論文誌の基礎知識
chapter_title: 第1章：学会・論文誌の基礎知識
subtitle: 学術出版のエコシステムから査読プロセス、研究倫理まで
reading_time: 10-15分
difficulty: 入門
---

### 🎯 この章の学習目標

  * **レベル1（基本理解）** : 学術出版のエコシステム（学会、論文誌、査読システム）の役割を理解する
  * **レベル1（基本理解）** : 論文誌の種類（専門誌・総合誌、レター誌・フルペーパー誌）と特徴を説明できる
  * **レベル1（基本理解）** : Impact Factor、h-index、被引用数の意味と限界を理解する
  * **レベル1（基本理解）** : オープンアクセス vs 従来型出版の違いを理解する
  * **レベル1（基本理解）** : 査読プロセスの流れ（投稿→査読→リバイス→採択）を説明できる
  * **レベル2（実践スキル）** : Impact Factorとh-indexを計算し、論文誌の影響力を評価できる
  * **レベル2（実践スキル）** : 査読期間の統計分析を行い、投稿先選定に活用できる
  * **レベル2（実践スキル）** : 論文被引用ネットワークを可視化し、研究分野の構造を理解できる
  * **レベル2（実践スキル）** : arXiv APIを使って論文を検索し、プレプリントサーバーを活用できる
  * **レベル3（応用力）** : 研究内容に基づいて適切な論文誌を選択できる
  * **レベル3（応用力）** : 研究倫理（著者権、二重投稿、研究不正）の問題を判断できる
  * **レベル3（応用力）** : 投稿から採択までの戦略を立案できる

## 1.1 学術出版のエコシステム

学術研究の成果は、論文誌（Journal）や学会（Conference）を通じて発表・共有されます。このエコシステムは、研究者、出版社、査読者、学会組織、読者が相互に関わり合う複雑なシステムです。 
    
    
    ```mermaid
    flowchart TD
        A[研究者] -->|投稿| B[論文誌/学会]
        B -->|査読依頼| C[査読者]
        C -->|評価・フィードバック| B
        B -->|採択| D[出版社]
        D -->|論文公開| E[読者/研究コミュニティ]
        E -->|引用・利用| A
        A -->|査読協力| C
        D -->|購読料/OA費用| F[資金提供機関]
        F -->|研究費| A
    
        style A fill:#667eea,color:#fff
        style B fill:#764ba2,color:#fff
        style C fill:#9d4edd,color:#fff
        style D fill:#c44569,color:#fff
        style E fill:#f8b500,color:#fff
        style F fill:#27ae60,color:#fff
    ```

### 学術出版の主要な役割

役割 | 説明 | 担い手  
---|---|---  
**知識の伝播** | 研究成果を研究コミュニティに迅速に共有 | 論文誌、学会、プレプリントサーバー  
**品質保証** | 査読により研究の妥当性・信頼性を担保 | 査読者、編集委員  
**Priority確立** | 研究の優先権（誰が最初に発見したか）を記録 | 論文誌、プレプリントサーバー  
**キャリア評価** | 研究者の業績評価・採用・昇進の基準 | 大学、研究機関、資金提供機関  
**アーカイブ** | 研究成果を長期的に保存・アクセス可能に | 出版社、図書館、データベース  
  
### オープンアクセス vs 従来型出版

学術出版は大きく**従来型出版** （Subscription Model）と**オープンアクセス（OA）出版** に分けられます。 

出版形式 | 仕組み | メリット | デメリット | 代表例  
---|---|---|---|---  
**従来型（購読型）** | 読者・図書館が購読料を支払う | 著者負担なし、老舗誌の権威 | 購読者のみ閲覧可能、可視性低 | Nature Materials、Acta Materialia  
**Gold OA** | 著者が出版費用（APC）を支払い、誰でも無料閲覧 | 可視性最大、引用数増加 | 著者負担大（€3,000-5,000） | npj Computational Materials、InfoMat  
**Green OA** | 著者が論文をリポジトリに自己アーカイブ | 著者負担なし、購読型と併用可 | 公開までエンバーゴ期間（6-12ヶ月） | 機関リポジトリ、ResearchGate  
**Hybrid OA** | 購読型誌で著者がOA費用を支払うとOA化 | 著者が選択可能 | 二重課金問題（Double Dipping） | Computational Materials Science  
**Diamond OA** | 著者・読者双方無料、学会・機関が資金提供 | 著者・読者双方負担なし | 資金確保困難、少数派 | Journal of Cheminformatics  
  
**💡 オープンアクセスの潮流**  
近年、欧州を中心にオープンアクセス義務化が進んでいます。Plan S（2018年）は、公的資金による研究をGold OA誌で出版することを義務付け、2025年までに完全実施を目指しています。日本でもJSPS科研費の研究成果は機関リポジトリでの公開が推奨されています。 

### プレプリントサーバーの役割

プレプリントサーバーは、査読前の論文を公開するプラットフォームです。研究成果を迅速に共有し、Priority確保、早期フィードバック取得、可視性向上を実現します。 

サーバー | 対象分野 | 特徴 | URL  
---|---|---|---  
**arXiv** | 物理、数学、CS、材料科学 | 1991年開始、最大規模（200万件超） | <https://arxiv.org/>  
**ChemRxiv** | 化学、材料化学 | ACS運営、化学分野特化 | <https://chemrxiv.org/>  
**bioRxiv** | 生命科学、バイオマテリアル | 医学・生物学で広く利用 | <https://www.biorxiv.org/>  
  
**⚠️ プレプリント利用時の注意**  
一部の論文誌はプレプリント公開を認めない場合があります。投稿前に必ず投稿規定（Author Guidelines）を確認してください。ただし、Nature系列、Science、Cell、PLOS、Springerなど主要出版社のほとんどはプレプリント公開を認めています。 

## 1.2 論文誌の種類と特徴

### 専門誌 vs 総合誌

分類 | 説明 | メリット | デメリット | 代表例  
---|---|---|---|---  
**専門誌** | 特定分野に特化（MI、材料科学、化学など） | 専門読者に届く、査読者の専門性高い | 影響力範囲が限定的 | npj Computational Materials、Computational Materials Science  
**総合誌（学際誌）** | 広範な科学分野を扱う | 高いVisibility、Impact Factor高い | 採択率極めて低い（5-10%）、専門性薄い | Nature、Science、PNAS  
**材料科学総合誌** | 材料科学全般（金属、セラミック、ポリマーなど） | 材料コミュニティ全体に届く | MIのみの論文は掲載困難な場合あり | Advanced Materials、Acta Materialia  
  
### レター誌 vs フルペーパー誌

論文種別 | 長さ | 査読期間 | 適した研究 | 代表例  
---|---|---|---|---  
**レター（Letter）** | 3-5ページ（3,000-5,000語） | 1-2ヶ月（迅速） | 画期的発見、速報性重視 | Applied Physics Letters、Chemical Communications  
**フルペーパー（Full Paper）** | 8-15ページ（6,000-10,000語） | 2-6ヶ月（標準） | 包括的研究、詳細な解析 | Computational Materials Science、Acta Materialia  
**レビュー（Review）** | 20-50ページ（10,000-30,000語） | 3-9ヶ月（長い） | 分野の総説、招待論文が多い | Materials Today、Progress in Materials Science  
  
### Impact Factorとは

**Impact Factor（IF）** は、Clarivate Analytics社が毎年発表する論文誌の影響力指標です。過去2年間に発表された論文が当該年に平均何回引用されたかを示します。 

$$ \text{Impact Factor}_{\text{2024}} = \frac{\text{2024年の引用数（2022-2023年論文へ）}}{\text{2022-2023年の論文数}} $$ 

#### 計算例

InfoMat誌の2023年Impact Factor = 22.0は、以下を意味します： 

  * 2021-2022年にInfoMatが発表した論文が、2023年に平均22.0回引用された
  * 材料科学分野では非常に高い値（トップ5%）

**💡 Impact Factorの限界**  
IFは便利な指標ですが、以下の限界があります：  
\- **分野依存** : 材料科学（IF 2-40）と数学（IF 0.5-3）は引用文化が異なる  
\- **操作可能性** : レビュー論文を多く掲載するとIFが高くなりやすい  
\- **新興誌の不利** : 2年分のデータでは実績を反映しにくい  
\- **個別論文の質とは無関係** : IF 10の誌にも被引用数0の論文は存在する 

### h-indexと被引用数

**h-index** は、研究者の業績を評価する指標です。「h本以上の論文がそれぞれh回以上引用されている」という定義です。 

$$ h\text{-index} = \max \\{ h \in \mathbb{N} : \text{h本以上の論文が≥h回引用されている} \\} $$ 

#### 具体例

ある研究者の被引用数が [100, 50, 30, 20, 15, 10, 5, 3, 2, 1] の場合： 

  * 30本以上の論文が≥30回引用されている？ → No（20回が3番目）
  * 20本以上の論文が≥20回引用されている？ → No（15回が5番目）
  * 15本以上の論文が≥15回引用されている？ → Yes（5本が≥15回）
  * よって**h-index = 5**

## 1.3 学会の種類と選び方

### 国際学会 vs 国内学会

学会種別 | 特徴 | メリット | デメリット | 代表例  
---|---|---|---|---  
**国際学会** | 世界中から参加、英語発表 | 国際的ネットワーク、最新動向把握 | 費用高（10-20万円）、英語ハードル | MRS、E-MRS、IUMRS-ICAM  
**国内学会** | 日本国内開催、日本語可 | 参加費安（学生2,000-5,000円）、質疑応答しやすい | 国際的認知度低い | 日本材料学会、日本MRS、応用物理学会  
  
### ワークショップ vs フルカンファレンス

形式 | 規模 | 特徴 | 適した参加者  
---|---|---|---  
**ワークショップ** | 小規模（50-200名） | 専門テーマ、議論重視、招待講演多い | 中級者以上、特定テーマに興味がある研究者  
**フルカンファレンス** | 大規模（1,000-10,000名） | 広範なトピック、多数のセッション、企業展示 | 初学者〜全レベル、広く情報収集したい研究者  
  
### 学会参加のメリット

  * **最新研究動向の把握** : 論文より6-12ヶ月早い情報
  * **ネットワーキング** : 同分野の研究者との交流、共同研究機会
  * **フィードバック取得** : 質疑応答で研究の改善点を発見
  * **キャリア構築** : 若手研究者賞、企業からのリクルーティング
  * **プレゼン技術向上** : 発表経験を積むことでスキルアップ

## 1.4 査読プロセス

査読（Peer Review）は、論文の科学的妥当性・新規性・重要性を専門家が評価するプロセスです。学術出版の品質保証の中核を担います。 
    
    
    ```mermaid
    flowchart TD
        A[投稿] --> B{エディターチェック1-2週間}
        B -->|Scope外/明らかな欠陥| C[Desk Reject]
        B -->|審査対象| D[査読者選定1週間]
        D --> E[査読2-8週間]
        E --> F{判定}
        F -->|Accept| G[採択]
        F -->|Minor Revision| H[軽微修正1-2週間]
        F -->|Major Revision| I[大幅修正1-3ヶ月]
        F -->|Reject| J[不採択]
        H --> K[再査読1-2週間]
        I --> L[再査読2-4週間]
        K --> M{再判定}
        L --> M
        M -->|Accept| G
        M -->|Reject| J
        J --> N[他誌へ投稿]
    
        style A fill:#667eea,color:#fff
        style G fill:#27ae60,color:#fff
        style J fill:#e74c3c,color:#fff
        style F fill:#764ba2,color:#fff
        style M fill:#764ba2,color:#fff
    ```

### 査読プロセスの各段階

段階 | 期間 | 内容 | 判定  
---|---|---|---  
**投稿** | - | 論文原稿、図表、カバーレター提出 | -  
**エディターチェック** | 1-2週間 | 論文誌のScopeに合うか、明らかな欠陥がないか確認 | 審査対象 or Desk Reject  
**査読者選定** | 1週間 | 2-4名の専門家に査読依頼 | -  
**査読** | 2-8週間 | 査読者が論文を評価、コメント作成 | -  
**判定** | - | Accept/Minor Revision/Major Revision/Reject | 採択 or 修正要求 or 不採択  
**リバイス（修正）** | 1週間-3ヶ月 | 査読コメントに対応して論文修正 | -  
**再査読** | 1-4週間 | 修正版を査読者が再評価 | Accept or Reject  
  
### 査読判定の種類

  * **Accept（採択）** : 修正なしで掲載決定（稀）
  * **Minor Revision（軽微修正）** : 誤字訂正、図の修正など小さな変更のみ（1-2週間で対応）
  * **Major Revision（大幅修正）** : 追加実験、追加解析、大幅な書き直しが必要（1-3ヶ月で対応）
  * **Reject（不採択）** : 採択不可能（他誌への投稿を推奨）
  * **Reject and Resubmit** : 大幅修正後に再投稿可能（実質Major Revisionと同じ）

**💡 Major Revisionは前向きなサイン**  
Major Revisionは一見ネガティブに見えますが、「修正すれば掲載可能性がある」という前向きな判定です。実際、Major Revisionの約70-80%は最終的に採択されます。査読コメントを丁寧に読み、真摯に対応することが重要です。 

### 査読者の役割

査読者は、以下の観点から論文を評価します： 

  * **新規性（Novelty）** : 既存研究との違い、オリジナリティ
  * **妥当性（Validity）** : 手法の適切性、結果の信頼性
  * **重要性（Significance）** : 研究分野への貢献、インパクト
  * **明瞭性（Clarity）** : 論文の構成、図表の質、文章の読みやすさ
  * **再現性（Reproducibility）** : 手法の詳細度、データ・コードの公開

### リバイス対応の基本

  1. **全コメントに対応** : 無視・見落としは即Reject
  2. **Response Letterを作成** : 各コメントに対する回答を詳細に記述
  3. **変更箇所を明示** : 修正版でハイライトまたは行番号を明記
  4. **反論は丁寧に** : 査読者の誤解を指摘する場合も礼儀正しく
  5. **追加実験は迅速に** : Major Revisionの期限（通常1-3ヶ月）を守る

### リジェクト後の対処法

  * **査読コメントを活用** : 不採択でもコメントは論文改善に有用
  * **冷静に分析** : なぜRejectされたか（新規性不足、手法の問題など）を特定
  * **ステップダウン戦略** : IF 10誌でReject → IF 5-7誌に投稿
  * **大幅改訂** : 査読コメントを反映して論文を改善
  * **別誌へ投稿** : 同じ内容の二重投稿は厳禁、修正後に別誌へ

## 1.5 研究倫理と著者権

### 著者の順序とクレジット

論文著者の順序は、貢献度を反映します。分野や研究室によって慣習が異なりますが、以下が一般的です： 

著者位置 | 役割 | 貢献内容  
---|---|---  
**筆頭著者（First Author）** | 主実験者、論文執筆者 | 実験・計算の実施、データ解析、論文執筆  
**第二著者** | 副実験者、サポート | 一部実験、データ解析補助、論文校閲  
**中間著者** | 技術支援、議論参加 | 特定技術提供、実験協力、議論への貢献  
**最終著者（Last Author）** | 責任著者（PI）、研究指導者 | 研究構想、資金獲得、研究指導、論文校閲  
**Corresponding Author** | 連絡担当著者 | 査読対応、問い合わせ窓口（通常は筆頭or最終著者）  
  
**💡 Contributorship Statement**  
近年、多くの論文誌が**Contributorship Statement** （著者貢献記述）を要求しています。各著者がどの部分に貢献したかを明示することで、著者権の透明性を高めています。  
例: "A.S. designed research; B.T. performed experiments; C.U. analyzed data; A.S. and D.V. wrote the paper." 

### 二重投稿の禁止

**二重投稿（Duplicate Submission）** は、同じ内容の論文を複数の論文誌に同時に投稿する行為で、学術出版倫理で厳しく禁止されています。 

#### 禁止される行為

  * 同じ論文を複数誌に同時投稿
  * 査読中の論文を別誌に投稿
  * 既出版論文と実質的に同じ内容を別誌に投稿

#### 許容される行為

  * プレプリントサーバー（arXiv等）への投稿と論文誌への投稿の併用
  * 学会発表と論文投稿の併用
  * Reject後の別誌への投稿（修正後）
  * 異なる内容の論文を複数誌に投稿

### 研究不正の防止

不正行為 | 定義 | 例 | 罰則  
---|---|---|---  
**捏造（Fabrication）** | 存在しないデータを作成 | 実験していないデータを掲載 | 論文撤回、研究費停止、解雇  
**改ざん（Falsification）** | データを都合よく変更 | 外れ値の恣意的削除、画像の加工 | 論文撤回、研究費停止、解雇  
**盗用（Plagiarism）** | 他者の文章・アイデアを出典明記せず使用 | 論文のコピペ、引用なしの図表転載 | 論文撤回、著作権侵害訴訟  
**サラミ出版** | 1つの研究を不必要に複数論文に分割 | 同じデータセットで5本の論文 | 論文誌からの警告、業績評価減点  
  
**⚠️ 研究不正は絶対に行わない**  
研究不正は、研究者生命を終わらせる重大な倫理違反です。2014年のSTAP細胞事件では、論文撤回、研究費返還、研究機関の解体など、甚大な影響がありました。不正を疑われる行為（データの不適切な処理、引用漏れなど）も避け、透明性の高い研究を心がけましょう。 

### COI（利益相反）開示

**COI（Conflict of Interest, 利益相反）** は、研究結果の解釈に影響を与える可能性のある個人的・金銭的関係です。多くの論文誌でCOI開示が義務付けられています。 

#### 開示すべきCOI

  * 企業からの研究資金提供
  * 企業の株式保有、役員就任
  * 特許出願・保有
  * 講演料、コンサルタント料の受領
  * 家族の企業関係

#### COI開示の例

"Author A is a shareholder of XYZ Corporation. Author B received consulting fees from ABC Inc. The other authors declare no conflicts of interest." 

## 1.6 コード例で学ぶ定量的分析

### コード例1: Impact Factor計算シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_impact_factor(citations_2022_2023, papers_2022_2023, year=2024):
        """
        Impact Factorを計算
    
        Parameters:
        -----------
        citations_2022_2023 : int
            2024年に2022-2023年論文が受けた引用数
        papers_2022_2023 : int
            2022-2023年の論文数
        year : int
            計算対象年
    
        Returns:
        --------
        float : Impact Factor
        """
        if papers_2022_2023 == 0:
            return 0.0
        return citations_2022_2023 / papers_2022_2023
    
    # 例: InfoMat誌のIF計算
    citations = 2200  # 2024年の引用数
    papers = 100      # 2022-2023年の論文数
    if_2024 = calculate_impact_factor(citations, papers)
    print(f"Impact Factor 2024: {if_2024:.1f}")
    
    # 複数誌のIF比較
    journals = ['InfoMat', 'npj Comp Mat', 'Comp Mat Sci', 'Acta Mat', 'Digital Disc']
    ifs = [22.0, 9.7, 3.8, 9.4, 4.1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(journals, ifs, color=['#667eea', '#764ba2', '#9d4edd', '#c44569', '#f8b500'])
    plt.xlabel('Impact Factor', fontsize=12)
    plt.title('MI関連論文誌のImpact Factor比較（2023年）', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    for i, (j, if_val) in enumerate(zip(journals, ifs)):
        plt.text(if_val + 0.5, i, f'{if_val}', va='center', fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # 出力例:
    # Impact Factor 2024: 22.0
    

### コード例2: h-index計算スクリプト
    
    
    import numpy as np
    
    def calculate_h_index(citations):
        """
        h-indexを計算
    
        Parameters:
        -----------
        citations : list of int
            各論文の被引用数のリスト
    
        Returns:
        --------
        int : h-index
        """
        # 降順ソート
        citations_sorted = sorted(citations, reverse=True)
    
        h_index = 0
        for i, c in enumerate(citations_sorted, start=1):
            if c >= i:
                h_index = i
            else:
                break
        return h_index
    
    # 例: ある研究者の論文被引用数
    citations = [100, 50, 30, 20, 15, 10, 8, 5, 3, 2, 1, 1, 0, 0]
    h = calculate_h_index(citations)
    print(f"h-index: {h}")
    print(f"解釈: {h}本以上の論文が{h}回以上引用されている")
    
    # 可視化
    import matplotlib.pyplot as plt
    
    citations_sorted = sorted(citations, reverse=True)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(citations_sorted)+1), citations_sorted, color='#667eea', alpha=0.7, label='被引用数')
    plt.axhline(y=h, color='#e74c3c', linestyle='--', linewidth=2, label=f'h-index = {h}')
    plt.axvline(x=h, color='#e74c3c', linestyle='--', linewidth=2)
    plt.xlabel('論文番号（被引用数降順）', fontsize=12)
    plt.ylabel('被引用数', fontsize=12)
    plt.title('h-index可視化', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 出力例:
    # h-index: 8
    # 解釈: 8本以上の論文が8回以上引用されている
    

### コード例3: 査読期間の統計分析（pandas）
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプルデータ: 論文誌別の査読期間（週）
    data = {
        'Journal': ['InfoMat', 'npj Comp Mat', 'Comp Mat Sci', 'Acta Mat',
                    'Digital Disc', 'Nature Mat', 'Adv Funct Mat', 'MRS Bull'],
        'Review_Time_Min': [6, 8, 8, 10, 4, 8, 10, 12],
        'Review_Time_Max': [16, 20, 24, 28, 8, 24, 28, 32],
        'Avg_Review_Time': [10, 12, 14, 16, 6, 14, 18, 20]
    }
    
    df = pd.DataFrame(data)
    df['Review_Time_Range'] = df['Review_Time_Max'] - df['Review_Time_Min']
    
    print("=== 査読期間統計 ===")
    print(df.to_string(index=False))
    print(f"\n平均査読期間: {df['Avg_Review_Time'].mean():.1f}週")
    print(f"最速: {df.loc[df['Avg_Review_Time'].idxmin(), 'Journal']} ({df['Avg_Review_Time'].min()}週)")
    print(f"最遅: {df.loc[df['Avg_Review_Time'].idxmax(), 'Journal']} ({df['Avg_Review_Time'].max()}週)")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['Avg_Review_Time'], color='#667eea', alpha=0.7)
    ax.errorbar(df['Avg_Review_Time'], y_pos,
                xerr=[df['Avg_Review_Time']-df['Review_Time_Min'],
                      df['Review_Time_Max']-df['Avg_Review_Time']],
                fmt='none', ecolor='#764ba2', capsize=5, linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Journal'])
    ax.set_xlabel('査読期間（週）', fontsize=12)
    ax.set_title('MI関連論文誌の査読期間比較', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 出力例:
    # === 査読期間統計 ===
    #          Journal  Review_Time_Min  Review_Time_Max  Avg_Review_Time  Review_Time_Range
    #          InfoMat                6               16               10                 10
    #    npj Comp Mat                8               20               12                 12
    # ...
    # 平均査読期間: 13.8週
    # 最速: Digital Disc (6週)
    # 最遅: MRS Bull (20週)
    

### コード例4: 論文被引用ネットワーク可視化（NetworkX）
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # サンプルデータ: 論文間の引用関係
    citations = [
        ('Paper A', 'Paper B'),  # A が B を引用
        ('Paper A', 'Paper C'),
        ('Paper B', 'Paper D'),
        ('Paper C', 'Paper D'),
        ('Paper C', 'Paper E'),
        ('Paper D', 'Paper F'),
        ('Paper E', 'Paper F'),
        ('Paper F', 'Paper G'),
    ]
    
    # ネットワークグラフ作成
    G = nx.DiGraph()
    G.add_edges_from(citations)
    
    # 被引用数（入次数）を計算
    in_degrees = dict(G.in_degree())
    node_sizes = [v * 500 + 500 for v in in_degrees.values()]
    
    # 可視化
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='#667eea', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='#764ba2',
                           arrows=True, arrowsize=20, width=2, alpha=0.6)
    plt.title('論文被引用ネットワーク（ノードサイズ = 被引用数）', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 最も引用された論文
    most_cited = max(in_degrees, key=in_degrees.get)
    print(f"最も引用された論文: {most_cited} ({in_degrees[most_cited]}回)")
    
    # 出力例:
    # 最も引用された論文: Paper D (2回)
    

### コード例5: ジャーナル選択決定木（scikit-learn）
    
    
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt
    
    # サンプルデータ: 研究タイプと推奨ジャーナル
    data = {
        'Novelty': [3, 2, 1, 3, 2, 1, 3, 2],  # 1: 低, 2: 中, 3: 高
        'Speed': [1, 3, 2, 1, 2, 3, 1, 3],    # 1: 迅速, 2: 標準, 3: 長期OK
        'Interdisciplinary': [1, 0, 0, 1, 1, 0, 0, 1],  # 0: 専門, 1: 学際
        'OA_Budget': [1, 0, 1, 0, 1, 0, 1, 0],  # 0: なし, 1: あり
        'Journal': ['Nature Mat', 'Comp Mat Sci', 'Acta Mat', 'InfoMat',
                    'npj Comp Mat', 'J Alloys', 'Digital Disc', 'Adv Funct Mat']
    }
    
    df = pd.DataFrame(data)
    
    # 特徴量とラベル
    X = df[['Novelty', 'Speed', 'Interdisciplinary', 'OA_Budget']]
    y = df['Journal']
    
    # 決定木モデル
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # 可視化
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=['Novelty', 'Speed', 'Interdisciplinary', 'OA_Budget'],
              class_names=clf.classes_, filled=True, rounded=True, fontsize=10)
    plt.title('ジャーナル選択決定木', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 予測例
    new_research = [[2, 2, 0, 1]]  # 新規性中、標準速度、専門、OA予算あり
    predicted_journal = clf.predict(new_research)
    print(f"推奨ジャーナル: {predicted_journal[0]}")
    
    # 出力例:
    # 推奨ジャーナル: npj Comp Mat
    

### コード例6: オープンアクセス vs 従来型のコスト比較
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # コスト設定（€, ユーロ）
    journals = ['InfoMat\n(Gold OA)', 'npj Comp Mat\n(Gold OA)',
                'Digital Disc\n(Gold OA)', 'Acta Mat\n(Hybrid)',
                'Comp Mat Sci\n(購読型)']
    apc_costs = [4500, 3690, 2900, 0, 0]  # Article Processing Charge
    subscription_costs = [0, 0, 0, 200, 300]  # 年間購読料（個人）
    
    # 可視化
    x = np.arange(len(journals))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, apc_costs, width, label='APC（著者負担）',
                   color='#667eea')
    bars2 = ax.bar(x + width/2, subscription_costs, width,
                   label='購読料（読者負担）', color='#764ba2')
    
    ax.set_xlabel('論文誌', fontsize=12)
    ax.set_ylabel('費用（€）', fontsize=12)
    ax.set_title('オープンアクセス vs 従来型のコスト比較', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(journals)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 値ラベル
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'€{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 総コスト分析
    print("=== 5年間の総コスト（論文3本/年の場合）===")
    for j, apc, sub in zip(journals, apc_costs, subscription_costs):
        total_author = apc * 3 * 5  # 著者負担
        total_reader = sub * 5       # 読者負担
        print(f"{j.replace(chr(10), ' ')}: 著者€{total_author} + 読者€{total_reader} = €{total_author + total_reader}")
    

### コード例7: プレプリント vs 正式出版のタイムライン比較
    
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from datetime import datetime, timedelta
    
    # タイムライン設定
    start_date = datetime(2024, 1, 1)
    
    # プレプリント経路
    preprint_events = [
        ('論文執筆完了', 0),
        ('arXiv投稿', 7),
        ('arXiv公開', 8),
        ('論文誌投稿', 14),
        ('査読完了', 84),
        ('リバイス提出', 112),
        ('採択', 126),
        ('論文誌公開', 140),
    ]
    
    # 従来型（プレプリントなし）
    traditional_events = [
        ('論文執筆完了', 0),
        ('論文誌投稿', 14),
        ('査読完了', 84),
        ('リバイス提出', 112),
        ('採択', 126),
        ('論文誌公開', 140),
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # プレプリント経路
    for i, (event, days) in enumerate(preprint_events):
        date = start_date + timedelta(days=days)
        ax1.plot([days, days], [0, 1], 'o-', color='#667eea', markersize=10, linewidth=2)
        ax1.text(days, 1.15, event, ha='center', fontsize=10, rotation=15)
        if i > 0:
            prev_days = preprint_events[i-1][1]
            ax1.plot([prev_days, days], [1, 1], '-', color='#764ba2', linewidth=2)
    
    ax1.axvspan(8, 140, alpha=0.2, color='green', label='公開期間（arXiv）')
    ax1.set_ylim(-0.5, 2)
    ax1.set_xlim(-10, 150)
    ax1.set_xlabel('日数', fontsize=12)
    ax1.set_title('プレプリント活用経路（arXiv + 論文誌）', fontsize=14)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_yticks([])
    
    # 従来型経路
    for i, (event, days) in enumerate(traditional_events):
        date = start_date + timedelta(days=days)
        ax2.plot([days, days], [0, 1], 'o-', color='#c44569', markersize=10, linewidth=2)
        ax2.text(days, 1.15, event, ha='center', fontsize=10, rotation=15)
        if i > 0:
            prev_days = traditional_events[i-1][1]
            ax2.plot([prev_days, days], [1, 1], '-', color='#f8b500', linewidth=2)
    
    ax2.axvspan(140, 150, alpha=0.2, color='green', label='公開期間（論文誌のみ）')
    ax2.set_ylim(-0.5, 2)
    ax2.set_xlim(-10, 150)
    ax2.set_xlabel('日数', fontsize=12)
    ax2.set_title('従来型経路（論文誌のみ）', fontsize=14)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    print("=== タイムライン比較 ===")
    print("プレプリント経路: 公開開始8日、論文誌公開140日")
    print("従来型経路: 公開開始140日（132日遅い）")
    print("メリット: Priority確保、早期フィードバック、可視性向上")
    

### コード例8: arXiv APIを使った論文検索スクリプト
    
    
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET
    import pandas as pd
    
    def search_arxiv(query, max_results=10):
        """
        arXiv APIで論文を検索
    
        Parameters:
        -----------
        query : str
            検索キーワード（例: "materials informatics"）
        max_results : int
            最大取得件数
    
        Returns:
        --------
        pd.DataFrame : 検索結果（タイトル、著者、要約、URL）
        """
        base_url = 'http://export.arxiv.org/api/query?'
        search_query = urllib.parse.quote(query)
        url = f"{base_url}search_query=all:{search_query}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
        with urllib.request.urlopen(url) as response:
            xml_data = response.read().decode('utf-8')
    
        # XML解析
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')[:200] + '...'
            link = entry.find('atom:id', ns).text
            published = entry.find('atom:published', ns).text[:10]
    
            results.append({
                'Title': title,
                'Authors': ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else ''),
                'Published': published,
                'Summary': summary,
                'URL': link
            })
    
        return pd.DataFrame(results)
    
    # 使用例
    query = "materials informatics machine learning"
    df = search_arxiv(query, max_results=5)
    
    print("=== arXiv検索結果（最新5件）===")
    for i, row in df.iterrows():
        print(f"\n[{i+1}] {row['Title']}")
        print(f"著者: {row['Authors']}")
        print(f"公開日: {row['Published']}")
        print(f"要約: {row['Summary']}")
        print(f"URL: {row['URL']}")
    
    # CSV保存
    df.to_csv('arxiv_search_results.csv', index=False, encoding='utf-8')
    print("\n検索結果をarxiv_search_results.csvに保存しました。")
    
    # 出力例:
    # === arXiv検索結果（最新5件）===
    # [1] Accelerating Materials Discovery with Machine Learning
    # 著者: John Smith, Jane Doe, et al.
    # 公開日: 2024-10-15
    # 要約: We present a machine learning framework for rapid materials discovery...
    # URL: http://arxiv.org/abs/2410.12345
    

## 1.7 演習問題

#### 演習1（Easy）: 基本用語の理解

**問題** : 以下の用語を説明してください。

  1. Impact Factor（IF）
  2. h-index
  3. オープンアクセス（OA）
  4. プレプリント

解答を表示

**解答例** : 

  * **a) Impact Factor** : 過去2年間の論文が当該年に平均何回引用されたかを示す論文誌の影響力指標。Clarivate Analytics社が毎年発表。
  * **b) h-index** : 研究者の業績評価指標。h本以上の論文がそれぞれh回以上引用されている場合のhの最大値。
  * **c) オープンアクセス（OA）** : 論文を誰でも無料で閲覧できる出版形式。Gold OA（著者が費用負担）、Green OA（リポジトリ公開）などがある。
  * **d) プレプリント** : 査読前の論文を公開するプラットフォーム。arXiv、ChemRxivなどが代表例。Priority確保、早期フィードバック取得が目的。

#### 演習2（Easy）: Impact Factor計算

**問題** : ある論文誌の2022-2023年の論文数は150本、2024年にこれらの論文が受けた引用数は1,200回でした。2024年のImpact Factorを計算してください。

解答を表示

**解答** :  
$$\text{IF}_{2024} = \frac{1200}{150} = 8.0$$  
この論文誌のImpact Factorは**8.0** です。材料科学分野では中〜高程度の影響力を持つ論文誌と言えます。 

#### 演習3（Easy）: h-index計算

**問題** : ある研究者の論文被引用数が [45, 30, 25, 18, 12, 10, 7, 5, 3, 2] の場合、h-indexを計算してください。

解答を表示

**解答** :  
降順ソート済みなので、各論文の被引用数を確認：  
\- 10本以上の論文が≥10回引用？ → Yes（45, 30, 25, 18, 12, 10が6本）  
\- 11本以上の論文が≥11回引用？ → No（7回が7番目）  
\- よって**h-index = 10**  
解釈: 10本以上の論文が10回以上引用されている。 

#### 演習4（Medium）: 査読プロセスの理解

**問題** : 論文投稿から採択までの標準的な査読プロセスを、各段階の期間とともに説明してください。Major Revisionの場合を想定してください。

解答を表示

**解答例** :  
**標準的な査読プロセス（Major Revisionの場合）** :  
1\. **投稿** → 原稿・図表・カバーレター提出  
2\. **エディターチェック（1-2週間）** → Scopeと基本要件確認  
3\. **査読者選定（1週間）** → 2-4名の専門家に依頼  
4\. **査読（2-8週間）** → 査読者が評価・コメント作成  
5\. **判定** → Major Revision（大幅修正要求）  
6\. **リバイス（1-3ヶ月）** → 追加実験・大幅書き直し  
7\. **再査読（2-4週間）** → 修正版を再評価  
8\. **最終判定** → Accept（採択）  
9\. **出版（1-2ヶ月）** → 組版・校正・オンライン公開  
**総期間: 4-9ヶ月**

#### 演習5（Medium）: ジャーナル選択シナリオ

**問題** : 以下の研究をどの論文誌に投稿すべきか、理由とともに答えてください。  
**研究内容** : ベイズ最適化を用いた新規高温超伝導体の発見（実験で検証済み）。画期的な成果で、Nature誌レベルの新規性がある。オープンアクセス費用の予算はなし。  
**候補誌** : (a) Nature Materials, (b) InfoMat, (c) npj Computational Materials, (d) Computational Materials Science

解答を表示

**推奨解答** : **(a) Nature Materials**  
**理由** :  
\- **新規性レベル** : "画期的な成果"とあり、Nature誌レベルの新規性があるため、トップジャーナルを狙うべき  
\- **分野適合性** : 超伝導体は材料科学の重要テーマで、Nature Materialsの中核Scope  
\- **オープンアクセス** : Nature Materialsは購読型（Hybrid OA）なので、著者負担なしで投稿可能  
\- **Impact Factor** : IF 37.2（材料科学分野で最高峰）により、可視性最大化  
\- **キャリア効果** : Nature系列の論文は研究者のキャリアに大きく貢献  
  
**代替案** :  
\- **(b) InfoMat** : MI特化でIF 22.0と高いが、OA費用€4,500が必要（予算なしで不適）  
\- **(c) npj Computational Materials** : MI専門誌だが、画期的成果には専門誌よりも総合誌が適切  
\- **(d) Computational Materials Science** : 漸進的改善向けで、画期的成果には過小評価される  
  
**戦略** : まずNature Materialsに挑戦 → Rejectの場合はAdvanced Materials（IF 27.4）→ 次にInfoMat（予算確保後） 

#### 演習6（Medium）: オープンアクセス vs 従来型の選択

**問題** : あなたはMIの新手法（中程度の新規性）を発表したいと考えています。研究資金にOA費用€3,000が含まれています。以下の2つの論文誌から選択してください。理由も説明してください。  
**選択肢** : (a) npj Computational Materials (Gold OA, IF 9.7, APC €3,690), (b) Computational Materials Science (Hybrid OA, IF 3.8, APC €3,200または購読型€0)

解答を表示

**推奨解答** : **(b) Computational Materials Science（購読型）**  
**理由** :  
\- **予算制約** : OA費用€3,000の予算だが、(a)は€3,690（予算超過690€）、(b)は購読型なら€0で予算内  
\- **新規性レベル** : "中程度の新規性"では、IF 9.7の高ランク誌（npj）は採択ハードルが高い可能性  
\- **専門コミュニティ** : Computational Materials ScienceはMI/計算材料科学の標準誌で、対象読者に確実に届く  
\- **査読期間** : Comp Mat Sciは2-4ヶ月と標準的（npjも同程度）  
\- **長期戦略** : 購読型で投稿 → 採択後にOA化（€3,200）を選択することも可能（予算追加確保後）  
  
**代替案** :  
予算を€700追加確保できる場合は、**(a) npj Computational Materials（OA）** を選択。IF 9.7で可視性が高く、MI専門誌としての権威がある。ただし、採択ハードルが高いため、リスクも考慮。 

#### 演習7（Medium）: プレプリント活用戦略

**問題** : あなたは競争の激しいMI分野で新手法を開発しました。論文執筆が完了し、トップジャーナルに投稿予定ですが、査読に4-6ヶ月かかることが予想されます。プレプリントサーバー（arXiv）の利用を検討すべきか、メリット・デメリットを挙げて判断してください。

解答を表示

**推奨解答** : **プレプリントサーバーの利用を推奨**  
**メリット** :  
\- **Priority確保** : 査読前に公開することで、発見の優先権を確立（競争分野では重要）  
\- **可視性向上** : 論文誌公開の4-6ヶ月前から研究コミュニティに認知される  
\- **早期フィードバック** : コメント・引用により論文改善の機会  
\- **引用数増加** : 早期公開により、論文誌公開時には既に引用されている可能性  
\- **無料** : arXivは無料で公開可能  
  
**デメリット（リスク）** :  
\- **査読前公開** : 査読を経ていないため、誤りがある可能性（ただし修正版をアップロード可能）  
\- **一部誌の制限** : 一部の論文誌はプレプリント公開を認めない（Nature、Science等は認めている）  
\- **アイデア流出** : 競合に手法を真似される可能性（ただしPriority確保により対抗可能）  
  
**戦略** :  
1\. 投稿予定誌のプレプリントポリシーを確認（Nature Materials等は許可）  
2\. 論文誌投稿と同時にarXivに投稿  
3\. arXiv公開URLをカバーレターに記載（Editor向け情報）  
4\. 採択後、論文誌DOIをarXivに追記  
  
**結論** : 競争分野では、プレプリント利用はメリットがデメリットを大きく上回る。積極的に活用すべき。 

#### 演習8（Hard）: 研究倫理ケーススタディ1 - 著者権

**問題** : 以下のシナリオで、誰が著者として適切か判断してください。  
**シナリオ** : あなたは修士学生で、MIによる材料探索の研究を行いました。以下の人物が関与しています。  
\- **A（あなた）** : 実験計画、データ取得、解析、論文執筆  
\- **B（指導教員）** : 研究構想、議論、論文校閲、資金獲得  
\- **C（研究室の先輩）** : 実験手法の教育、初期データ解析のサポート  
\- **D（共同研究先の技術者）** : 特定装置の操作サポート（2日間）  
\- **E（学部生）** : データ整理のアルバイト（1週間）  
誰を著者に含めるべきか、著者順も含めて答えてください。

解答を表示

**推奨解答** : **著者: A（筆頭）, C（第二）, B（最終/責任著者）**  
**理由** :  
\- **A（あなた）** : 筆頭著者。実験・解析・執筆の主担当で最大の貢献  
\- **B（指導教員）** : 最終著者（Corresponding Author）。研究構想・資金獲得・論文校閲で責任著者の役割  
\- **C（先輩）** : 第二著者。実験手法教育と初期解析サポートは実質的貢献に該当  
\- **D（技術者）** : 謝辞（Acknowledgments）のみ。装置操作サポートは技術支援であり著者資格には不十分  
\- **E（学部生）** : 謝辞（Acknowledgments）のみ。データ整理は補助作業であり著者資格には不十分  
  
**著者資格の判断基準（ICMJE基準）** :  
1\. 研究の構想・設計、データ取得・分析・解釈に実質的貢献  
2\. 原稿の執筆または重要な知的内容の批判的校閲  
3\. 出版原稿の最終承認  
4\. 研究のあらゆる部分の正確性・誠実性について責任を負う  
  
**謝辞の記載例** :  
"We thank D (XYZ Corp.) for technical support with the ABC instrument, and E for assistance with data organization." 

#### 演習9（Hard）: 研究倫理ケーススタディ2 - 二重投稿

**問題** : 以下のシナリオで、倫理的に問題があるか判断してください。  
**シナリオ1** : あなたはMIの新手法をNature Materialsに投稿しました。3ヶ月後にRejectされたため、同じ原稿を修正してComputational Materials Scienceに投稿しました。  
**シナリオ2** : あなたはMIの新手法をNature Materialsに投稿中です（査読中）。同時に、同じ内容をInfoMatにも投稿しました。  
**シナリオ3** : あなたはMIの新手法をarXivにプレプリント公開しました。1週間後、同じ原稿をNature Materialsに投稿しました。

解答を表示

**解答** :  
**シナリオ1** : **倫理的に問題なし（推奨される行為）**  
\- Reject後の別誌への投稿は標準的な手順  
\- 修正を加えることでより良い論文になる  
\- 同時投稿ではないため二重投稿に該当しない  
\- **推奨** : Nature Materialsの査読コメントを活用して論文を改善  
  
**シナリオ2** : **重大な倫理違反（二重投稿）**  
\- 査読中の論文を別誌に投稿することは二重投稿に該当  
\- ほとんどの論文誌が明示的に禁止  
\- 発覚した場合、両誌からRejectされ、ブラックリストに載る可能性  
\- 研究者としての信頼を失う重大な不正行為  
\- **正しい対応** : Nature Materialsの結果を待つ → Rejectならば別誌へ投稿  
  
**シナリオ3** : **倫理的に問題なし（推奨される戦略）**  
\- ほとんどの主要誌（Nature、Science、PLOS等）はプレプリント公開を認めている  
\- arXivは査読前公開プラットフォームであり、論文誌への投稿とは別  
\- Priority確保と可視性向上のメリット大  
\- **注意** : 投稿前に必ず投稿規定（Author Guidelines）でプレプリントポリシーを確認  
\- **推奨** : カバーレターにarXiv公開URLを記載して透明性を確保 

#### 演習10（Hard）: 投稿戦略の立案

**問題** : 以下の研究内容に基づいて、投稿戦略（第一志望誌、第二志望誌、第三志望誌）を立案してください。理由も詳しく説明してください。  
**研究内容** : ガウス過程回帰を用いた高温合金の機械的特性予測。予測精度は既存手法より10%向上。実験で検証済み。新規性は中程度。  
**制約条件** :  
\- オープンアクセス費用の予算: €2,500  
\- 出版期限: 6ヶ月以内（学位論文に含めるため）  
\- キャリア目標: 博士課程進学を目指しており、Impact Factorも重視したい

解答を表示

**推奨投稿戦略** :  
  
**第一志望誌: Digital Discovery (Royal Society of Chemistry)**  
\- **IF** : 4.1（中程度だが新興誌で成長中）  
\- **OA費用** : €2,400（予算€2,500内）  
\- **査読期間** : 1-2ヶ月（最速クラス）→ 出版期限6ヶ月を余裕で達成  
\- **分野適合性** : MI/データ駆動科学に特化、ガウス過程回帰は中核テーマ  
\- **新規性** : 中程度の新規性でも採択可能性高い（専門誌のため）  
\- **リスク** : 新興誌（2023年創刊）で認知度は発展途上  
  
**第二志望誌: Computational Materials Science (Elsevier)**  
\- **IF** : 3.8（中程度）  
\- **OA費用** : €0（購読型で投稿、採択後にOA化選択可能）  
\- **査読期間** : 2-4ヶ月（標準的）→ 出版期限6ヶ月はギリギリ達成可能  
\- **分野適合性** : 計算材料科学の標準誌、合金特性予測は典型的Scope  
\- **新規性** : 中程度の新規性で採択可能性高い  
\- **メリット** : 長年の実績と広い読者層  
  
**第三志望誌: Modelling and Simulation in Materials Science and Engineering (IOP)**  
\- **IF** : 2.7（中〜低程度）  
\- **OA費用** : €2,200（予算内）  
\- **査読期間** : 2-4ヶ月  
\- **分野適合性** : 材料モデリング・シミュレーションに特化  
\- **新規性** : 中程度の新規性で採択可能性非常に高い  
\- **メリット** : 安定した採択率、専門コミュニティに確実に届く  
  
**戦略の詳細** :  
1\. **プレプリント公開** : 投稿前にarXivに公開してPriority確保  
2\. **第一志望（Digital Discovery）に投稿** : 迅速査読を期待  
3\. **1ヶ月後に判定** : Accept/Minor Revision → 採択見込み、Major Revision → 対応  
4\. **Rejectの場合** : 即座に第二志望（Comp Mat Sci）へ投稿（査読コメントを反映）  
5\. **第二志望もRejectの場合** : 第三志望（Modelling and Sim）へ投稿（採択ほぼ確実）  
  
**タイムライン予測** :  
\- Digital Discovery → 投稿後1-2ヶ月で判定 → 採択なら3-4ヶ月で出版（期限内）  
\- Reject → Comp Mat Sci → 投稿後2-4ヶ月で判定 → 採択なら5-6ヶ月で出版（ギリギリ期限内）  
\- 再Reject → Modelling and Sim → 確実に採択だが期限オーバーの可能性  
  
**リスク管理** :  
\- Digital Discoveryで1ヶ月以内に判定がなければ、Editorに進捗確認  
\- Major Revisionの場合、追加実験なしで対応可能な範囲か即座に判断  
\- 予算€2,500は第一・第三志望で使用、第二志望は購読型（予算温存） 

## 1.8 参考文献

  1. 日本学術会議（2013）『科学者の行動規範』<https://www.scj.go.jp/ja/info/kohyo/pdf/kohyo-22-s168-1.pdf> （pp. 1-6: 研究倫理の基本原則、pp. 7-9: 著者資格と責任）
  2. Committee on Publication Ethics (COPE) (2017) 'Ethical Guidelines for Peer Reviewers', <https://publicationethics.org/resources/guidelines-new/cope-ethical-guidelines-peer-reviewers> （査読者倫理の国際基準、pp. 1-3: 査読者の責任、pp. 4-6: 利益相反管理）
  3. Nature Portfolio (2024) 'Guide to Authors', <https://www.nature.com/nature/for-authors> （Nature誌の投稿ガイド、pp. 1-5: 投稿準備、pp. 6-10: 査読プロセス、pp. 11-15: 著者責任）
  4. Clarivate Analytics (2024) 'Journal Citation Reports - Impact Factor Calculation', <https://clarivate.com/webofsciencegroup/essays/impact-factor/> （Impact Factor算出方法の公式文書、pp. 1-4: 計算式、pp. 5-8: 限界と注意点）
  5. Björk, B.-C., & Solomon, D. (2015) 'Article processing charges in OA journals: relationship between price and quality', *Scientometrics*, 103(2), pp. 373-385. doi:10.1007/s11192-015-1556-z （オープンアクセス費用と論文誌品質の関係を実証的に分析）
  6. Tennant, J. P., et al. (2016) 'The academic, economic and societal impacts of Open Access: an evidence-based review', *F1000Research*, 5:632. doi:10.12688/f1000research.8460.3 （オープンアクセスの影響を包括的にレビュー、pp. 1-10: 学術的影響、pp. 11-20: 経済的影響、pp. 21-30: 社会的影響）
  7. arXiv.org (2024) 'arXiv Help - Submit an Article', <https://info.arxiv.org/help/submit.html> （arXiv利用ガイド、pp. 1-5: 投稿手順、pp. 6-10: ポリシー、pp. 11-15: ライセンス）

## まとめ

この章では、学術出版のエコシステム、論文誌の種類と特徴、Impact FactorとH-indexの意味、学会の種類と選び方、査読プロセスの流れ、研究倫理と著者権について学びました。これらは研究者として論文を発表し、キャリアを構築していく上で不可欠な基礎知識です。 

次章では、MI専門誌21誌を詳細に解説し、各論文誌のImpact Factor、査読期間、オープンアクセス費用、適した研究タイプ、キャリアステージ別推奨度を紹介します。 

[← シリーズトップへ](<./index.html>) [第2章：MI専門誌21誌 →](<./chapter-2.html>)
