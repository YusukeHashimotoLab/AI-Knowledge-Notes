---
title: 第1章：RAGの基礎
chapter_title: 第1章：RAGの基礎
---

## 1\. RAGとは何か

### 1.1 RAGの概要

RAG（Retrieval-Augmented Generation）は、大規模言語モデル（LLM）に外部知識を組み込むための手法です。LLMの生成能力と検索システムを組み合わせることで、最新情報や専門知識に基づいた回答を可能にします。

**主な利点:**

  * **最新情報の活用** : モデルの学習データに含まれない情報にアクセス
  * **ハルシネーション削減** : 検索結果に基づく回答で精度向上
  * **コスト効率** : ファインチューニング不要で知識を追加
  * **透明性** : 情報源を明確化し、検証可能な回答を提供

### 1.2 RAGアーキテクチャ

RAGシステムは以下の3つの主要コンポーネントで構成されます：

**RAGパイプライン:**

  1. **インデックス構築** : ドキュメント読み込み → チャンキング → エンベディング → ベクトルDB保存
  2. **検索** : クエリ → エンベディング → 類似ドキュメント検索
  3. **生成** : 検索結果 + クエリ → LLMプロンプト → 回答生成

#### 実装例1: 基本的なRAGアーキテクチャ
    
    
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    
    class SimpleRAG:
        def __init__(self, api_key):
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            self.llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
            self.vectorstore = None
    
        def index_documents(self, file_paths):
            """ドキュメントをインデックス化"""
            documents = []
            for path in file_paths:
                loader = TextLoader(path, encoding='utf-8')
                documents.extend(loader.load())
    
            # チャンキング
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(documents)
    
            # ベクトルストア作成
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            print(f"インデックス化完了: {len(splits)}チャンク")
    
        def query(self, question):
            """質問応答"""
            if not self.vectorstore:
                raise ValueError("ドキュメントが未インデックス")
    
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
    
            result = qa_chain({"query": question})
            return result["result"]
    
    # 使用例
    rag = SimpleRAG(api_key="your-api-key")
    rag.index_documents(["docs/manual.txt", "docs/faq.txt"])
    answer = rag.query("製品の保証期間は？")
    print(answer)

## 2\. ドキュメント処理

### 2.1 ドキュメントローダー

様々な形式のドキュメントを読み込むためのローダーが用意されています：

#### 実装例2: 複数フォーマット対応ローダー
    
    
    from langchain.document_loaders import (
        TextLoader, PDFLoader, CSVLoader,
        UnstructuredMarkdownLoader, UnstructuredHTMLLoader
    )
    import os
    
    class UniversalDocumentLoader:
        """複数フォーマット対応ドキュメントローダー"""
    
        LOADERS = {
            '.txt': TextLoader,
            '.pdf': PDFLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
            '.html': UnstructuredHTMLLoader,
        }
    
        def load_documents(self, directory):
            """ディレクトリ内の全ドキュメント読み込み"""
            documents = []
    
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
    
                    if ext in self.LOADERS:
                        loader_class = self.LOADERS[ext]
                        try:
                            loader = loader_class(file_path)
                            docs = loader.load()
    
                            # メタデータ追加
                            for doc in docs:
                                doc.metadata['source_file'] = file
                                doc.metadata['file_type'] = ext
    
                            documents.extend(docs)
                            print(f"読み込み: {file} ({len(docs)}ドキュメント)")
                        except Exception as e:
                            print(f"エラー ({file}): {e}")
    
            return documents
    
    # 使用例
    loader = UniversalDocumentLoader()
    documents = loader.load_documents("./knowledge_base")
    print(f"総ドキュメント数: {len(documents)}")

### 2.2 メタデータ管理

メタデータを適切に管理することで、検索精度を向上させ、フィルタリングを可能にします。

#### 実装例3: メタデータ強化
    
    
    from datetime import datetime
    from langchain.schema import Document
    import hashlib
    
    class MetadataEnricher:
        """ドキュメントメタデータ強化"""
    
        def enrich_documents(self, documents):
            """メタデータ追加・強化"""
            enriched = []
    
            for doc in documents:
                # 基本メタデータ
                metadata = doc.metadata.copy()
    
                # タイムスタンプ
                metadata['indexed_at'] = datetime.now().isoformat()
    
                # ドキュメント長
                metadata['char_count'] = len(doc.page_content)
                metadata['word_count'] = len(doc.page_content.split())
    
                # ハッシュ値（重複検出用）
                content_hash = hashlib.md5(
                    doc.page_content.encode()
                ).hexdigest()
                metadata['content_hash'] = content_hash
    
                # カテゴリ推定（簡易版）
                metadata['category'] = self._estimate_category(doc.page_content)
    
                enriched.append(Document(
                    page_content=doc.page_content,
                    metadata=metadata
                ))
    
            return enriched
    
        def _estimate_category(self, text):
            """コンテンツからカテゴリ推定"""
            keywords = {
                'technical': ['API', 'コード', '実装', '関数'],
                'business': ['契約', '料金', '販売', 'ビジネス'],
                'support': ['問題', 'エラー', 'トラブル', 'サポート']
            }
    
            text_lower = text.lower()
            scores = {}
    
            for category, terms in keywords.items():
                score = sum(1 for term in terms if term.lower() in text_lower)
                scores[category] = score
    
            return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    # 使用例
    enricher = MetadataEnricher()
    enriched_docs = enricher.enrich_documents(documents)
    
    # メタデータによるフィルタリング
    technical_docs = [
        doc for doc in enriched_docs
        if doc.metadata.get('category') == 'technical'
    ]
    print(f"技術ドキュメント: {len(technical_docs)}件")

## 3\. チャンキング戦略

### 3.1 固定長チャンキング

最もシンプルな方法で、指定した文字数またはトークン数でドキュメントを分割します。

#### 実装例4: 固定長チャンキング
    
    
    from langchain.text_splitter import CharacterTextSplitter
    import tiktoken
    
    class FixedSizeChunker:
        """固定長チャンキング"""
    
        def __init__(self, chunk_size=500, chunk_overlap=50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
        def chunk_by_characters(self, text):
            """文字数ベース分割"""
            splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            return splitter.split_text(text)
    
        def chunk_by_tokens(self, text):
            """トークン数ベース分割"""
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_text(text)
    
        def analyze_chunks(self, chunks):
            """チャンク統計"""
            stats = {
                'total_chunks': len(chunks),
                'avg_length': sum(len(c) for c in chunks) / len(chunks),
                'min_length': min(len(c) for c in chunks),
                'max_length': max(len(c) for c in chunks),
            }
            return stats
    
    # 使用例
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    
    text = """長いドキュメントのテキスト..."""
    chunks = chunker.chunk_by_tokens(text)
    stats = chunker.analyze_chunks(chunks)
    
    print(f"チャンク数: {stats['total_chunks']}")
    print(f"平均長: {stats['avg_length']:.1f}文字")

### 3.2 セマンティックチャンキング

意味的なまとまりを考慮してドキュメントを分割する高度な手法です。

#### 実装例5: セマンティックチャンキング
    
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class SemanticChunker:
        """セマンティックチャンキング"""
    
        def __init__(self, embeddings, similarity_threshold=0.7):
            self.embeddings = embeddings
            self.threshold = similarity_threshold
    
        def chunk_by_similarity(self, text, min_chunk_size=100):
            """類似度ベース分割"""
            # まず文単位に分割
            sentences = self._split_sentences(text)
    
            if len(sentences) <= 1:
                return [text]
    
            # 各文のエンベディング取得
            sentence_embeddings = self.embeddings.embed_documents(sentences)
    
            # 類似度に基づいてグループ化
            chunks = []
            current_chunk = [sentences[0]]
    
            for i in range(1, len(sentences)):
                # 前の文との類似度計算
                sim = cosine_similarity(
                    [sentence_embeddings[i-1]],
                    [sentence_embeddings[i]]
                )[0][0]
    
                if sim >= self.threshold:
                    current_chunk.append(sentences[i])
                else:
                    # 新しいチャンク開始
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = [sentences[i]]
    
            # 最後のチャンク追加
            if current_chunk:
                chunks.append(' '.join(current_chunk))
    
            return chunks
    
        def _split_sentences(self, text):
            """文分割（簡易版）"""
            import re
            sentences = re.split(r'[。！？\n]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    # 使用例
    from langchain.embeddings import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(openai_api_key="your-api-key")
    semantic_chunker = SemanticChunker(embeddings, similarity_threshold=0.75)
    
    text = """機械学習は人工知能の一分野です。データから学習します。
    深層学習はニューラルネットワークを使用します。画像認識に優れています。
    自然言語処理はテキストを扱います。翻訳や要約が可能です。"""
    
    chunks = semantic_chunker.chunk_by_similarity(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"チャンク{i}: {chunk}")

### 3.3 階層的チャンキング

ドキュメントの構造（見出し、段落など）を考慮した階層的な分割を行います。

#### 実装例6: 階層的チャンキング
    
    
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    from typing import List, Dict
    
    class HierarchicalChunker:
        """階層的チャンキング"""
    
        def chunk_markdown(self, markdown_text):
            """Markdown構造に基づく分割"""
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]
    
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            splits = splitter.split_text(markdown_text)
    
            # 階層情報を含むチャンク作成
            hierarchical_chunks = []
            for split in splits:
                chunk = {
                    'content': split.page_content,
                    'metadata': split.metadata,
                    'hierarchy': self._build_hierarchy(split.metadata)
                }
                hierarchical_chunks.append(chunk)
    
            return hierarchical_chunks
    
        def _build_hierarchy(self, metadata: Dict) -> str:
            """階層パス構築"""
            parts = []
            for level in ['H1', 'H2', 'H3']:
                if level in metadata:
                    parts.append(metadata[level])
            return ' > '.join(parts)
    
        def chunk_with_context(self, text, chunk_size=500):
            """親チャンクのコンテキストを保持"""
            from langchain.text_splitter import RecursiveCharacterTextSplitter
    
            # 親チャンク作成
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 3,
                chunk_overlap=0
            )
            parent_chunks = parent_splitter.split_text(text)
    
            # 子チャンク作成（親情報を保持）
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=50
            )
    
            chunks_with_context = []
            for parent_idx, parent in enumerate(parent_chunks):
                child_chunks = child_splitter.split_text(parent)
    
                for child_idx, child in enumerate(child_chunks):
                    chunks_with_context.append({
                        'content': child,
                        'parent_id': parent_idx,
                        'child_id': child_idx,
                        'parent_summary': parent[:200] + '...'  # 親の要約
                    })
    
            return chunks_with_context
    
    # 使用例
    hierarchical_chunker = HierarchicalChunker()
    
    markdown_text = """
    # データサイエンス
    データ分析の基礎を学びます。
    
    ## 統計学
    ### 記述統計
    平均、分散、標準偏差を学習します。
    
    ### 推測統計
    仮説検定と信頼区間について説明します。
    
    ## 機械学習
    ### 教師あり学習
    回帰と分類のアルゴリズムを扱います。
    """
    
    chunks = hierarchical_chunker.chunk_markdown(markdown_text)
    for chunk in chunks:
        print(f"階層: {chunk['hierarchy']}")
        print(f"内容: {chunk['content'][:50]}...")
        print()

**チャンキング戦略の選択:**

  * **固定長** : シンプルで高速、一般的なドキュメントに適用
  * **セマンティック** : 意味的一貫性が重要な場合に使用
  * **階層的** : 構造化されたドキュメント（技術文書、マニュアル）に最適

## まとめ

  * RAGは検索と生成を組み合わせた強力なLLM拡張手法
  * ドキュメント処理では適切なローダーとメタデータ管理が重要
  * チャンキング戦略はユースケースに応じて選択
  * 固定長、セマンティック、階層的の3つのアプローチを理解

[← シリーズ目次](<./index.html>) [第2章へ →](<./chapter2-embeddings.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
