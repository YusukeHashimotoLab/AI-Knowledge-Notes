---
title: 第3章：高度なRAGテクニック
chapter_title: 第3章：高度なRAGテクニック
---

## 1\. クエリ最適化

### 1.1 Query Decomposition（クエリ分解）

複雑なクエリを複数のサブクエリに分解し、段階的に検索する手法です。

**例:**

元のクエリ: "2023年と2024年のAI市場規模の比較と将来予測"

分解後:

  1. "2023年のAI市場規模"
  2. "2024年のAI市場規模"
  3. "AI市場の将来予測"

#### 実装例1: クエリ分解システム
    
    
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing import List
    
    class DecomposedQuery(BaseModel):
        """分解されたクエリ"""
        sub_queries: List[str] = Field(description="サブクエリのリスト")
        reasoning: str = Field(description="分解の理由")
    
    class QueryDecomposer:
        """クエリ分解システム"""
    
        def __init__(self, llm):
            self.llm = llm
            self.parser = PydanticOutputParser(pydantic_object=DecomposedQuery)
    
        def decompose(self, query: str) -> DecomposedQuery:
            """クエリを分解"""
            template = """以下のクエリを、より単純なサブクエリに分解してください。
            各サブクエリは独立して検索可能である必要があります。
    
            元のクエリ: {query}
    
            {format_instructions}
            """
    
            prompt = ChatPromptTemplate.from_template(template)
            messages = prompt.format_messages(
                query=query,
                format_instructions=self.parser.get_format_instructions()
            )
    
            response = self.llm(messages)
            result = self.parser.parse(response.content)
    
            return result
    
        def search_and_combine(self, query: str, vectorstore, k=3):
            """分解検索と結果統合"""
            # クエリ分解
            decomposed = self.decompose(query)
    
            # 各サブクエリで検索
            all_results = []
            for sub_query in decomposed.sub_queries:
                results = vectorstore.similarity_search(sub_query, k=k)
                all_results.extend(results)
    
            # 重複除去（content hashベース）
            unique_results = []
            seen_contents = set()
    
            for doc in all_results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append(doc)
    
            return {
                'sub_queries': decomposed.sub_queries,
                'reasoning': decomposed.reasoning,
                'results': unique_results
            }
    
    # 使用例
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key="your-api-key")
    decomposer = QueryDecomposer(llm)
    
    query = "機械学習と深層学習の違い、およびそれぞれの応用例"
    result = decomposer.search_and_combine(query, vectorstore)
    
    print("サブクエリ:")
    for i, sq in enumerate(result['sub_queries'], 1):
        print(f"{i}. {sq}")
    
    print(f"\n検索結果: {len(result['results'])}件")

### 1.2 HyDE（Hypothetical Document Embeddings）

クエリから仮想的な回答ドキュメントを生成し、それをクエリとして使用する手法です。

#### 実装例2: HyDE実装
    
    
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    class HyDERetriever:
        """HyDE検索システム"""
    
        def __init__(self, llm, vectorstore, embeddings):
            self.llm = llm
            self.vectorstore = vectorstore
            self.embeddings = embeddings
    
        def generate_hypothetical_document(self, query: str) -> str:
            """仮想ドキュメント生成"""
            template = """以下の質問に対する詳細な回答を書いてください。
            実際の知識がなくても構いません。質問に答える形式で、
            具体的で専門的な内容を含む文章を生成してください。
    
            質問: {query}
    
            回答:"""
    
            prompt = PromptTemplate(template=template, input_variables=["query"])
            response = self.llm(prompt.format(query=query))
    
            return response.content
    
        def search_with_hyde(self, query: str, k=5):
            """HyDE検索"""
            # 仮想ドキュメント生成
            hypothetical_doc = self.generate_hypothetical_document(query)
    
            print(f"仮想ドキュメント:\n{hypothetical_doc[:200]}...\n")
    
            # 仮想ドキュメントで検索
            results = self.vectorstore.similarity_search(hypothetical_doc, k=k)
    
            return results
    
        def hybrid_hyde_search(self, query: str, k=5, alpha=0.5):
            """HyDEと通常検索のハイブリッド
    
            alpha: HyDEの重み（0=通常検索のみ, 1=HyDEのみ）
            """
            # 通常検索
            normal_results = self.vectorstore.similarity_search_with_score(query, k=k)
    
            # HyDE検索
            hyde_doc = self.generate_hypothetical_document(query)
            hyde_results = self.vectorstore.similarity_search_with_score(hyde_doc, k=k)
    
            # スコア統合
            combined_scores = {}
    
            for doc, score in normal_results:
                doc_id = id(doc)
                combined_scores[doc_id] = {
                    'doc': doc,
                    'score': (1 - alpha) * score
                }
    
            for doc, score in hyde_results:
                doc_id = id(doc)
                if doc_id in combined_scores:
                    combined_scores[doc_id]['score'] += alpha * score
                else:
                    combined_scores[doc_id] = {
                        'doc': doc,
                        'score': alpha * score
                    }
    
            # スコアでソート
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:k]
    
            return [item['doc'] for item in sorted_results]
    
    # 使用例
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", openai_api_key="your-api-key")
    hyde_retriever = HyDERetriever(llm, vectorstore, embeddings)
    
    query = "Transformerモデルの注意機構の数学的原理"
    results = hyde_retriever.search_with_hyde(query, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")

## 2\. リランキング

### 2.1 Cross-Encoderリランキング

初回検索結果を、より精度の高いモデルで再評価する手法です。

**Bi-Encoder vs Cross-Encoder:**

  * **Bi-Encoder** : クエリとドキュメントを個別にエンコード（高速、初回検索向け）
  * **Cross-Encoder** : クエリとドキュメントを同時にエンコード（高精度、リランキング向け）

#### 実装例3: Cross-Encoderリランキング
    
    
    from sentence_transformers import CrossEncoder
    import numpy as np
    
    class ReRanker:
        """リランキングシステム"""
    
        def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
            self.cross_encoder = CrossEncoder(model_name)
    
        def rerank(self, query: str, documents: list, top_k: int = 5):
            """Cross-Encoderでリランキング"""
            # クエリとドキュメントのペア作成
            pairs = [[query, doc.page_content] for doc in documents]
    
            # スコア計算
            scores = self.cross_encoder.predict(pairs)
    
            # スコアでソート
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
    
            # Top-K取得
            top_results = scored_docs[:top_k]
    
            return [
                {
                    'document': doc,
                    'score': float(score),
                    'rank': i + 1
                }
                for i, (doc, score) in enumerate(top_results)
            ]
    
        def two_stage_retrieval(self, query: str, vectorstore,
                               first_k: int = 20, final_k: int = 5):
            """2段階検索（初回検索→リランキング）"""
            # 第1段階: ベクトル検索で候補取得
            candidates = vectorstore.similarity_search(query, k=first_k)
            print(f"第1段階: {len(candidates)}件取得")
    
            # 第2段階: Cross-Encoderでリランキング
            reranked = self.rerank(query, candidates, top_k=final_k)
            print(f"第2段階: Top {final_k}件にリランキング")
    
            return reranked
    
    # 使用例
    reranker = ReRanker()
    
    query = "機械学習モデルの評価指標"
    results = reranker.two_stage_retrieval(
        query,
        vectorstore,
        first_k=20,
        final_k=5
    )
    
    print("\nリランキング結果:")
    for result in results:
        print(f"ランク{result['rank']}: スコア {result['score']:.4f}")
        print(f"  {result['document'].page_content[:80]}...\n")

### 2.2 MMR（Maximal Marginal Relevance）

関連性と多様性のバランスを取る検索手法です。

**MMRアルゴリズム:** $$\text{MMR} = \arg\max_{D_i \in R \setminus S} [\lambda \cdot \text{Sim}_1(D_i, Q) - (1-\lambda) \cdot \max_{D_j \in S} \text{Sim}_2(D_i, D_j)]$$ 

λ: 関連性と多様性のバランスパラメータ（0-1）

#### 実装例4: カスタムMMR実装
    
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class MMRRetriever:
        """MMR検索実装"""
    
        def __init__(self, embeddings):
            self.embeddings = embeddings
    
        def mmr_search(self, query: str, documents: list,
                       lambda_param: float = 0.5, k: int = 5):
            """MMR検索
    
            Args:
                query: 検索クエリ
                documents: 候補ドキュメント
                lambda_param: 関連性の重み（0=多様性重視, 1=関連性重視）
                k: 返すドキュメント数
            """
            # エンベディング取得
            query_emb = self.embeddings.embed_query(query)
            doc_texts = [doc.page_content for doc in documents]
            doc_embs = self.embeddings.embed_documents(doc_texts)
    
            # クエリとの類似度
            query_similarity = cosine_similarity(
                [query_emb], doc_embs
            )[0]
    
            # 選択済みドキュメント
            selected_indices = []
            selected_docs = []
    
            # 最初のドキュメント（最も関連性の高いもの）
            first_idx = np.argmax(query_similarity)
            selected_indices.append(first_idx)
            selected_docs.append(documents[first_idx])
    
            # k個になるまで繰り返し
            while len(selected_indices) < k:
                mmr_scores = []
    
                for i, doc in enumerate(documents):
                    if i in selected_indices:
                        mmr_scores.append(-np.inf)
                        continue
    
                    # 関連性スコア
                    relevance = query_similarity[i]
    
                    # 冗長性スコア（選択済みドキュメントとの最大類似度）
                    redundancy = max(
                        cosine_similarity(
                            [doc_embs[i]], [doc_embs[j]]
                        )[0][0]
                        for j in selected_indices
                    )
    
                    # MMRスコア
                    mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
                    mmr_scores.append(mmr)
    
                # 最大MMRスコアのドキュメント選択
                next_idx = np.argmax(mmr_scores)
                selected_indices.append(next_idx)
                selected_docs.append(documents[next_idx])
    
            return selected_docs
    
        def compare_strategies(self, query: str, documents: list, k: int = 5):
            """異なるλ値での比較"""
            strategies = {
                '関連性重視 (λ=0.9)': 0.9,
                'バランス型 (λ=0.5)': 0.5,
                '多様性重視 (λ=0.1)': 0.1
            }
    
            results = {}
            for name, lambda_val in strategies.items():
                docs = self.mmr_search(query, documents, lambda_param=lambda_val, k=k)
                results[name] = docs
    
            return results
    
    # 使用例
    mmr_retriever = MMRRetriever(embeddings)
    
    # 候補ドキュメント取得
    query = "機械学習の評価方法"
    candidates = vectorstore.similarity_search(query, k=20)
    
    # 異なる戦略で比較
    comparison = mmr_retriever.compare_strategies(query, candidates, k=5)
    
    for strategy_name, docs in comparison.items():
        print(f"\n【{strategy_name}】")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content[:60]}...")

## 3\. ハイブリッド検索

### 3.1 ベクトル検索とキーワード検索の融合

BM25などのキーワード検索とベクトル検索を組み合わせることで、両方の長所を活かします。

#### 実装例5: ハイブリッド検索実装
    
    
    from rank_bm25 import BM25Okapi
    import numpy as np
    
    class HybridSearcher:
        """ハイブリッド検索システム"""
    
        def __init__(self, vectorstore, embeddings):
            self.vectorstore = vectorstore
            self.embeddings = embeddings
            self.bm25 = None
            self.documents = []
    
        def initialize_bm25(self, documents):
            """BM25インデックス初期化"""
            self.documents = documents
    
            # トークン化
            tokenized_docs = [
                doc.page_content.split() for doc in documents
            ]
    
            # BM25インデックス作成
            self.bm25 = BM25Okapi(tokenized_docs)
            print(f"BM25インデックス作成: {len(documents)}ドキュメント")
    
        def bm25_search(self, query: str, k: int = 10):
            """BM25キーワード検索"""
            if not self.bm25:
                raise ValueError("BM25が未初期化")
    
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
    
            # Top-K取得
            top_indices = np.argsort(scores)[::-1][:k]
    
            results = [
                {
                    'document': self.documents[idx],
                    'score': float(scores[idx])
                }
                for idx in top_indices
            ]
    
            return results
    
        def vector_search(self, query: str, k: int = 10):
            """ベクトル類似度検索"""
            results = self.vectorstore.similarity_search_with_score(query, k=k)
    
            return [
                {
                    'document': doc,
                    'score': float(score)
                }
                for doc, score in results
            ]
    
        def hybrid_search(self, query: str, k: int = 5,
                         vector_weight: float = 0.5):
            """ハイブリッド検索
    
            Args:
                query: 検索クエリ
                k: 返すドキュメント数
                vector_weight: ベクトル検索の重み（0-1）
            """
            # 両方の検索を実行
            bm25_results = self.bm25_search(query, k=k*2)
            vector_results = self.vector_search(query, k=k*2)
    
            # スコア正規化
            bm25_scores = [r['score'] for r in bm25_results]
            vector_scores = [r['score'] for r in vector_results]
    
            bm25_normalized = self._normalize_scores(bm25_scores)
            vector_normalized = self._normalize_scores(vector_scores)
    
            # スコア統合
            combined_scores = {}
    
            for i, result in enumerate(bm25_results):
                doc_hash = hash(result['document'].page_content)
                combined_scores[doc_hash] = {
                    'document': result['document'],
                    'score': (1 - vector_weight) * bm25_normalized[i]
                }
    
            for i, result in enumerate(vector_results):
                doc_hash = hash(result['document'].page_content)
                if doc_hash in combined_scores:
                    combined_scores[doc_hash]['score'] += vector_weight * vector_normalized[i]
                else:
                    combined_scores[doc_hash] = {
                        'document': result['document'],
                        'score': vector_weight * vector_normalized[i]
                    }
    
            # ソートしてTop-K取得
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:k]
    
            return sorted_results
    
        def _normalize_scores(self, scores):
            """スコア正規化（0-1範囲）"""
            scores = np.array(scores)
            if scores.max() == scores.min():
                return np.ones_like(scores)
            return (scores - scores.min()) / (scores.max() - scores.min())
    
        def compare_search_methods(self, query: str, k: int = 5):
            """検索手法の比較"""
            results = {
                'BM25のみ': self.bm25_search(query, k=k),
                'ベクトル検索のみ': self.vector_search(query, k=k),
                'ハイブリッド(50:50)': self.hybrid_search(query, k=k, vector_weight=0.5),
                'ハイブリッド(ベクトル重視)': self.hybrid_search(query, k=k, vector_weight=0.7)
            }
    
            return results
    
    # 使用例
    hybrid_searcher = HybridSearcher(vectorstore, embeddings)
    
    # BM25初期化（全ドキュメント）
    all_docs = vectorstore.similarity_search("", k=1000)  # 全取得の代替
    hybrid_searcher.initialize_bm25(all_docs)
    
    # 比較検索
    query = "機械学習 評価指標 精度"
    comparison = hybrid_searcher.compare_search_methods(query, k=3)
    
    for method_name, results in comparison.items():
        print(f"\n【{method_name}】")
        for i, result in enumerate(results, 1):
            print(f"{i}. スコア: {result['score']:.4f}")
            print(f"   {result['document'].page_content[:60]}...")

## 4\. コンテキスト圧縮

### 4.1 コンテキスト圧縮の必要性

検索結果をLLMに渡す際、トークン数削減と関連情報抽出のために圧縮します。

**圧縮の利点:**

  * **コスト削減** : トークン数削減によりAPI費用を低減
  * **精度向上** : ノイズ除去により回答品質が向上
  * **レスポンス高速化** : 処理時間短縮

#### 実装例6: コンテキスト圧縮
    
    
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.chat_models import ChatOpenAI
    
    class ContextCompressor:
        """コンテキスト圧縮システム"""
    
        def __init__(self, llm):
            self.llm = llm
    
        def create_compression_retriever(self, base_retriever):
            """圧縮レトリーバー作成"""
            # LLMベースの抽出器
            compressor = LLMChainExtractor.from_llm(self.llm)
    
            # 圧縮レトリーバー
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
    
            return compression_retriever
    
        def extract_relevant_parts(self, query: str, documents: list):
            """関連部分のみ抽出"""
            template = """以下のドキュメントから、質問に関連する部分のみを抽出してください。
            関連しない情報は除外してください。
    
            質問: {query}
    
            ドキュメント:
            {document}
    
            関連部分:"""
    
            extracted = []
            for doc in documents:
                prompt = template.format(
                    query=query,
                    document=doc.page_content
                )
    
                response = self.llm(prompt)
                extracted.append(response.content)
    
            return extracted
    
        def summarize_for_context(self, documents: list, max_tokens: int = 500):
            """コンテキスト用要約"""
            combined_text = "\n\n".join([doc.page_content for doc in documents])
    
            prompt = f"""以下のドキュメント群を{max_tokens}トークン以内に要約してください。
            重要な情報を保持しながら簡潔にまとめてください。
    
            ドキュメント:
            {combined_text}
    
            要約:"""
    
            summary = self.llm(prompt)
            return summary.content
    
    # 使用例
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key="your-api-key")
    compressor = ContextCompressor(llm)
    
    # 検索結果取得
    query = "機械学習モデルの過学習対策"
    search_results = vectorstore.similarity_search(query, k=5)
    
    # 関連部分抽出
    extracted = compressor.extract_relevant_parts(query, search_results)
    
    print("圧縮前:")
    total_chars_before = sum(len(doc.page_content) for doc in search_results)
    print(f"総文字数: {total_chars_before}")
    
    print("\n圧縮後:")
    total_chars_after = sum(len(text) for text in extracted)
    print(f"総文字数: {total_chars_after}")
    print(f"圧縮率: {(1 - total_chars_after/total_chars_before)*100:.1f}%")
    
    # 要約版
    summary = compressor.summarize_for_context(search_results, max_tokens=300)
    print(f"\n要約:\n{summary}")

## まとめ

  * クエリ分解とHyDEにより検索精度を向上
  * Cross-EncoderとMMRで検索結果を最適化
  * ハイブリッド検索でキーワードとベクトルの長所を統合
  * コンテキスト圧縮でコストと精度を両立

[← 第2章](<./chapter2-embeddings.html>) [第4章へ →](<./chapter4-production.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
