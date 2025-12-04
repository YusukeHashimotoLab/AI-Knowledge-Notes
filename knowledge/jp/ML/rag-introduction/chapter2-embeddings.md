---
title: 第2章：エンベディングと検索
chapter_title: 第2章：エンベディングと検索
---

## 1\. ベクトルエンベディング

### 1.1 エンベディングの概念

エンベディングとは、テキストを高次元ベクトル空間上の点として表現する技術です。意味的に類似したテキストは、ベクトル空間でも近い位置に配置されます。

**エンベディングの特性:**

  * **意味表現** : 単語やテキストの意味を数値ベクトルで捉える
  * **次元削減** : 高次元の言語空間を固定次元（例: 1536次元）に圧縮
  * **比較可能性** : ベクトル演算により類似度を計算

**コサイン類似度:**

2つのベクトル \\(\mathbf{u}\\) と \\(\mathbf{v}\\) の類似度:

$$\text{similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$ 

範囲: -1（正反対）～ 1（完全一致）

#### 実装例1: エンベディング生成と類似度計算
    
    
    from openai import OpenAI
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class EmbeddingGenerator:
        """エンベディング生成と類似度計算"""
    
        def __init__(self, api_key, model="text-embedding-3-small"):
            self.client = OpenAI(api_key=api_key)
            self.model = model
    
        def get_embedding(self, text):
            """単一テキストのエンベディング取得"""
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return np.array(response.data[0].embedding)
    
        def get_embeddings_batch(self, texts, batch_size=100):
            """バッチ処理でエンベディング取得"""
            embeddings = []
    
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [
                    np.array(data.embedding) for data in response.data
                ]
                embeddings.extend(batch_embeddings)
    
            return np.array(embeddings)
    
        def cosine_similarity(self, vec1, vec2):
            """コサイン類似度計算"""
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
        def find_most_similar(self, query_text, document_texts, top_k=5):
            """最も類似したドキュメントを検索"""
            # エンベディング取得
            query_emb = self.get_embedding(query_text)
            doc_embs = self.get_embeddings_batch(document_texts)
    
            # 類似度計算
            similarities = cosine_similarity([query_emb], doc_embs)[0]
    
            # Top-K取得
            top_indices = np.argsort(similarities)[::-1][:top_k]
    
            results = [
                {
                    'text': document_texts[idx],
                    'score': float(similarities[idx]),
                    'rank': rank + 1
                }
                for rank, idx in enumerate(top_indices)
            ]
    
            return results
    
    # 使用例
    generator = EmbeddingGenerator(api_key="your-api-key")
    
    documents = [
        "機械学習はデータから学習するAI技術です",
        "深層学習はニューラルネットワークを使用します",
        "自然言語処理はテキスト分析の手法です",
        "コンピュータビジョンは画像認識に特化しています"
    ]
    
    query = "AIによるテキスト解析"
    results = generator.find_most_similar(query, documents, top_k=3)
    
    for result in results:
        print(f"ランク{result['rank']}: {result['text']}")
        print(f"類似度: {result['score']:.4f}\n")

### 1.2 エンベディングモデルの選択

様々なエンベディングモデルが存在し、用途に応じて選択します。

#### 実装例2: 複数エンベディングモデルの比較
    
    
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings import (
        OpenAIEmbeddings, HuggingFaceEmbeddings
    )
    import time
    
    class EmbeddingComparison:
        """複数エンベディングモデルの比較"""
    
        def __init__(self):
            self.models = {}
    
        def load_models(self, openai_api_key=None):
            """各種モデル読み込み"""
            # OpenAI
            if openai_api_key:
                self.models['openai-small'] = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=openai_api_key
                )
                self.models['openai-large'] = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=openai_api_key
                )
    
            # Sentence Transformers (ローカル)
            self.models['multilingual'] = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.models['japanese'] = SentenceTransformer(
                'sentence-transformers/distiluse-base-multilingual-cased-v1'
            )
    
        def benchmark_model(self, model_name, texts):
            """モデルのベンチマーク"""
            model = self.models[model_name]
    
            start = time.time()
    
            if isinstance(model, SentenceTransformer):
                embeddings = model.encode(texts)
            else:
                embeddings = model.embed_documents(texts)
    
            elapsed = time.time() - start
    
            return {
                'model': model_name,
                'num_texts': len(texts),
                'time': elapsed,
                'time_per_text': elapsed / len(texts),
                'dimension': len(embeddings[0])
            }
    
        def compare_all_models(self, test_texts):
            """全モデル比較"""
            results = []
    
            for model_name in self.models.keys():
                try:
                    result = self.benchmark_model(model_name, test_texts)
                    results.append(result)
                    print(f"{model_name}: {result['time']:.2f}秒 "
                          f"(次元: {result['dimension']})")
                except Exception as e:
                    print(f"{model_name}: エラー - {e}")
    
            return results
    
    # 使用例
    comparator = EmbeddingComparison()
    comparator.load_models(openai_api_key="your-api-key")
    
    test_texts = [
        "機械学習の基礎を学ぶ",
        "深層学習モデルの構築",
        "自然言語処理の応用"
    ] * 10  # 30テキスト
    
    results = comparator.compare_all_models(test_texts)

## 2\. 類似度検索

### 2.1 検索アルゴリズム

ベクトルデータベースは、大規模なエンベディングコレクションから高速に類似ベクトルを検索します。

**主要な検索手法:**

  * **総当たり検索** : 全ベクトルと比較（小規模データ向け）
  * **近似最近傍探索（ANN）** : HNSW、IVF等のインデックス構造
  * **ハイブリッド検索** : ベクトル検索 + キーワード検索

## 3\. ベクトルデータベース

### 3.1 FAISS（Facebook AI Similarity Search）

Metaが開発した高速類似度検索ライブラリで、ローカル環境で動作します。

#### 実装例3: FAISS実装
    
    
    import faiss
    import numpy as np
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.docstore.document import Document
    
    class FAISSVectorStore:
        """FAISS ベクトルストア実装"""
    
        def __init__(self, embeddings):
            self.embeddings = embeddings
            self.vectorstore = None
    
        def create_index(self, documents, index_type='flat'):
            """インデックス作成"""
            # Langchain FAISS使用
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embeddings
            )
    
            # カスタムインデックス設定も可能
            if index_type == 'ivf':
                self._create_ivf_index(documents)
    
            print(f"インデックス作成完了: {len(documents)}ドキュメント")
    
        def _create_ivf_index(self, documents):
            """IVF（Inverted File）インデックス作成"""
            # エンベディング取得
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings).astype('float32')
    
            # 次元数
            dimension = embeddings_array.shape[1]
    
            # IVFインデックス作成
            nlist = 100  # クラスタ数
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
            # 学習
            index.train(embeddings_array)
            index.add(embeddings_array)
    
            print(f"IVFインデックス作成: {nlist}クラスタ")
            return index
    
        def search(self, query, k=5, score_threshold=None):
            """類似ドキュメント検索"""
            if score_threshold:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=k
                )
                # スコアでフィルタリング
                filtered = [
                    (doc, score) for doc, score in results
                    if score >= score_threshold
                ]
                return filtered
            else:
                return self.vectorstore.similarity_search(query, k=k)
    
        def search_with_metadata_filter(self, query, k=5, filter_dict=None):
            """メタデータフィルタ付き検索"""
            if filter_dict:
                return self.vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            return self.search(query, k=k)
    
        def save_local(self, path):
            """ローカル保存"""
            self.vectorstore.save_local(path)
            print(f"保存完了: {path}")
    
        def load_local(self, path):
            """ローカル読み込み"""
            self.vectorstore = FAISS.load_local(
                path, self.embeddings
            )
            print(f"読み込み完了: {path}")
    
    # 使用例
    embeddings = OpenAIEmbeddings(openai_api_key="your-api-key")
    faiss_store = FAISSVectorStore(embeddings)
    
    # ドキュメント準備
    documents = [
        Document(
            page_content="Pythonは人気のプログラミング言語です",
            metadata={"category": "programming", "language": "ja"}
        ),
        Document(
            page_content="機械学習にはPythonがよく使われます",
            metadata={"category": "ml", "language": "ja"}
        )
    ]
    
    # インデックス作成
    faiss_store.create_index(documents)
    
    # 検索
    results = faiss_store.search("プログラミング言語", k=2)
    for doc in results:
        print(f"- {doc.page_content}")
    
    # 保存
    faiss_store.save_local("./faiss_index")

### 3.2 ChromaDB

オープンソースのベクトルデータベースで、メタデータフィルタリングに優れています。

#### 実装例4: ChromaDB実装
    
    
    import chromadb
    from chromadb.config import Settings
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    
    class ChromaVectorStore:
        """ChromaDB ベクトルストア実装"""
    
        def __init__(self, embeddings, persist_directory="./chroma_db"):
            self.embeddings = embeddings
            self.persist_directory = persist_directory
            self.vectorstore = None
    
            # クライアント設定
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))
    
        def create_collection(self, documents, collection_name="default"):
            """コレクション作成"""
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )
    
            # 永続化
            self.vectorstore.persist()
            print(f"コレクション作成: {collection_name}")
    
        def add_documents(self, documents):
            """ドキュメント追加"""
            if not self.vectorstore:
                raise ValueError("コレクション未作成")
    
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            print(f"{len(documents)}ドキュメント追加")
    
        def search_with_filter(self, query, k=5, where=None, where_document=None):
            """高度なフィルタリング検索"""
            # メタデータフィルタ
            if where:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=where
                )
            # ドキュメント内容フィルタ
            elif where_document:
                results = self.vectorstore.similarity_search(
                    query, k=k, where_document=where_document
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
    
            return results
    
        def mmr_search(self, query, k=5, fetch_k=20, lambda_mult=0.5):
            """MMR（Maximal Marginal Relevance）検索
    
            多様性と関連性のバランスを取る検索
            """
            results = self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult  # 0=多様性重視, 1=関連性重視
            )
            return results
    
        def delete_collection(self, collection_name):
            """コレクション削除"""
            self.client.delete_collection(collection_name)
            print(f"削除完了: {collection_name}")
    
    # 使用例
    embeddings = OpenAIEmbeddings(openai_api_key="your-api-key")
    chroma_store = ChromaVectorStore(embeddings, persist_directory="./chroma_db")
    
    documents = [
        Document(
            page_content="Python機械学習入門",
            metadata={"type": "tutorial", "level": "beginner", "year": 2024}
        ),
        Document(
            page_content="高度な深層学習テクニック",
            metadata={"type": "advanced", "level": "expert", "year": 2024}
        ),
        Document(
            page_content="データサイエンス基礎",
            metadata={"type": "tutorial", "level": "beginner", "year": 2023}
        )
    ]
    
    # コレクション作成
    chroma_store.create_collection(documents, collection_name="ml_docs")
    
    # メタデータフィルタ検索
    results = chroma_store.search_with_filter(
        "機械学習",
        k=2,
        where={"level": "beginner", "year": 2024}
    )
    
    for doc in results:
        print(f"- {doc.page_content}")
        print(f"  メタデータ: {doc.metadata}")
    
    # MMR検索（多様性重視）
    diverse_results = chroma_store.mmr_search(
        "機械学習の学習",
        k=3,
        lambda_mult=0.3  # 多様性重視
    )
    print(f"\nMMR検索結果: {len(diverse_results)}件")

### 3.3 Pinecone

クラウドネイティブなベクトルデータベースで、スケーラビリティに優れています。

#### 実装例5: Pinecone実装
    
    
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings
    import time
    
    class PineconeVectorStore:
        """Pinecone ベクトルストア実装"""
    
        def __init__(self, api_key, environment, embeddings):
            self.embeddings = embeddings
    
            # Pinecone初期化
            pinecone.init(
                api_key=api_key,
                environment=environment
            )
    
        def create_index(self, index_name, dimension=1536, metric='cosine'):
            """インデックス作成"""
            # 既存インデックス確認
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    pods=1,
                    pod_type='p1.x1'
                )
                # インデックス準備待ち
                time.sleep(1)
                print(f"インデックス作成: {index_name}")
            else:
                print(f"既存インデックス使用: {index_name}")
    
        def upsert_documents(self, index_name, documents):
            """ドキュメントアップサート"""
            vectorstore = Pinecone.from_documents(
                documents,
                self.embeddings,
                index_name=index_name
            )
            print(f"{len(documents)}ドキュメントアップサート完了")
            return vectorstore
    
        def search_with_namespace(self, index_name, query, k=5, namespace=None):
            """名前空間指定検索"""
            vectorstore = Pinecone.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings,
                namespace=namespace
            )
    
            results = vectorstore.similarity_search_with_score(query, k=k)
            return results
    
        def hybrid_search(self, index_name, query, k=5, alpha=0.5):
            """ハイブリッド検索（密ベクトル + 疎ベクトル）
    
            alpha: 0=キーワード検索のみ, 1=ベクトル検索のみ
            """
            # Pineconeのハイブリッド検索機能
            index = pinecone.Index(index_name)
    
            # クエリエンベディング
            query_vector = self.embeddings.embed_query(query)
    
            # ハイブリッド検索実行
            results = index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
                # ハイブリッド検索パラメータ
                alpha=alpha
            )
    
            return results
    
        def delete_index(self, index_name):
            """インデックス削除"""
            if index_name in pinecone.list_indexes():
                pinecone.delete_index(index_name)
                print(f"インデックス削除: {index_name}")
    
        def get_index_stats(self, index_name):
            """インデックス統計取得"""
            index = pinecone.Index(index_name)
            stats = index.describe_index_stats()
            return stats
    
    # 使用例
    embeddings = OpenAIEmbeddings(openai_api_key="your-openai-key")
    pinecone_store = PineconeVectorStore(
        api_key="your-pinecone-key",
        environment="us-west1-gcp",
        embeddings=embeddings
    )
    
    # インデックス作成
    index_name = "ml-knowledge-base"
    pinecone_store.create_index(index_name, dimension=1536)
    
    # ドキュメントアップサート
    documents = [
        Document(
            page_content="機械学習の基礎理論",
            metadata={"category": "ml", "level": "basic"}
        ),
        Document(
            page_content="深層学習の実装方法",
            metadata={"category": "dl", "level": "advanced"}
        )
    ]
    
    vectorstore = pinecone_store.upsert_documents(index_name, documents)
    
    # 検索
    results = pinecone_store.search_with_namespace(
        index_name, "機械学習の学び方", k=3
    )
    
    for doc, score in results:
        print(f"スコア: {score:.4f}")
        print(f"内容: {doc.page_content}")
        print(f"メタデータ: {doc.metadata}\n")
    
    # 統計情報
    stats = pinecone_store.get_index_stats(index_name)
    print(f"総ベクトル数: {stats['total_vector_count']}")

### 3.4 ベクトルDBの比較と選択

#### 実装例6: ベクトルDB性能比較
    
    
    import time
    from typing import List, Dict
    from langchain.schema import Document
    
    class VectorDBBenchmark:
        """ベクトルデータベース性能比較"""
    
        def __init__(self):
            self.results = []
    
        def benchmark_indexing(self, db_name, vectorstore, documents):
            """インデックス作成時間測定"""
            start = time.time()
    
            if db_name == "FAISS":
                vectorstore.create_index(documents)
            elif db_name == "Chroma":
                vectorstore.create_collection(documents)
            elif db_name == "Pinecone":
                vectorstore.upsert_documents("benchmark", documents)
    
            elapsed = time.time() - start
    
            return {
                'db': db_name,
                'operation': 'indexing',
                'num_docs': len(documents),
                'time': elapsed,
                'docs_per_sec': len(documents) / elapsed
            }
    
        def benchmark_search(self, db_name, vectorstore, queries, k=5):
            """検索時間測定"""
            start = time.time()
    
            for query in queries:
                if db_name == "FAISS":
                    vectorstore.search(query, k=k)
                elif db_name == "Chroma":
                    vectorstore.search_with_filter(query, k=k)
                elif db_name == "Pinecone":
                    vectorstore.search_with_namespace("benchmark", query, k=k)
    
            elapsed = time.time() - start
    
            return {
                'db': db_name,
                'operation': 'search',
                'num_queries': len(queries),
                'time': elapsed,
                'queries_per_sec': len(queries) / elapsed,
                'avg_latency_ms': (elapsed / len(queries)) * 1000
            }
    
        def compare_features(self):
            """機能比較表"""
            comparison = {
                'FAISS': {
                    'type': 'ローカルライブラリ',
                    'deployment': 'セルフホスト',
                    'scalability': '中',
                    'metadata_filter': '限定的',
                    'cost': '無料（インフラコストのみ）',
                    'best_for': '小〜中規模、オフライン環境'
                },
                'Chroma': {
                    'type': 'ローカル/サーバー',
                    'deployment': 'セルフホスト/クラウド',
                    'scalability': '中〜高',
                    'metadata_filter': '強力',
                    'cost': '無料（オープンソース）',
                    'best_for': '中規模、開発環境'
                },
                'Pinecone': {
                    'type': 'クラウドサービス',
                    'deployment': 'マネージド',
                    'scalability': '非常に高い',
                    'metadata_filter': '強力',
                    'cost': '有料（従量課金）',
                    'best_for': '大規模、本番環境'
                }
            }
            return comparison
    
        def print_comparison(self):
            """比較結果表示"""
            features = self.compare_features()
    
            print("=" * 80)
            print("ベクトルデータベース機能比較")
            print("=" * 80)
    
            for db_name, features_dict in features.items():
                print(f"\n【{db_name}】")
                for key, value in features_dict.items():
                    print(f"  {key:20s}: {value}")
    
    # 使用例
    benchmark = VectorDBBenchmark()
    
    # 機能比較表示
    benchmark.print_comparison()
    
    # テストデータ
    test_documents = [
        Document(page_content=f"ドキュメント{i}")
        for i in range(1000)
    ]
    
    test_queries = [f"クエリ{i}" for i in range(100)]
    
    # 各DBでベンチマーク実行
    # faiss_result = benchmark.benchmark_indexing("FAISS", faiss_store, test_documents)
    # chroma_result = benchmark.benchmark_indexing("Chroma", chroma_store, test_documents)
    
    print("\n性能ベンチマーク完了")

**ベクトルDB選択ガイド:**

  * **FAISS** : プロトタイプ、小規模、オフライン環境
  * **Chroma** : 開発環境、中規模、メタデータ活用
  * **Pinecone** : 本番環境、大規模、マネージドサービス希望

## まとめ

  * エンベディングはテキストをベクトル空間に変換し、意味的類似度を計算可能にする
  * コサイン類似度が最も一般的な類似度メトリクス
  * FAISS、Chroma、Pineconeはそれぞれ異なる特性を持つ
  * 用途とスケールに応じて適切なベクトルDBを選択

[← 第1章](<./chapter1-rag-basics.html>) [第3章へ →](<./chapter3-advanced-rag.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
