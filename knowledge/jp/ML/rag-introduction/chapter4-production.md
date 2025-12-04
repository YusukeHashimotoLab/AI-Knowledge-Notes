---
title: 第4章：プロダクション環境への展開
chapter_title: 第4章：プロダクション環境への展開
---

## 1\. システムアーキテクチャ

### 1.1 マイクロサービス設計

本番環境のRAGシステムは、複数のサービスに分離して構築します。

**RAGシステムの主要コンポーネント:**

  * **インジェストサービス** : ドキュメント取り込みと前処理
  * **エンベディングサービス** : ベクトル化処理
  * **検索サービス** : ベクトルDB検索とリランキング
  * **生成サービス** : LLM呼び出しと回答生成
  * **APIゲートウェイ** : リクエスト管理と認証

#### 実装例1: FastAPIベースRAGシステム
    
    
    from fastapi import FastAPI, HTTPException, Depends
    from pydantic import BaseModel
    from typing import List, Optional
    import asyncio
    from functools import lru_cache
    
    app = FastAPI(title="RAG API", version="1.0.0")
    
    # リクエスト/レスポンスモデル
    class QueryRequest(BaseModel):
        query: str
        top_k: int = 5
        filters: Optional[dict] = None
    
    class SearchResult(BaseModel):
        content: str
        score: float
        metadata: dict
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[SearchResult]
        processing_time: float
    
    # 依存性注入
    @lru_cache()
    def get_rag_service():
        """RAGサービスのシングルトン取得"""
        from services.rag_service import RAGService
        return RAGService()
    
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(
        request: QueryRequest,
        rag_service = Depends(get_rag_service)
    ):
        """RAGクエリエンドポイント"""
        try:
            import time
            start = time.time()
    
            # 検索と生成を並行実行
            answer, sources = await rag_service.query(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters
            )
    
            processing_time = time.time() - start
    
            return QueryResponse(
                answer=answer,
                sources=sources,
                processing_time=processing_time
            )
    
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    class IndexRequest(BaseModel):
        documents: List[dict]
        collection_name: str
    
    @app.post("/index")
    async def index_endpoint(
        request: IndexRequest,
        rag_service = Depends(get_rag_service)
    ):
        """ドキュメントインデックス作成"""
        try:
            result = await rag_service.index_documents(
                documents=request.documents,
                collection_name=request.collection_name
            )
            return {"status": "success", "indexed": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """ヘルスチェック"""
        return {"status": "healthy"}
    
    # RAGサービス実装（services/rag_service.py）
    class RAGService:
        """RAGビジネスロジック"""
    
        def __init__(self):
            self.vectorstore = self._init_vectorstore()
            self.llm = self._init_llm()
            self.embeddings = self._init_embeddings()
    
        def _init_vectorstore(self):
            # ベクトルストア初期化
            pass
    
        def _init_llm(self):
            # LLM初期化
            pass
    
        def _init_embeddings(self):
            # エンベディングモデル初期化
            pass
    
        async def query(self, query: str, top_k: int = 5, filters: dict = None):
            """非同期クエリ処理"""
            # 検索
            search_results = await self._search(query, top_k, filters)
    
            # 生成
            answer = await self._generate(query, search_results)
    
            return answer, search_results
    
        async def _search(self, query: str, top_k: int, filters: dict):
            """非同期検索"""
            # 実装
            pass
    
        async def _generate(self, query: str, context: list):
            """非同期生成"""
            # 実装
            pass
    
        async def index_documents(self, documents: list, collection_name: str):
            """非同期インデックス作成"""
            # 実装
            pass
    
    # 実行
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

## 2\. パフォーマンス最適化

### 2.1 キャッシング戦略

頻繁なクエリに対してキャッシュを活用し、レスポンス時間を短縮します。

#### 実装例2: マルチレベルキャッシング
    
    
    import redis
    from functools import lru_cache
    import hashlib
    import json
    import time
    
    class MultiLevelCache:
        """マルチレベルキャッシングシステム"""
    
        def __init__(self, redis_url: str = "redis://localhost:6379"):
            # L1: メモリキャッシュ（LRU）
            self.memory_cache_size = 100
    
            # L2: Redis
            self.redis_client = redis.from_url(redis_url)
    
            # キャッシュ統計
            self.stats = {
                'hits': 0,
                'misses': 0,
                'l1_hits': 0,
                'l2_hits': 0
            }
    
        def _generate_key(self, query: str, params: dict) -> str:
            """キャッシュキー生成"""
            cache_input = f"{query}:{json.dumps(params, sort_keys=True)}"
            return hashlib.md5(cache_input.encode()).hexdigest()
    
        @lru_cache(maxsize=100)
        def _l1_get(self, key: str):
            """L1キャッシュ取得（メモリ）"""
            # LRUデコレータで自動管理
            return None
    
        def get(self, query: str, params: dict):
            """キャッシュ取得（L1 → L2）"""
            key = self._generate_key(query, params)
    
            # L1チェック
            try:
                result = self._l1_get(key)
                if result:
                    self.stats['hits'] += 1
                    self.stats['l1_hits'] += 1
                    return result
            except:
                pass
    
            # L2チェック（Redis）
            try:
                cached = self.redis_client.get(key)
                if cached:
                    result = json.loads(cached)
    
                    # L1にプロモート
                    self._l1_set(key, result)
    
                    self.stats['hits'] += 1
                    self.stats['l2_hits'] += 1
                    return result
            except Exception as e:
                print(f"Redis error: {e}")
    
            self.stats['misses'] += 1
            return None
    
        def _l1_set(self, key: str, value):
            """L1キャッシュ設定"""
            # LRUキャッシュに設定
            self._l1_get.__wrapped__(self, key)  # トリガー
            self._l1_get.cache_info()
    
        def set(self, query: str, params: dict, value, ttl: int = 3600):
            """キャッシュ設定（L1 & L2）"""
            key = self._generate_key(query, params)
    
            # L1設定
            self._l1_set(key, value)
    
            # L2設定（Redis）
            try:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value)
                )
            except Exception as e:
                print(f"Redis set error: {e}")
    
        def invalidate(self, pattern: str = "*"):
            """キャッシュ無効化"""
            # L1クリア
            self._l1_get.cache_clear()
    
            # L2クリア（パターンマッチ）
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                print(f"Redis invalidate error: {e}")
    
        def get_stats(self):
            """キャッシュ統計"""
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0
    
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total
            }
    
    # RAGシステムとの統合
    class CachedRAGService:
        """キャッシュ付きRAGサービス"""
    
        def __init__(self):
            self.cache = MultiLevelCache()
            self.rag_service = RAGService()
    
        async def query(self, query: str, top_k: int = 5):
            """キャッシュ考慮クエリ"""
            params = {'top_k': top_k}
    
            # キャッシュチェック
            cached_result = self.cache.get(query, params)
            if cached_result:
                print("Cache hit!")
                return cached_result
    
            # キャッシュミス: 実行
            print("Cache miss, executing query...")
            result = await self.rag_service.query(query, top_k)
    
            # キャッシュ保存（1時間TTL）
            self.cache.set(query, params, result, ttl=3600)
    
            return result
    
        def get_cache_stats(self):
            """キャッシュ統計取得"""
            return self.cache.get_stats()
    
    # 使用例
    cached_rag = CachedRAGService()
    
    # 初回クエリ（キャッシュミス）
    result1 = await cached_rag.query("機械学習とは")
    
    # 同じクエリ（キャッシュヒット）
    result2 = await cached_rag.query("機械学習とは")
    
    # 統計表示
    stats = cached_rag.get_cache_stats()
    print(f"ヒット率: {stats['hit_rate']:.2%}")

### 2.2 バッチ処理最適化

複数のドキュメント処理を効率化するバッチ処理を実装します。

#### 実装例3: バッチインデックス作成
    
    
    import asyncio
    from typing import List
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    
    class BatchIndexer:
        """バッチインデックス作成システム"""
    
        def __init__(self, embeddings, vectorstore, batch_size=50, max_workers=4):
            self.embeddings = embeddings
            self.vectorstore = vectorstore
            self.batch_size = batch_size
            self.max_workers = max_workers
    
        def create_batches(self, documents: List, batch_size: int):
            """ドキュメントをバッチに分割"""
            for i in range(0, len(documents), batch_size):
                yield documents[i:i + batch_size]
    
        async def process_batch_async(self, batch: List):
            """バッチ非同期処理"""
            # エンベディング生成（並行）
            texts = [doc.page_content for doc in batch]
    
            # バッチエンベディング取得
            embeddings = await asyncio.to_thread(
                self.embeddings.embed_documents,
                texts
            )
    
            # ベクトルストアに追加
            await asyncio.to_thread(
                self.vectorstore.add_documents,
                batch
            )
    
            return len(batch)
    
        async def index_documents_parallel(self, documents: List):
            """並列バッチインデックス作成"""
            batches = list(self.create_batches(documents, self.batch_size))
    
            print(f"処理開始: {len(documents)}ドキュメント, {len(batches)}バッチ")
    
            # 並列処理
            tasks = [
                self.process_batch_async(batch)
                for batch in batches
            ]
    
            # セマフォで同時実行数制限
            semaphore = asyncio.Semaphore(self.max_workers)
    
            async def limited_task(task):
                async with semaphore:
                    return await task
    
            results = await asyncio.gather(
                *[limited_task(task) for task in tasks]
            )
    
            total_indexed = sum(results)
            print(f"インデックス完了: {total_indexed}ドキュメント")
    
            return total_indexed
    
        def index_with_progress(self, documents: List):
            """プログレスバー付きインデックス作成"""
            from tqdm import tqdm
    
            batches = list(self.create_batches(documents, self.batch_size))
    
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
    
                for batch in batches:
                    future = executor.submit(self._process_batch_sync, batch)
                    futures.append(future)
    
                # プログレス表示
                with tqdm(total=len(documents), desc="インデックス作成") as pbar:
                    for future in futures:
                        count = future.result()
                        pbar.update(count)
    
            print("インデックス作成完了")
    
        def _process_batch_sync(self, batch: List):
            """同期バッチ処理"""
            texts = [doc.page_content for doc in batch]
            self.embeddings.embed_documents(texts)
            self.vectorstore.add_documents(batch)
            return len(batch)
    
    # 使用例
    batch_indexer = BatchIndexer(
        embeddings=embeddings,
        vectorstore=vectorstore,
        batch_size=50,
        max_workers=4
    )
    
    # 大量ドキュメント
    large_documents = [...]  # 10000ドキュメント
    
    # 非同期並列処理
    await batch_indexer.index_documents_parallel(large_documents)
    
    # プログレスバー付き処理
    batch_indexer.index_with_progress(large_documents)

## 3\. モニタリングと評価

### 3.1 メトリクス設計

RAGシステムの品質とパフォーマンスを測定するメトリクスを定義します。

**主要メトリクス:**

  * **レイテンシ** : 検索時間、生成時間、総処理時間
  * **精度** : 検索精度、回答の正確性
  * **関連性** : 検索結果とクエリの関連度
  * **コスト** : APIコール数、トークン使用量

#### 実装例4: メトリクス収集システム
    
    
    from prometheus_client import Counter, Histogram, Gauge
    import time
    from functools import wraps
    
    # Prometheusメトリクス定義
    query_counter = Counter(
        'rag_queries_total',
        'Total number of RAG queries',
        ['status']
    )
    
    query_latency = Histogram(
        'rag_query_duration_seconds',
        'RAG query duration',
        ['component']
    )
    
    search_results_count = Gauge(
        'rag_search_results',
        'Number of search results returned'
    )
    
    llm_tokens = Counter(
        'rag_llm_tokens_total',
        'Total LLM tokens used',
        ['type']  # prompt or completion
    )
    
    class RAGMetrics:
        """RAGメトリクス収集"""
    
        def __init__(self):
            self.metrics_data = []
    
        def track_query(self, func):
            """クエリ処理メトリクス"""
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
    
                try:
                    result = await func(*args, **kwargs)
                    query_counter.labels(status='success').inc()
    
                    # レイテンシ記録
                    duration = time.time() - start
                    query_latency.labels(component='total').observe(duration)
    
                    # 結果数記録
                    if hasattr(result, 'sources'):
                        search_results_count.set(len(result.sources))
    
                    return result
    
                except Exception as e:
                    query_counter.labels(status='error').inc()
                    raise
    
            return wrapper
    
        def track_search(self, func):
            """検索メトリクス"""
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
    
                result = await func(*args, **kwargs)
    
                duration = time.time() - start
                query_latency.labels(component='search').observe(duration)
    
                return result
    
            return wrapper
    
        def track_generation(self, func):
            """生成メトリクス"""
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
    
                result = await func(*args, **kwargs)
    
                duration = time.time() - start
                query_latency.labels(component='generation').observe(duration)
    
                # トークン数記録
                if hasattr(result, 'usage'):
                    llm_tokens.labels(type='prompt').inc(result.usage.prompt_tokens)
                    llm_tokens.labels(type='completion').inc(result.usage.completion_tokens)
    
                return result
    
            return wrapper
    
    class RAGEvaluator:
        """RAG品質評価"""
    
        def __init__(self, llm):
            self.llm = llm
    
        def evaluate_retrieval(self, query: str, retrieved_docs: list, relevant_docs: list):
            """検索精度評価"""
            # Precision@K
            retrieved_ids = {doc.metadata.get('id') for doc in retrieved_docs}
            relevant_ids = {doc.metadata.get('id') for doc in relevant_docs}
    
            hits = retrieved_ids.intersection(relevant_ids)
    
            precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(hits) / len(relevant_ids) if relevant_ids else 0
    
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
    
        async def evaluate_answer_quality(self, query: str, answer: str, ground_truth: str):
            """回答品質評価（LLMベース）"""
            prompt = f"""以下の質問、回答、正解を評価してください。
            1-5のスコアで評価（5が最高）し、理由も述べてください。
    
            質問: {query}
            回答: {answer}
            正解: {ground_truth}
    
            評価（JSON形式）:
            {{
                "accuracy_score": <1-5>,
                "relevance_score": <1-5>,
                "completeness_score": <1-5>,
                "reasoning": "<理由>"
            }}
            """
    
            response = await self.llm(prompt)
    
            # JSONパース
            import json
            evaluation = json.loads(response.content)
    
            return evaluation
    
        def calculate_mrr(self, queries: list, results: list):
            """MRR（Mean Reciprocal Rank）計算"""
            reciprocal_ranks = []
    
            for query_results in results:
                for rank, doc in enumerate(query_results, 1):
                    if doc.metadata.get('is_relevant'):
                        reciprocal_ranks.append(1 / rank)
                        break
                else:
                    reciprocal_ranks.append(0)
    
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    
            return mrr
    
    # 使用例
    metrics = RAGMetrics()
    evaluator = RAGEvaluator(llm)
    
    # メトリクスデコレータ適用
    class MonitoredRAGService:
        def __init__(self):
            self.metrics = RAGMetrics()
    
        @metrics.track_query
        async def query(self, query: str):
            # クエリ処理
            search_results = await self.search(query)
            answer = await self.generate(query, search_results)
            return answer
    
        @metrics.track_search
        async def search(self, query: str):
            # 検索処理
            pass
    
        @metrics.track_generation
        async def generate(self, query: str, context: list):
            # 生成処理
            pass
    
    # 評価実行
    evaluation = await evaluator.evaluate_answer_quality(
        query="機械学習とは",
        answer="生成された回答",
        ground_truth="正解の回答"
    )
    
    print(f"精度スコア: {evaluation['accuracy_score']}/5")

### 3.2 A/Bテスト実装

異なるRAG設定の効果を比較するA/Bテストを実施します。

#### 実装例5: A/Bテストフレームワーク
    
    
    import random
    from typing import Dict, Any
    from dataclasses import dataclass
    from collections import defaultdict
    
    @dataclass
    class ExperimentVariant:
        """実験バリアント"""
        name: str
        config: Dict[str, Any]
        traffic_ratio: float  # 0.0-1.0
    
    class ABTestManager:
        """A/Bテスト管理"""
    
        def __init__(self):
            self.experiments = {}
            self.results = defaultdict(lambda: defaultdict(list))
    
        def create_experiment(self, experiment_name: str, variants: list):
            """実験作成"""
            # トラフィック比率の合計チェック
            total_ratio = sum(v.traffic_ratio for v in variants)
            if abs(total_ratio - 1.0) > 0.001:
                raise ValueError("トラフィック比率の合計は1.0である必要があります")
    
            self.experiments[experiment_name] = variants
            print(f"実験作成: {experiment_name}")
    
        def assign_variant(self, experiment_name: str, user_id: str):
            """ユーザーにバリアント割り当て"""
            if experiment_name not in self.experiments:
                raise ValueError(f"実験が存在しません: {experiment_name}")
    
            # 決定的割り当て（同じユーザーには同じバリアント）
            hash_val = hash(f"{experiment_name}:{user_id}") % 1000 / 1000
    
            cumulative_ratio = 0
            for variant in self.experiments[experiment_name]:
                cumulative_ratio += variant.traffic_ratio
                if hash_val < cumulative_ratio:
                    return variant
    
            # フォールバック
            return self.experiments[experiment_name][0]
    
        def record_result(self, experiment_name: str, variant_name: str,
                         metric_name: str, value: float):
            """結果記録"""
            self.results[experiment_name][variant_name].append({
                'metric': metric_name,
                'value': value
            })
    
        def analyze_results(self, experiment_name: str, metric_name: str):
            """結果分析"""
            if experiment_name not in self.results:
                return None
    
            analysis = {}
    
            for variant_name, results in self.results[experiment_name].items():
                metric_values = [
                    r['value'] for r in results
                    if r['metric'] == metric_name
                ]
    
                if metric_values:
                    analysis[variant_name] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'count': len(metric_values),
                        'min': min(metric_values),
                        'max': max(metric_values)
                    }
    
            return analysis
    
    # RAGシステムでの使用
    class ABTestedRAGService:
        """A/Bテスト対応RAGサービス"""
    
        def __init__(self):
            self.ab_test = ABTestManager()
            self._setup_experiments()
    
        def _setup_experiments(self):
            """実験セットアップ"""
            # チャンキング戦略テスト
            chunking_variants = [
                ExperimentVariant(
                    name="fixed_500",
                    config={"chunk_size": 500, "overlap": 50},
                    traffic_ratio=0.5
                ),
                ExperimentVariant(
                    name="fixed_1000",
                    config={"chunk_size": 1000, "overlap": 100},
                    traffic_ratio=0.5
                )
            ]
            self.ab_test.create_experiment("chunking_strategy", chunking_variants)
    
            # リランキングテスト
            rerank_variants = [
                ExperimentVariant(
                    name="no_rerank",
                    config={"use_reranking": False},
                    traffic_ratio=0.33
                ),
                ExperimentVariant(
                    name="cross_encoder",
                    config={"use_reranking": True, "method": "cross_encoder"},
                    traffic_ratio=0.33
                ),
                ExperimentVariant(
                    name="mmr",
                    config={"use_reranking": True, "method": "mmr"},
                    traffic_ratio=0.34
                )
            ]
            self.ab_test.create_experiment("reranking_method", rerank_variants)
    
        async def query(self, query: str, user_id: str):
            """バリアント適用クエリ"""
            # バリアント割り当て
            chunking_variant = self.ab_test.assign_variant("chunking_strategy", user_id)
            rerank_variant = self.ab_test.assign_variant("reranking_method", user_id)
    
            print(f"ユーザー{user_id}:")
            print(f"  チャンキング: {chunking_variant.name}")
            print(f"  リランキング: {rerank_variant.name}")
    
            # 設定適用
            start = time.time()
    
            # クエリ実行（設定に基づく）
            result = await self._execute_query(
                query,
                chunking_variant.config,
                rerank_variant.config
            )
    
            latency = time.time() - start
    
            # 結果記録
            self.ab_test.record_result(
                "chunking_strategy",
                chunking_variant.name,
                "latency",
                latency
            )
    
            self.ab_test.record_result(
                "reranking_method",
                rerank_variant.name,
                "latency",
                latency
            )
    
            return result
    
        async def _execute_query(self, query: str, chunking_config: dict,
                               rerank_config: dict):
            """設定を適用したクエリ実行"""
            # 実装
            pass
    
        def get_experiment_results(self):
            """実験結果取得"""
            chunking_results = self.ab_test.analyze_results(
                "chunking_strategy", "latency"
            )
    
            rerank_results = self.ab_test.analyze_results(
                "reranking_method", "latency"
            )
    
            return {
                'chunking': chunking_results,
                'reranking': rerank_results
            }
    
    # 使用例
    ab_rag = ABTestedRAGService()
    
    # テストクエリ実行
    for user_id in range(100):
        await ab_rag.query("機械学習の評価指標", f"user_{user_id}")
    
    # 結果分析
    results = ab_rag.get_experiment_results()
    print("\nチャンキング戦略比較:")
    for variant, stats in results['chunking'].items():
        print(f"{variant}: 平均 {stats['mean']:.3f}秒 (n={stats['count']})")

## 4\. スケーラビリティとセキュリティ

### 4.1 分散処理とロードバランシング

大規模トラフィックに対応する分散アーキテクチャを構築します。

#### 実装例6: Celeryベース非同期処理
    
    
    from celery import Celery
    from kombu import Queue
    import os
    
    # Celery設定
    celery_app = Celery(
        'rag_tasks',
        broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    )
    
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='Asia/Tokyo',
        enable_utc=True,
        task_routes={
            'rag_tasks.index_document': {'queue': 'indexing'},
            'rag_tasks.generate_embedding': {'queue': 'embedding'},
            'rag_tasks.query_rag': {'queue': 'query'}
        }
    )
    
    # タスク定義
    @celery_app.task(name='rag_tasks.index_document')
    def index_document_task(document_data: dict):
        """ドキュメントインデックス作成タスク"""
        from services.indexer import DocumentIndexer
    
        indexer = DocumentIndexer()
        result = indexer.index(document_data)
    
        return {
            'status': 'completed',
            'document_id': document_data.get('id'),
            'indexed_chunks': result['chunks']
        }
    
    @celery_app.task(name='rag_tasks.generate_embedding')
    def generate_embedding_task(text: str):
        """エンベディング生成タスク"""
        from services.embeddings import EmbeddingService
    
        embedding_service = EmbeddingService()
        embedding = embedding_service.generate(text)
    
        return embedding.tolist()
    
    @celery_app.task(name='rag_tasks.query_rag', bind=True)
    def query_rag_task(self, query: str, user_id: str):
        """RAGクエリタスク（リトライ付き）"""
        try:
            from services.rag_service import RAGService
    
            rag = RAGService()
            result = rag.query_sync(query)
    
            return {
                'status': 'success',
                'answer': result['answer'],
                'sources': result['sources']
            }
    
        except Exception as e:
            # リトライ（最大3回、指数バックオフ）
            raise self.retry(exc=e, countdown=2 ** self.request.retries, max_retries=3)
    
    # FastAPI統合
    from fastapi import BackgroundTasks
    
    @app.post("/query_async")
    async def query_async(request: QueryRequest, background_tasks: BackgroundTasks):
        """非同期クエリ"""
        # Celeryタスク実行
        task = query_rag_task.delay(request.query, "user_123")
    
        return {
            'task_id': task.id,
            'status': 'processing'
        }
    
    @app.get("/task_status/{task_id}")
    async def get_task_status(task_id: str):
        """タスク状態確認"""
        task = celery_app.AsyncResult(task_id)
    
        if task.ready():
            return {
                'status': 'completed',
                'result': task.result
            }
        else:
            return {
                'status': 'processing'
            }
    
    # バッチインデックス作成
    @app.post("/batch_index")
    async def batch_index(documents: List[dict]):
        """バッチインデックス作成"""
        # 並列タスク実行
        tasks = [
            index_document_task.delay(doc)
            for doc in documents
        ]
    
        return {
            'status': 'processing',
            'task_count': len(tasks),
            'task_ids': [task.id for task in tasks]
        }
    
    # Celeryワーカー起動コマンド
    # celery -A tasks.celery_app worker --loglevel=info --queues=indexing,embedding,query

**プロダクション運用のベストプラクティス:**

  * **モニタリング** : Prometheus + Grafanaで可視化
  * **ログ管理** : 構造化ログ（JSON）とElasticsearch統合
  * **セキュリティ** : APIキー認証、レート制限、入力検証
  * **CI/CD** : 自動テスト、段階的デプロイ
  * **災害復旧** : バックアップ、レプリケーション

## まとめ

  * マイクロサービスアーキテクチャで各コンポーネントを分離
  * 多層キャッシングとバッチ処理でパフォーマンス最適化
  * 包括的なメトリクス収集とA/Bテストで継続的改善
  * Celeryによる非同期処理で大規模トラフィックに対応
  * セキュリティとスケーラビリティを考慮した本番運用設計

[← 第3章](<./chapter3-advanced-rag.html>) [シリーズ目次へ](<./index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
