---
title: "Chapter 3: Advanced RAG Techniques"
chapter_title: "Chapter 3: Advanced RAG Techniques"
---

This chapter covers advanced topics in Advanced RAG Techniques. You will master essential concepts and techniques.

## 1\. Query Optimization

### 1.1 Query Decomposition

A technique that breaks down complex queries into multiple sub-queries and searches them step by step.

**Example:**

Original Query: "Comparison and future forecasts of AI market size in 2023 and 2024"

After Decomposition:

  1. "AI market size in 2023"
  2. "AI market size in 2024"
  3. "AI market future forecasts"

#### Implementation Example 1: Query Decomposition System
    
    
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing import List
    
    class DecomposedQuery(BaseModel):
        """Decomposed Query"""
        sub_queries: List[str] = Field(description="List of sub-queries")
        reasoning: str = Field(description="Reasoning for decomposition")
    
    class QueryDecomposer:
        """Query Decomposition System"""
    
        def __init__(self, llm):
            self.llm = llm
            self.parser = PydanticOutputParser(pydantic_object=DecomposedQuery)
    
        def decompose(self, query: str) -> DecomposedQuery:
            """Decompose a query"""
            template = """Please decompose the following query into simpler sub-queries.
            Each sub-query must be independently searchable.
    
            Original Query: {query}
    
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
            """Decomposition search and result integration"""
            # Query decomposition
            decomposed = self.decompose(query)
    
            # Search with each sub-query
            all_results = []
            for sub_query in decomposed.sub_queries:
                results = vectorstore.similarity_search(sub_query, k=k)
                all_results.extend(results)
    
            # Remove duplicates (content hash-based)
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
    
    # Usage example
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key="your-api-key")
    decomposer = QueryDecomposer(llm)
    
    query = "Differences between machine learning and deep learning, and their respective applications"
    result = decomposer.search_and_combine(query, vectorstore)
    
    print("Sub-queries:")
    for i, sq in enumerate(result['sub_queries'], 1):
        print(f"{i}. {sq}")
    
    print(f"\nSearch Results: {len(result['results'])} items")

### 1.2 HyDE (Hypothetical Document Embeddings)

A technique that generates a hypothetical answer document from a query and uses it as the query.

#### Implementation Example 2: HyDE Implementation
    
    
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    class HyDERetriever:
        """HyDE Retrieval System"""
    
        def __init__(self, llm, vectorstore, embeddings):
            self.llm = llm
            self.vectorstore = vectorstore
            self.embeddings = embeddings
    
        def generate_hypothetical_document(self, query: str) -> str:
            """Generate hypothetical document"""
            template = """Please write a detailed answer to the following question.
            You don't need actual knowledge. Write a text in the form of an answer
            that includes specific and technical content.
    
            Question: {query}
    
            Answer:"""
    
            prompt = PromptTemplate(template=template, input_variables=["query"])
            response = self.llm(prompt.format(query=query))
    
            return response.content
    
        def search_with_hyde(self, query: str, k=5):
            """HyDE search"""
            # Generate hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(query)
    
            print(f"Hypothetical Document:\n{hypothetical_doc[:200]}...\n")
    
            # Search with hypothetical document
            results = self.vectorstore.similarity_search(hypothetical_doc, k=k)
    
            return results
    
        def hybrid_hyde_search(self, query: str, k=5, alpha=0.5):
            """Hybrid of HyDE and normal search
    
            alpha: HyDE weight (0=normal search only, 1=HyDE only)
            """
            # Normal search
            normal_results = self.vectorstore.similarity_search_with_score(query, k=k)
    
            # HyDE search
            hyde_doc = self.generate_hypothetical_document(query)
            hyde_results = self.vectorstore.similarity_search_with_score(hyde_doc, k=k)
    
            # Score integration
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
    
            # Sort by score
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:k]
    
            return [item['doc'] for item in sorted_results]
    
    # Usage example
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", openai_api_key="your-api-key")
    hyde_retriever = HyDERetriever(llm, vectorstore, embeddings)
    
    query = "Mathematical principles of attention mechanism in Transformer models"
    results = hyde_retriever.search_with_hyde(query, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")

## 2\. Re-ranking

### 2.1 Cross-Encoder Re-ranking

A technique that re-evaluates initial search results with a higher-accuracy model.

**Bi-Encoder vs Cross-Encoder:**

  * **Bi-Encoder** : Encodes query and documents separately (fast, suitable for initial search)
  * **Cross-Encoder** : Encodes query and documents together (high precision, suitable for re-ranking)

#### Implementation Example 3: Cross-Encoder Re-ranking
    
    
    from sentence_transformers import CrossEncoder
    import numpy as np
    
    class ReRanker:
        """Re-ranking System"""
    
        def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
            self.cross_encoder = CrossEncoder(model_name)
    
        def rerank(self, query: str, documents: list, top_k: int = 5):
            """Re-rank with Cross-Encoder"""
            # Create query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]
    
            # Calculate scores
            scores = self.cross_encoder.predict(pairs)
    
            # Sort by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
    
            # Get Top-K
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
            """Two-stage retrieval (initial search → re-ranking)"""
            # Stage 1: Get candidates with vector search
            candidates = vectorstore.similarity_search(query, k=first_k)
            print(f"Stage 1: Retrieved {len(candidates)} items")
    
            # Stage 2: Re-rank with Cross-Encoder
            reranked = self.rerank(query, candidates, top_k=final_k)
            print(f"Stage 2: Re-ranked to top {final_k} items")
    
            return reranked
    
    # Usage example
    reranker = ReRanker()
    
    query = "Machine learning model evaluation metrics"
    results = reranker.two_stage_retrieval(
        query,
        vectorstore,
        first_k=20,
        final_k=5
    )
    
    print("\nRe-ranking Results:")
    for result in results:
        print(f"Rank {result['rank']}: Score {result['score']:.4f}")
        print(f"  {result['document'].page_content[:80]}...\n")

### 2.2 MMR (Maximal Marginal Relevance)

A search technique that balances relevance and diversity.

**MMR Algorithm:** $$\text{MMR} = \arg\max_{D_i \in R \setminus S} [\lambda \cdot \text{Sim}_1(D_i, Q) - (1-\lambda) \cdot \max_{D_j \in S} \text{Sim}_2(D_i, D_j)]$$ 

λ: Balance parameter between relevance and diversity (0-1)

#### Implementation Example 4: Custom MMR Implementation
    
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class MMRRetriever:
        """MMR Search Implementation"""
    
        def __init__(self, embeddings):
            self.embeddings = embeddings
    
        def mmr_search(self, query: str, documents: list,
                       lambda_param: float = 0.5, k: int = 5):
            """MMR Search
    
            Args:
                query: Search query
                documents: Candidate documents
                lambda_param: Relevance weight (0=diversity-focused, 1=relevance-focused)
                k: Number of documents to return
            """
            # Get embeddings
            query_emb = self.embeddings.embed_query(query)
            doc_texts = [doc.page_content for doc in documents]
            doc_embs = self.embeddings.embed_documents(doc_texts)
    
            # Similarity with query
            query_similarity = cosine_similarity(
                [query_emb], doc_embs
            )[0]
    
            # Selected documents
            selected_indices = []
            selected_docs = []
    
            # First document (most relevant)
            first_idx = np.argmax(query_similarity)
            selected_indices.append(first_idx)
            selected_docs.append(documents[first_idx])
    
            # Repeat until k documents
            while len(selected_indices) < k:
                mmr_scores = []
    
                for i, doc in enumerate(documents):
                    if i in selected_indices:
                        mmr_scores.append(-np.inf)
                        continue
    
                    # Relevance score
                    relevance = query_similarity[i]
    
                    # Redundancy score (max similarity with selected documents)
                    redundancy = max(
                        cosine_similarity(
                            [doc_embs[i]], [doc_embs[j]]
                        )[0][0]
                        for j in selected_indices
                    )
    
                    # MMR score
                    mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
                    mmr_scores.append(mmr)
    
                # Select document with max MMR score
                next_idx = np.argmax(mmr_scores)
                selected_indices.append(next_idx)
                selected_docs.append(documents[next_idx])
    
            return selected_docs
    
        def compare_strategies(self, query: str, documents: list, k: int = 5):
            """Comparison with different λ values"""
            strategies = {
                'Relevance-focused (λ=0.9)': 0.9,
                'Balanced (λ=0.5)': 0.5,
                'Diversity-focused (λ=0.1)': 0.1
            }
    
            results = {}
            for name, lambda_val in strategies.items():
                docs = self.mmr_search(query, documents, lambda_param=lambda_val, k=k)
                results[name] = docs
    
            return results
    
    # Usage example
    mmr_retriever = MMRRetriever(embeddings)
    
    # Get candidate documents
    query = "Machine learning evaluation methods"
    candidates = vectorstore.similarity_search(query, k=20)
    
    # Compare different strategies
    comparison = mmr_retriever.compare_strategies(query, candidates, k=5)
    
    for strategy_name, docs in comparison.items():
        print(f"\n【{strategy_name}】")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content[:60]}...")

## 3\. Hybrid Search

### 3.1 Fusion of Vector Search and Keyword Search

By combining keyword search such as BM25 with vector search, we can leverage the strengths of both.

#### Implementation Example 5: Hybrid Search Implementation
    
    
    from rank_bm25 import BM25Okapi
    import numpy as np
    
    class HybridSearcher:
        """Hybrid Search System"""
    
        def __init__(self, vectorstore, embeddings):
            self.vectorstore = vectorstore
            self.embeddings = embeddings
            self.bm25 = None
            self.documents = []
    
        def initialize_bm25(self, documents):
            """Initialize BM25 index"""
            self.documents = documents
    
            # Tokenization
            tokenized_docs = [
                doc.page_content.split() for doc in documents
            ]
    
            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
            print(f"BM25 index created: {len(documents)} documents")
    
        def bm25_search(self, query: str, k: int = 10):
            """BM25 keyword search"""
            if not self.bm25:
                raise ValueError("BM25 is not initialized")
    
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
    
            # Get Top-K
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
            """Vector similarity search"""
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
            """Hybrid search
    
            Args:
                query: Search query
                k: Number of documents to return
                vector_weight: Vector search weight (0-1)
            """
            # Execute both searches
            bm25_results = self.bm25_search(query, k=k*2)
            vector_results = self.vector_search(query, k=k*2)
    
            # Score normalization
            bm25_scores = [r['score'] for r in bm25_results]
            vector_scores = [r['score'] for r in vector_results]
    
            bm25_normalized = self._normalize_scores(bm25_scores)
            vector_normalized = self._normalize_scores(vector_scores)
    
            # Score integration
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
    
            # Sort and get Top-K
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:k]
    
            return sorted_results
    
        def _normalize_scores(self, scores):
            """Score normalization (0-1 range)"""
            scores = np.array(scores)
            if scores.max() == scores.min():
                return np.ones_like(scores)
            return (scores - scores.min()) / (scores.max() - scores.min())
    
        def compare_search_methods(self, query: str, k: int = 5):
            """Comparison of search methods"""
            results = {
                'BM25 only': self.bm25_search(query, k=k),
                'Vector search only': self.vector_search(query, k=k),
                'Hybrid (50:50)': self.hybrid_search(query, k=k, vector_weight=0.5),
                'Hybrid (Vector-focused)': self.hybrid_search(query, k=k, vector_weight=0.7)
            }
    
            return results
    
    # Usage example
    hybrid_searcher = HybridSearcher(vectorstore, embeddings)
    
    # Initialize BM25 (all documents)
    all_docs = vectorstore.similarity_search("", k=1000)  # Alternative to get all
    hybrid_searcher.initialize_bm25(all_docs)
    
    # Comparison search
    query = "machine learning evaluation metrics accuracy"
    comparison = hybrid_searcher.compare_search_methods(query, k=3)
    
    for method_name, results in comparison.items():
        print(f"\n【{method_name}】")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   {result['document'].page_content[:60]}...")

## 4\. Context Compression

### 4.1 Need for Context Compression

When passing search results to an LLM, we compress them to reduce token count and extract relevant information.

**Benefits of Compression:**

  * **Cost Reduction** : Lower API costs by reducing token count
  * **Improved Accuracy** : Better answer quality through noise removal
  * **Faster Response** : Reduced processing time

#### Implementation Example 6: Context Compression
    
    
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.chat_models import ChatOpenAI
    
    class ContextCompressor:
        """Context Compression System"""
    
        def __init__(self, llm):
            self.llm = llm
    
        def create_compression_retriever(self, base_retriever):
            """Create compression retriever"""
            # LLM-based extractor
            compressor = LLMChainExtractor.from_llm(self.llm)
    
            # Compression retriever
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
    
            return compression_retriever
    
        def extract_relevant_parts(self, query: str, documents: list):
            """Extract only relevant parts"""
            template = """Please extract only the parts relevant to the question from the following document.
            Exclude irrelevant information.
    
            Question: {query}
    
            Document:
            {document}
    
            Relevant parts:"""
    
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
            """Summarize for context"""
            combined_text = "\n\n".join([doc.page_content for doc in documents])
    
            prompt = f"""Please summarize the following documents in under {max_tokens} tokens.
            Summarize concisely while retaining important information.
    
            Documents:
            {combined_text}
    
            Summary:"""
    
            summary = self.llm(prompt)
            return summary.content
    
    # Usage example
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key="your-api-key")
    compressor = ContextCompressor(llm)
    
    # Get search results
    query = "Countermeasures against overfitting in machine learning models"
    search_results = vectorstore.similarity_search(query, k=5)
    
    # Extract relevant parts
    extracted = compressor.extract_relevant_parts(query, search_results)
    
    print("Before Compression:")
    total_chars_before = sum(len(doc.page_content) for doc in search_results)
    print(f"Total characters: {total_chars_before}")
    
    print("\nAfter Compression:")
    total_chars_after = sum(len(text) for text in extracted)
    print(f"Total characters: {total_chars_after}")
    print(f"Compression rate: {(1 - total_chars_after/total_chars_before)*100:.1f}%")
    
    # Summarized version
    summary = compressor.summarize_for_context(search_results, max_tokens=300)
    print(f"\nSummary:\n{summary}")

## Summary

  * Improved search accuracy through query decomposition and HyDE
  * Optimized search results with Cross-Encoder and MMR
  * Integrated advantages of keyword and vector with hybrid search
  * Balanced cost and accuracy with context compression

[← Chapter 2](<./chapter2-embeddings.html>) [Chapter 4 →](<./chapter4-production.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
