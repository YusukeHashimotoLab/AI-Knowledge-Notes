---
title: "Chapter 1: RAG Fundamentals"
chapter_title: "Chapter 1: RAG Fundamentals"
---

This chapter covers the fundamentals of RAG Fundamentals, which 1. what is rag. You will learn essential concepts and techniques.

## 1\. What is RAG

### 1.1 RAG Overview

RAG (Retrieval-Augmented Generation) is a technique for incorporating external knowledge into large language models (LLMs). By combining the generative capabilities of LLMs with retrieval systems, it enables responses based on up-to-date information and specialized knowledge.

**Key Benefits:**

  * **Utilizing Current Information** : Access to information not included in the model's training data
  * **Reducing Hallucinations** : Improved accuracy through responses based on retrieval results
  * **Cost Efficiency** : Adding knowledge without the need for fine-tuning
  * **Transparency** : Clarifying information sources and providing verifiable responses

### 1.2 RAG Architecture

A RAG system consists of three main components:

**RAG Pipeline:**

  1. **Index Building** : Document loading → Chunking → Embedding → Vector DB storage
  2. **Retrieval** : Query → Embedding → Similar document retrieval
  3. **Generation** : Retrieval results + Query → LLM prompt → Response generation

#### Implementation Example 1: Basic RAG Architecture
    
    
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
            """Index documents"""
            documents = []
            for path in file_paths:
                loader = TextLoader(path, encoding='utf-8')
                documents.extend(loader.load())
    
            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(documents)
    
            # Create vector store
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            print(f"Indexing complete: {len(splits)} chunks")
    
        def query(self, question):
            """Question answering"""
            if not self.vectorstore:
                raise ValueError("Documents not indexed")
    
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
    
            result = qa_chain({"query": question})
            return result["result"]
    
    # Usage example
    rag = SimpleRAG(api_key="your-api-key")
    rag.index_documents(["docs/manual.txt", "docs/faq.txt"])
    answer = rag.query("What is the product warranty period?")
    print(answer)

## 2\. Document Processing

### 2.1 Document Loaders

Various loaders are available for loading documents in different formats:

#### Implementation Example 2: Multi-Format Loader
    
    
    from langchain.document_loaders import (
        TextLoader, PDFLoader, CSVLoader,
        UnstructuredMarkdownLoader, UnstructuredHTMLLoader
    )
    import os
    
    class UniversalDocumentLoader:
        """Multi-format document loader"""
    
        LOADERS = {
            '.txt': TextLoader,
            '.pdf': PDFLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
            '.html': UnstructuredHTMLLoader,
        }
    
        def load_documents(self, directory):
            """Load all documents in directory"""
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
    
                            # Add metadata
                            for doc in docs:
                                doc.metadata['source_file'] = file
                                doc.metadata['file_type'] = ext
    
                            documents.extend(docs)
                            print(f"Loaded: {file} ({len(docs)} documents)")
                        except Exception as e:
                            print(f"Error ({file}): {e}")
    
            return documents
    
    # Usage example
    loader = UniversalDocumentLoader()
    documents = loader.load_documents("./knowledge_base")
    print(f"Total documents: {len(documents)}")

### 2.2 Metadata Management

Proper metadata management improves retrieval accuracy and enables filtering.

#### Implementation Example 3: Metadata Enrichment
    
    
    from datetime import datetime
    from langchain.schema import Document
    import hashlib
    
    class MetadataEnricher:
        """Document metadata enrichment"""
    
        def enrich_documents(self, documents):
            """Add and enhance metadata"""
            enriched = []
    
            for doc in documents:
                # Basic metadata
                metadata = doc.metadata.copy()
    
                # Timestamp
                metadata['indexed_at'] = datetime.now().isoformat()
    
                # Document length
                metadata['char_count'] = len(doc.page_content)
                metadata['word_count'] = len(doc.page_content.split())
    
                # Hash value (for duplicate detection)
                content_hash = hashlib.md5(
                    doc.page_content.encode()
                ).hexdigest()
                metadata['content_hash'] = content_hash
    
                # Category estimation (simplified)
                metadata['category'] = self._estimate_category(doc.page_content)
    
                enriched.append(Document(
                    page_content=doc.page_content,
                    metadata=metadata
                ))
    
            return enriched
    
        def _estimate_category(self, text):
            """Estimate category from content"""
            keywords = {
                'technical': ['API', 'code', 'implementation', 'function'],
                'business': ['contract', 'pricing', 'sales', 'business'],
                'support': ['issue', 'error', 'trouble', 'support']
            }
    
            text_lower = text.lower()
            scores = {}
    
            for category, terms in keywords.items():
                score = sum(1 for term in terms if term.lower() in text_lower)
                scores[category] = score
    
            return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    # Usage example
    enricher = MetadataEnricher()
    enriched_docs = enricher.enrich_documents(documents)
    
    # Filtering by metadata
    technical_docs = [
        doc for doc in enriched_docs
        if doc.metadata.get('category') == 'technical'
    ]
    print(f"Technical documents: {len(technical_docs)}")

## 3\. Chunking Strategies

### 3.1 Fixed-Size Chunking

The simplest method that divides documents by a specified number of characters or tokens.

#### Implementation Example 4: Fixed-Size Chunking
    
    
    from langchain.text_splitter import CharacterTextSplitter
    import tiktoken
    
    class FixedSizeChunker:
        """Fixed-size chunking"""
    
        def __init__(self, chunk_size=500, chunk_overlap=50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
        def chunk_by_characters(self, text):
            """Character-based splitting"""
            splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            return splitter.split_text(text)
    
        def chunk_by_tokens(self, text):
            """Token-based splitting"""
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_text(text)
    
        def analyze_chunks(self, chunks):
            """Chunk statistics"""
            stats = {
                'total_chunks': len(chunks),
                'avg_length': sum(len(c) for c in chunks) / len(chunks),
                'min_length': min(len(c) for c in chunks),
                'max_length': max(len(c) for c in chunks),
            }
            return stats
    
    # Usage example
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    
    text = """Long document text..."""
    chunks = chunker.chunk_by_tokens(text)
    stats = chunker.analyze_chunks(chunks)
    
    print(f"Number of chunks: {stats['total_chunks']}")
    print(f"Average length: {stats['avg_length']:.1f} characters")

### 3.2 Semantic Chunking

An advanced technique that divides documents considering semantic coherence.

#### Implementation Example 5: Semantic Chunking
    
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class SemanticChunker:
        """Semantic chunking"""
    
        def __init__(self, embeddings, similarity_threshold=0.7):
            self.embeddings = embeddings
            self.threshold = similarity_threshold
    
        def chunk_by_similarity(self, text, min_chunk_size=100):
            """Similarity-based splitting"""
            # First split into sentences
            sentences = self._split_sentences(text)
    
            if len(sentences) <= 1:
                return [text]
    
            # Get embeddings for each sentence
            sentence_embeddings = self.embeddings.embed_documents(sentences)
    
            # Group based on similarity
            chunks = []
            current_chunk = [sentences[0]]
    
            for i in range(1, len(sentences)):
                # Calculate similarity with previous sentence
                sim = cosine_similarity(
                    [sentence_embeddings[i-1]],
                    [sentence_embeddings[i]]
                )[0][0]
    
                if sim >= self.threshold:
                    current_chunk.append(sentences[i])
                else:
                    # Start new chunk
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = [sentences[i]]
    
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
    
            return chunks
    
        def _split_sentences(self, text):
            """Sentence splitting (simplified)"""
            import re
            sentences = re.split(r'[。！？\n]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    # Usage example
    from langchain.embeddings import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(openai_api_key="your-api-key")
    semantic_chunker = SemanticChunker(embeddings, similarity_threshold=0.75)
    
    text = """Machine learning is a branch of artificial intelligence. It learns from data.
    Deep learning uses neural networks. It excels at image recognition.
    Natural language processing deals with text. It enables translation and summarization."""
    
    chunks = semantic_chunker.chunk_by_similarity(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")

### 3.3 Hierarchical Chunking

Performs hierarchical splitting that considers document structure (headings, paragraphs, etc.).

#### Implementation Example 6: Hierarchical Chunking
    
    
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    from typing import List, Dict
    
    class HierarchicalChunker:
        """Hierarchical chunking"""
    
        def chunk_markdown(self, markdown_text):
            """Structure-based splitting for Markdown"""
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]
    
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            splits = splitter.split_text(markdown_text)
    
            # Create chunks with hierarchy information
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
            """Build hierarchy path"""
            parts = []
            for level in ['H1', 'H2', 'H3']:
                if level in metadata:
                    parts.append(metadata[level])
            return ' > '.join(parts)
    
        def chunk_with_context(self, text, chunk_size=500):
            """Retain parent chunk context"""
            from langchain.text_splitter import RecursiveCharacterTextSplitter
    
            # Create parent chunks
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 3,
                chunk_overlap=0
            )
            parent_chunks = parent_splitter.split_text(text)
    
            # Create child chunks (preserving parent information)
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
                        'parent_summary': parent[:200] + '...'  # Parent summary
                    })
    
            return chunks_with_context
    
    # Usage example
    hierarchical_chunker = HierarchicalChunker()
    
    markdown_text = """
    # Data Science
    Learn the fundamentals of data analysis.
    
    ## Statistics
    ### Descriptive Statistics
    Study mean, variance, and standard deviation.
    
    ### Inferential Statistics
    Explain hypothesis testing and confidence intervals.
    
    ## Machine Learning
    ### Supervised Learning
    Cover regression and classification algorithms.
    """
    
    chunks = hierarchical_chunker.chunk_markdown(markdown_text)
    for chunk in chunks:
        print(f"Hierarchy: {chunk['hierarchy']}")
        print(f"Content: {chunk['content'][:50]}...")
        print()

**Choosing Chunking Strategies:**

  * **Fixed-Size** : Simple and fast, applicable to general documents
  * **Semantic** : Use when semantic consistency is important
  * **Hierarchical** : Optimal for structured documents (technical documentation, manuals)

## Summary

  * RAG is a powerful LLM augmentation technique combining retrieval and generation
  * Proper loaders and metadata management are crucial in document processing
  * Choose chunking strategies based on use cases
  * Understand three approaches: fixed-size, semantic, and hierarchical

[← Series Index](<./index.html>) [Chapter 2 →](<./chapter2-embeddings.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
