---
title: "Chapter 1: Clustering Algorithms"
chapter_title: "Chapter 1: Clustering Algorithms"
subtitle: K-Means, DBSCAN, and Hierarchical Clustering
---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/unsupervised-learning-introduction/chapter1-clustering.html>) | Last sync: 2025-11-16

[ML Dojo](<../index.html>) > [Unsupervised Learning](<index.html>) > Ch1

## 1.1 K-Means Clustering

K-Means partitions data into K clusters by minimizing within-cluster variance.

**📐 K-Means Objective:** $$\min_C \sum_{i=1}^K \sum_{x \in C_i} \|x - \mu_i\|^2$$ where $\mu_i$ is centroid of cluster $C_i$

### 💻 Code Example 1: K-Means Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    class ClusteringAnalysis:
        """Comprehensive clustering analysis"""
        
        def __init__(self, algorithm='kmeans', n_clusters=3):
            self.algorithm = algorithm
            self.n_clusters = n_clusters
            self.model = None
        
        def fit(self, X):
            """Fit clustering model"""
            if self.algorithm == 'kmeans':
                self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            elif self.algorithm == 'dbscan':
                self.model = DBSCAN(eps=0.5, min_samples=5)
            elif self.algorithm == 'hierarchical':
                self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
            
            self.labels_ = self.model.fit_predict(X)
            return self
        
        def evaluate(self, X):
            """Evaluate clustering quality"""
            metrics = {}
            
            # Silhouette score
            if len(np.unique(self.labels_)) > 1:
                metrics['silhouette'] = silhouette_score(X, self.labels_)
                metrics['davies_bouldin'] = davies_bouldin_score(X, self.labels_)
            
            # Inertia (K-Means only)
            if hasattr(self.model, 'inertia_'):
                metrics['inertia'] = self.model.inertia_
            
            metrics['n_clusters'] = len(np.unique(self.labels_))
            
            return metrics
        
        def find_optimal_k(self, X, k_range=(2, 10)):
            """Find optimal number of clusters using elbow method"""
            inertias = []
            silhouettes = []
            
            for k in range(k_range[0], k_range[1] + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)
                
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X, labels))
            
            # Plot elbow curve
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            k_values = range(k_range[0], k_range[1] + 1)
            ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Clusters (K)')
            ax1.set_ylabel('Inertia')
            ax1.set_title('Elbow Method')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(k_values, silhouettes, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Number of Clusters (K)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return k_values, inertias, silhouettes
    
    # Example usage
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # K-Means clustering
    clustering = ClusteringAnalysis(algorithm='kmeans', n_clusters=4)
    clustering.fit(X)
    
    metrics = clustering.evaluate(X)
    print(f"Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}")
    
    # Find optimal K
    k_values, inertias, silhouettes = clustering.find_optimal_k(X, k_range=(2, 10))

## 1.2-1.7 Additional Clustering Methods

DBSCAN for density-based clustering, hierarchical clustering, Gaussian mixture models, evaluation metrics.

### 💻 Code Examples 2-7
    
    
    # DBSCAN implementation and parameter tuning
    # Hierarchical clustering with dendrograms
    # Gaussian Mixture Models (GMM)
    # Cluster evaluation metrics
    # Dimensionality reduction + clustering
    # Real-world applications
    # See complete implementations

## 📝 Exercises

  1. Apply K-Means to Iris dataset and determine optimal K.
  2. Compare K-Means vs DBSCAN on moon-shaped data.
  3. Create dendrogram for hierarchical clustering.
  4. Implement GMM and compare with K-Means.
  5. Evaluate clustering using multiple metrics (silhouette, DB index, Calinski-Harabasz).

## Summary

  * K-Means: partitional clustering minimizing within-cluster variance
  * DBSCAN: density-based, finds arbitrary shapes, handles noise
  * Hierarchical: creates dendrogram, agglomerative or divisive
  * GMM: probabilistic clustering with Gaussian distributions
  * Evaluation: silhouette score, Davies-Bouldin index, elbow method
  * Applications: customer segmentation, image compression, anomaly detection

[← Overview](<index.html>) [Ch2: Dimensionality Reduction →](<chapter2-dimensionality-reduction.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
