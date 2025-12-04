---
title: Materials Property Mapping Introduction Series v1.0
chapter_title: Materials Property Mapping Introduction Series v1.0
---

# Materials Property Mapping Introduction Series v1.0

**Visualization and exploration of high-dimensional materials space using GNN and dimensionality reduction**

## Series Overview

This series is a comprehensive 4-chapter educational program teaching practical skills to effectively visualize high-dimensional materials space and accelerate materials discovery by combining Graph Neural Networks (GNN) representation learning with dimensionality reduction techniques.

**Materials property mapping** is a technology that represents thousands to tens of thousands of materials as points in high-dimensional property space, projects them into 2D or 3D space, and visualizes them to intuitively understand material similarity, structure-property relationships, and regions to explore. By using embeddings automatically learned by GNNs, which capture structural information that cannot be captured by conventional composition-based descriptors, more intrinsic materials space mapping becomes possible.

### Why This Series Is Needed

**Background and Challenges** : Finding optimal materials from tens of thousands of candidates in materials discovery is extremely difficult. Traditional approaches relied on human experience and intuition, but intuition does not work well in high-dimensional property spaces, leading to many promising materials being overlooked. In particular, even if compositions are similar, properties can differ significantly if crystal structures differ, and conversely, even if compositions differ, similar structures can exhibit similar properties. Appropriate visualization and mapping techniques are essential to understand such complex relationships and efficiently explore materials space.

**What You Will Learn in This Series** : This series provides systematic learning from the basics of materials space visualization to dimensionality reduction techniques such as PCA, t-SNE, and UMAP, materials representation learning using GNN, and construction of practical materials mapping systems that integrate both, with 70 executable Python code examples. You will master the complete end-to-end workflow from acquiring real data from Materials Project API, through GNN training, embedding extraction, dimensionality reduction, clustering, to materials recommendation systems.

**Features:** \- ✅ **Practice-oriented** : 70 executable code examples, Materials Project API integration \- ✅ **Progressive structure** : Basic visualization → Dimensionality reduction → GNN → Integrated system \- ✅ **Latest technologies** : Combination of CGCNN, MEGNet, SchNet with UMAP, t-SNE \- ✅ **Interactive visualization** : Exploratory data analysis using Plotly and Bokeh \- ✅ **Practical applications** : Materials recommendation system, clustering analysis, extrapolation region detection

**Total Learning Time** : 90-110 minutes (including code execution and exercises)

**Target Audience** : \- Graduate students in materials science/chemistry (master's and doctoral programs) \- R&D engineers in industry (materials development, data analysis) \- Computational materials scientists (experience with DFT, MD simulations) \- Data scientists (aiming to apply to materials/chemistry fields)

* * *

## How to Learn

### Recommended Learning Order
    
    
    ```mermaid
    flowchart TD
        A[Chapter 1: Fundamentals of Materials Space Visualization] --> B[Chapter 2: Dimensionality Reduction Methods]
        B --> C[Chapter 3: Materials Representation Learning with GNN]
        C --> D[Chapter 4: Practical - Building Integrated Systems]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

**For beginners (first time learning GNN and dimensionality reduction):** \- Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 (all chapters recommended) \- Duration: 90-110 minutes \- Prerequisites: GNN Introduction Series or deep learning basics, advanced Python level

**For intermediate learners (with GNN experience):** \- Chapter 2 → Chapter 3 → Chapter 4 \- Duration: 70-90 minutes \- Chapter 1 can be skipped (refer as needed)

**For practical skill enhancement (implementation-focused):** \- Chapter 3 (GNN implementation) → Chapter 4 (integrated system) \- Duration: 50-70 minutes \- Refer to Chapters 1 and 2 for theory as needed

* * *

## Chapter Details

### [Chapter 1: Fundamentals of Materials Space Visualization](<chapter-1.html>)

**Difficulty** : Introductory **Reading Time** : 20-25 minutes **Code Examples** : 5

#### Learning Content

  1. **What is Materials Space** \- Dimensions of property space and challenges of high-dimensional data \- Concept of representing materials as points \- The curse of dimensionality and visualization limits

  2. **Preparing Materials Data** \- Calculating basic statistics \- Property distribution histograms \- Data preprocessing and cleaning

  3. **Basic Visualization with 2D Scatter Plots** \- Scatter plots between two properties \- Pair plots (multivariate correlations) \- Color coding and size mapping

  4. **Correlation Matrix Visualization** \- Correlation analysis using heatmaps \- Identifying property pairs with strong correlations

#### Learning Objectives

After completing this chapter, you will be able to:

  * ✅ Explain the concept of materials space and challenges of high-dimensional data
  * ✅ Visualize basic statistics and data distributions
  * ✅ Analyze relationships between properties using scatter plots and pair plots
  * ✅ Find important property combinations from correlation matrices

**[Read Chapter 1 →](<chapter-1.html>)**

* * *

### [Chapter 2: Mapping Materials Space with Dimensionality Reduction Methods](<chapter-2.html>)

**Difficulty** : Beginner to Intermediate **Reading Time** : 25-30 minutes **Code Examples** : 15

#### Learning Content

  1. **Principal Component Analysis (PCA)** \- PCA basic implementation and variance contribution analysis \- Scree plot \- Loading plot (biplot)

  2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)** \- t-SNE implementation and effect of perplexity parameter \- Clustering result visualization \- Neighborhood preservation rate evaluation

  3. **UMAP (Uniform Manifold Approximation and Projection)** \- UMAP implementation and n_neighbors parameter optimization \- Creating density maps \- 3D visualization with 3D UMAP

  4. **Method Comparison** \- Performance comparison of PCA vs t-SNE vs UMAP \- Quantitative evaluation using neighborhood preservation rate \- Method selection according to use cases

  5. **Interactive Visualization** \- 3D visualization with Plotly \- Interactive scatter plots with Bokeh \- Visualization of dimensionality reduction process through animation

#### Learning Objectives

  * ✅ Understand the principles and implementation of PCA, t-SNE, and UMAP
  * ✅ Appropriately adjust parameters for each method
  * ✅ Compare methods using evaluation metrics such as neighborhood preservation rate
  * ✅ Create interactive visualizations with Plotly and Bokeh
  * ✅ Select optimal dimensionality reduction method according to purpose

**[Read Chapter 2 →](<chapter-2.html>)**

* * *

### [Chapter 3: Materials Representation Learning with GNN](<chapter-3.html>)

**Difficulty** : Intermediate to Advanced **Reading Time** : 25-30 minutes **Code Examples** : 20 (all executable)

#### Learning Content

  1. **Graph Representation of Materials** \- Conversion from crystal structure to graph \- Atomic feature encoding \- PyTorch Geometric data structure

  2. **Crystal Graph Convolutional Neural Network (CGCNN)** \- CGCNN convolution layer implementation \- Complete CGCNN model construction \- Training loop and Early Stopping

  3. **MEGNet (MatErials Graph Network)** \- MEGNet block considering global state \- Complete MEGNet model \- Comparison with CGCNN

  4. **SchNet** \- Continuous filter convolution layer \- Distance embedding using Gaussian basis functions \- Complete SchNet model

  5. **Embedding Visualization and Analysis** \- Visualizing GNN embeddings with UMAP \- Comparing multiple models with t-SNE \- Clustering and property analysis \- Quantitative evaluation of embedding quality

#### Learning Objectives

  * ✅ Understand and implement methods for representing materials as graphs
  * ✅ Implement and compare performance of CGCNN, MEGNet, and SchNet
  * ✅ Extract embeddings obtained from GNN
  * ✅ Visualize embeddings with UMAP and t-SNE, and analyze clusters
  * ✅ Evaluate embedding quality using metrics such as silhouette score

**[Read Chapter 3 →](<chapter-3.html>)**

* * *

### [Chapter 4: Practical - Materials Mapping with GNN + Dimensionality Reduction](<chapter-4.html>)

**Difficulty** : Intermediate to Advanced **Reading Time** : 30-35 minutes **Code Examples** : 30 (end-to-end implementation)

#### Learning Content

  1. **Environment Setup and Data Collection** \- Materials Project API configuration \- Real data acquisition and exploratory analysis

  2. **Building Graph Datasets** \- Optimized conversion from crystal structure to graph \- Creating custom dataset classes \- DataLoader construction

  3. **GNN Model Training** \- Improved CGCNN model \- Training loop with Early Stopping \- Evaluation on test data

  4. **Embedding Extraction and Dimensionality Reduction** \- Extracting embeddings from all data \- Dimensionality reduction using PCA, UMAP, t-SNE \- Comparing dimensionality reduction methods

  5. **Materials Space Analysis** \- Clustering and property analysis \- Creating density maps \- Searching for neighboring materials \- Implementing materials recommendation system

  6. **Interactive Visualization** \- 3D UMAP with Plotly \- Interactive scatter plots with Bokeh \- Dashboard with Dash (optional)

  7. **Advanced Analysis and Applications** \- Voronoi tessellation \- Visualizing property gradients \- Detecting extrapolation regions \- Generating comprehensive reports

#### Learning Objectives

  * ✅ Acquire real data from Materials Project API
  * ✅ Build complete GNN training pipeline
  * ✅ Visualize trained GNN embeddings with dimensionality reduction
  * ✅ Implement clustering and materials recommendation system
  * ✅ Create interactive exploration systems with Plotly and Bokeh
  * ✅ Apply to actual materials design tasks

**[Read Chapter 4 →](<chapter-4.html>)**

* * *

## Overall Learning Outcomes

Upon completing this series, you will acquire the following skills and knowledge:

### Knowledge Level (Understanding)

  * ✅ Can explain concepts of materials space and high-dimensional data visualization
  * ✅ Understand principles and usage of PCA, t-SNE, and UMAP
  * ✅ Understand features and implementation of CGCNN, MEGNet, and SchNet
  * ✅ Can explain advantages of combining GNN embeddings with dimensionality reduction

### Practical Skills (Doing)

  * ✅ Can acquire materials data from Materials Project API
  * ✅ Can convert crystal structures to graph data
  * ✅ Can implement and train CGCNN, MEGNet, and SchNet
  * ✅ Can extract GNN embeddings and visualize with UMAP/t-SNE
  * ✅ Can build clustering and materials recommendation systems
  * ✅ Can create interactive visualizations with Plotly and Bokeh

### Application Ability (Applying)

  * ✅ Can build mapping systems for new materials datasets
  * ✅ Can obtain materials design insights from cluster analysis
  * ✅ Can suggest next materials to synthesize using materials recommendation system
  * ✅ Can detect extrapolation regions and evaluate prediction reliability
  * ✅ Can generate comprehensive analysis reports and utilize in research

* * *

## Recommended Learning Patterns

### Pattern 1: Complete Mastery (For Beginners)

**Target** : Those learning GNN and dimensionality reduction for the first time **Duration** : 2-3 weeks **Approach** :
    
    
    Week 1:
    - Day 1-2: Chapter 1 (Fundamentals of Materials Space Visualization)
    - Day 3-5: Chapter 2 (Dimensionality Reduction Methods)
    - Day 6-7: Chapter 2 exercises, method comparison
    
    Week 2:
    - Day 1-3: Chapter 3 (GNN implementation + embedding extraction)
    - Day 4-7: Chapter 3 (multiple model implementation + visualization)
    
    Week 3:
    - Day 1-4: Chapter 4 (building integrated system)
    - Day 5-7: Chapter 4 (advanced analysis + report generation)
    

**Deliverables** : \- Materials mapping system with Materials Project data \- Interactive 3D visualization (Plotly) \- Materials recommendation system implementation \- GitHub repository (all code + README)

### Pattern 2: Intensive (For Experienced Learners)

**Target** : Those with GNN and machine learning basics **Duration** : 1 week **Approach** :
    
    
    Day 1: Chapter 2 (dimensionality reduction method implementation)
    Day 2-3: Chapter 3 (GNN implementation)
    Day 4-6: Chapter 4 (building integrated system)
    Day 7: Application to original data
    

**Deliverables** : \- Performance comparison of multiple GNN models (CGCNN, MEGNet, SchNet) \- Integrated materials mapping system \- Interactive dashboard

* * *

## FAQ (Frequently Asked Questions)

### Q1: Can I understand without completing the GNN Introduction Series?

**A** : **Basic GNN knowledge is a prerequisite**. Chapters 3 and 4 assume GNN implementation experience. Strongly recommend completing Chapters 1-3 of the GNN Introduction Series first. Minimum required skills: PyTorch Geometric basics, message passing concept, graph data handling.

### Q2: Can I learn without a Materials Project API key?

**A** : **Yes, you can learn without an API key**. Chapter 4 provides dummy data generation code, so all code examples can be executed without an API key. However, if you want to try with real data, create a free account at [Materials Project](<https://materialsproject.org/>) and obtain an API key (takes about 5 minutes).

### Q3: How much computing resources (GPU) are required?

**A** : **GPU is recommended for training but CPU is also possible** :

**CPU only** : \- Possible: Training with dummy data (1000 materials) \- Training time: 10-30 minutes (CGCNN) \- Google Colab free tier (CPU) is sufficient

**GPU recommended** : \- Real data training (10,000+ materials) \- Recommended GPU: NVIDIA RTX 3060 (12GB VRAM) or higher \- Training time: 5-15 minutes (CGCNN) \- Google Colab Pro (GPU) is convenient

**Google Colab free tier is sufficient** for this series.

### Q4: Should I use UMAP or t-SNE?

**A** : **UMAP is recommended, but it depends on the purpose** :

**UMAP advantages** : \- Large-scale data (10,000+ points) \- Limited computation time \- Want to preserve global structure \- 3D visualization needed

**t-SNE advantages** : \- Small-scale data (1,000 or fewer points) \- Emphasizing cluster structure is important \- Want to use commonly used method in papers

**Best practice** : Try both and compare, select according to purpose.

### Q5: How should I interpret clustering results?

**A** : **Start by comparing average property values for each cluster** :

  1. Calculate average property values for each cluster (see Chapter 4 code example 16)
  2. Identify characteristic clusters (e.g., high band gap cluster)
  3. Investigate materials within clusters in detail
  4. Confirm structural similarity (same crystal system, similar composition, etc.)
  5. Determine direction for new materials exploration

Important to note: **Clusters are just statistical groups** and physical meaning requires separate validation.

### Q6: Why is detecting extrapolation regions important?

**A** : **To evaluate prediction reliability** :

  * **Within training data range** : GNN predictions are highly accurate (MAE < 0.1 eV)
  * **Extrapolation region** : Prediction accuracy may decrease (MAE > 0.3 eV)

Detecting extrapolation regions allows: \- Distinguish reliable predictions from uncertain ones \- Identify materials requiring additional experiments \- Utilize in active learning for next sample selection

Implementation method explained in Chapter 4 code example 29.

### Q7: Can I become a materials mapping expert just from this series?

**A** : This series targets "bridging from intermediate to advanced." To reach expert level:

  1. Build foundation with this series (2-3 weeks)
  2. Execute projects with original data (1-3 months) \- Build mapping system with your research data \- Experiment with new GNN architectures
  3. Read papers and track latest technologies (ongoing) \- Latest papers on GNN + UMAP \- Trends in Materials Informatics field
  4. Conference presentations and paper writing (6 months~1 year)

1-2 years of continuous learning and practice required. This series is optimal as a starting point.

### Q8: How accurate is the materials recommendation system?

**A** : **Depends on data and purpose, but general indicators** :

  * **Top-5 recommendation precision** : 60-80% (within ±10% of target property)
  * **Top-10 recommendation precision** : 70-90%
  * **Acceleration of new materials discovery** : Reduce experiments by 50-90%

Points for improving accuracy: 1\. Improve GNN model prediction accuracy (target R² > 0.9) 2\. Sufficient training data (10,000+ materials) 3\. Appropriate embedding dimensions (64-128 dimensions) 4\. Adjust distance metric (cosine similarity, Euclidean distance)

* * *

## Prerequisites and Related Series

### Prerequisites

**Required** : \- [ ] **GNN Basics** : Message passing, graph convolution, PyTorch Geometric \- [ ] **Advanced Python** : Classes, generators, decorators, type hints \- [ ] **Machine Learning** : Training loops, overfitting, evaluation metrics

**Recommended** : \- [ ] **Linear Algebra** : Matrix operations, eigenvalue decomposition, PCA \- [ ] **Materials Science** : Crystal structures, material properties, band gap \- [ ] **Data Visualization** : Matplotlib, Seaborn, Plotly

### Prerequisite Series

  1. **[GNN Introduction Series](<../../ML/gnn-introduction/index.html>)** (Intermediate) \- Content: GNN basic theory, PyTorch Geometric, CGCNN/SchNet implementation \- Learning time: 110-130 minutes \- Why recommended: Systematically learn GNN basics \- **Required** : Necessary to understand Chapters 3 and 4

### Related Series

  1. **[Bayesian Optimization Introduction](<../bayesian-optimization-introduction/index.html>)** (Intermediate) \- Relevance: Efficient materials search utilizing materials mapping results \- Link: [../bayesian-optimization-introduction/index.html](<../bayesian-optimization-introduction/index.html>)

  2. **[Active Learning Introduction](<../active-learning-introduction/index.html>)** (Intermediate) \- Relevance: Next sample selection in embedding space \- Link: [../active-learning-introduction/index.html](<../active-learning-introduction/index.html>)

  3. ****(Introductory) \- Relevance: Overall picture and basics of materials informatics \- Link:

* * *

## Tools and Resources

### Main Tools

Tool Name | Purpose | License | Installation  
---|---|---|---  
PyTorch Geometric | GNN implementation | MIT | `pip install torch-geometric`  
UMAP | Dimensionality reduction | BSD-3 | `pip install umap-learn`  
scikit-learn | Machine learning/dimensionality reduction | BSD-3 | `pip install scikit-learn`  
Plotly | Interactive visualization | MIT | `pip install plotly`  
Bokeh | Interactive visualization | BSD-3 | `pip install bokeh`  
pymatgen | Materials structure manipulation | MIT | `pip install pymatgen`  
mp-api | Materials Project API | BSD | `pip install mp-api`  
  
### Databases

Database Name | Description | Data Count | Access  
---|---|---|---  
Materials Project | Crystal structure and DFT calculation data | 140,000 materials | <https://materialsproject.org/>  
AFLOW | High-throughput computation data | 3,500,000 materials | <http://aflowlib.org/>  
OQMD | Quantum materials database | 1,000,000 materials | <http://oqmd.org/>  
  
### Learning Resources

**Papers and Reviews** : \- Xie, T. & Grossman, J. C. (2018). "Crystal Graph Convolutional Neural Networks". _Physical Review Letters_. \- McInnes, L. et al. (2018). "UMAP: Uniform Manifold Approximation and Projection". _arXiv:1802.03426_. \- van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE". _JMLR_.

**Online Resources** : \- [UMAP Documentation](<https://umap-learn.readthedocs.io/>) \- [Plotly Python Graphing Library](<https://plotly.com/python/>) \- [Materials Project API Docs](<https://docs.materialsproject.org/>)

* * *

## Next Steps

### Recommended Actions After Series Completion

**Immediate (within 1-2 weeks):** 1\. ✅ Create portfolio on GitHub 2\. ✅ Build mapping system with original data 3\. ✅ Publish interactive dashboard 4\. ✅ Write blog article (Qiita, Medium)

**Short-term (1-3 months):** 1\. ✅ Accelerate materials search by combining with Bayesian optimization 2\. ✅ Efficiently collect data with active learning 3\. ✅ Conference presentation (Japan Society of Materials Science, MRS) 4\. ✅ Proceed to [Bayesian Optimization Introduction Series](<../bayesian-optimization-introduction/index.html>)

**Long-term (6 months or more):** 1\. ✅ Paper writing (_npj Computational Materials_ , _Chemistry of Materials_) 2\. ✅ Industrial application projects 3\. ✅ Develop new GNN + dimensionality reduction methods

* * *

## License and Terms of Use

**CC BY 4.0** (Creative Commons Attribution 4.0 International)

### What You Can Do

  * ✅ Free viewing and downloading
  * ✅ Educational use (classes, training, study groups)
  * ✅ Modifications and derivative works
  * ✅ Commercial use (corporate training, paid courses)

### Conditions

  * 📌 Author credit display: "Dr. Yusuke Hashimoto, Tohoku University - AI Terakoya"
  * 📌 License inheritance (remain CC BY 4.0)

* * *

## Let's Get Started!

Ready? Start with Chapter 1 and begin your journey into the world of materials property mapping!

**[Chapter 1: Fundamentals of Materials Space Visualization →](<chapter-1.html>)**

* * *

## Version History

Version | Date | Changes | Author  
---|---|---|---  
1.0 | 2025-10-20 | Initial release | Dr. Yusuke Hashimoto  
  
* * *

**Your materials mapping learning journey starts here!**
    
    
    ```mermaid
    flowchart LR
        subgraph "Materials Mapping Workflow"
        A[Materials Data\nMP/AFLOW] --> B[GNN Training\nCGCNN/MEGNet]
        B --> C[Embedding\nExtraction]
        C --> D[Dimensionality\nReduction]
        D --> E[Visualization &\nAnalysis]
        E --> F[Materials\nDiscovery]
        end
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#ffebee
    ```
