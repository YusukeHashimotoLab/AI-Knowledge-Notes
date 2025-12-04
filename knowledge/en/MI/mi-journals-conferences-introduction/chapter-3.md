---
title: "Chapter 3: Selecting Materials Science & Machine Learning Journals"
chapter_title: "Chapter 3: Selecting Materials Science & Machine Learning Journals"
subtitle: Optimal Submission Strategies and Decision Trees by Research Type
---

This chapter covers Selecting Materials Science & Machine Learning Journals. You will learn essential concepts and techniques.

### What You'll Learn in This Chapter

#### 📘 Level 1: Basic Understanding

  * Understand the compatibility between research types and journals
  * Classify the 21 journals from Chapter 2 by research content
  * Understand the MI-PI boundary domain

#### 📗 Level 2: Practical Skills

  * Select submission targets according to your research type using decision trees
  * Prioritize multiple candidate journals
  * Implement a journal recommendation system in Python

#### 📕 Level 3: Applied Skills

  * Develop strategic submission plans (first choice, step-down targets)
  * Design submission strategies for MI-PI boundary research
  * Build advanced journal selection systems using machine learning

## Optimal Journal Selection by Research Type

MI research takes many different forms. This section provides a systematic guide for selecting the optimal submission target from the 21 journals introduced in Chapter 2 based on your research content.

### Research Type Classification (8 Main Categories)

Recommended Journals by Research Type (1st to 3rd Choices) Research Content | 1st Choice | 2nd Choice | 3rd Choice  
---|---|---|---  
**Novel ML Methods (GNN, etc.)** | npj Computational Materials | Nature Machine Intelligence | Machine Learning: Sci. & Tech.  
**Transfer Learning, Low-Data ML** | npj Computational Materials | ACS Central Science | Computational Materials Science  
**Battery & Energy Materials MI** | Energy Storage Materials | Advanced Energy Materials | ACS Applied Mat. & Interfaces  
**Metallic & Structural Materials MI** | Acta Materialia | Computational Materials Science | Physical Review Materials  
**Molecular & Organic Materials MI** | J. Chem. Info. & Model. | J. Cheminformatics | Digital Discovery  
**Database Publication** | Materials Genome Eng. Adv. | J. Cheminformatics | Scientific Data  
**Software Tools** | J. Cheminformatics | J. Open Source Software | SoftwareX  
**Comprehensive Reviews** | Materials Today | npj Computational Materials | Computational Materials Science  
      
    
    ```mermaid
    flowchart TD
        A[Research Output] --> B{Research Type?}
    
        B -->|ML Method Novelty Focus| C[Generality of ML Method?]
        C -->|Effective Across Multiple Fields| D[Nature Machine Intelligence]
        C -->|Materials-Specific| E[npj Computational Materials]
    
        B -->|Materials Application Focus| F[Materials Field?]
        F -->|Energy Materials| G[Energy Storage Materials]
        F -->|Metallic/Structural Materials| H[Acta Materialia]
        F -->|Molecular/Organic Materials| I[J. Chem. Info. & Model.]
    
        B -->|Data/Tools| J[Type of Output?]
        J -->|Database| K[Scientific Data]
        J -->|Software| L[J. Cheminformatics]
    
        B -->|Review Paper| M[Materials Today]
    
        style A fill:#e1f5ff
        style D fill:#d4edda
        style E fill:#d4edda
        style G fill:#d4edda
        style H fill:#d4edda
        style I fill:#d4edda
        style K fill:#d4edda
        style L fill:#d4edda
        style M fill:#d4edda
    ```

## MI-PI Boundary Domain: Integration with Process Informatics

Materials Informatics (MI) and Process Informatics (PI) overlap in the optimization of materials manufacturing processes.

### Key Topics in the Boundary Domain

  * **Materials Process Optimization** : Sintering, heat treatment, thin film growth, 3D printing
  * **Process-Structure-Property Relationships** : Process conditions → microstructure → material properties
  * **Quality Control** : Real-time prediction, anomaly detection, process control

Journals Suitable for MI-PI Boundary Research Journal | Category | Suitable Research Topics  
---|---|---  
Computational Materials Science | MI-PI Boundary | Process modeling, sintering simulation  
Materials Today | MI-PI Boundary | Special issues on materials processing, manufacturing processes  
Chemical Engineering Journal | PI-leaning | Materials synthesis processes, catalytic reaction engineering  
J. Manufacturing Systems | PI-leaning | Materials processing, smart manufacturing  
  
**💡 Key Points for MI→PI Transition**

  * ✅ Emphasize process parameter optimization
  * ✅ Discuss scalability and cost reduction
  * ✅ Clearly state process economics
  * ❌ Avoid purely materials discovery topics

## Python Code Examples

Code Example 1: Research Type-based Recommendation System

Recommends optimal journals from research content using decision trees.
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Recommends optimal journals from research content using deci
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import LabelEncoder
    
    # Research type and recommended journal data
    data = {
        'research_type': ['GNN novel method', 'GNN materials-specific', 'transfer learning',
                          'energy materials', 'metallic materials', 'molecular materials',
                          'database', 'software', 'review'],
        'novelty': ['high', 'medium', 'medium', 'medium', 'medium', 'medium', 'low', 'low', 'low'],
        'experimental': ['no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'no', 'no'],
        'recommended_journal': [
            'Nature Machine Intelligence',
            'npj Computational Materials',
            'ACS Central Science',
            'Energy Storage Materials',
            'Acta Materialia',
            'J. Chem. Info. & Model.',
            'Scientific Data',
            'J. Cheminformatics',
            'Materials Today'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Label encoding
    le_type = LabelEncoder()
    le_nov = LabelEncoder()
    le_exp = LabelEncoder()
    
    X = pd.DataFrame({
        'type': le_type.fit_transform(df['research_type']),
        'novelty': le_nov.fit_transform(df['novelty']),
        'experimental': le_exp.fit_transform(df['experimental'])
    })
    y = df['recommended_journal']
    
    # Decision tree model
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # Visualize decision tree (text format)
    tree_rules = export_text(clf, feature_names=['research_type', 'novelty', 'experimental'])
    print("=== Journal Selection Decision Tree ===")
    print(tree_rules)
    
    # Recommendation example for new research
    def recommend_journal(research_type, novelty, has_experimental):
        """
        Recommend journal from research type
    
        Parameters:
        -----------
        research_type : str
            Research type (e.g., 'GNN novel method', 'energy materials')
        novelty : str
            Novelty level ('high', 'medium', 'low')
        has_experimental : str
            Experimental validation ('yes', 'no')
        """
        # Encoding
        type_enc = le_type.transform([research_type])[0]
        nov_enc = le_nov.transform([novelty])[0]
        exp_enc = le_exp.transform([has_experimental])[0]
    
        # Prediction
        X_new = [[type_enc, nov_enc, exp_enc]]
        prediction = clf.predict(X_new)[0]
    
        return prediction
    
    # Test cases
    test_cases = [
        ('GNN novel method', 'high', 'no'),
        ('energy materials', 'medium', 'yes'),
        ('database', 'low', 'no')
    ]
    
    print("\n=== Recommendation Examples ===")
    for research_type, novelty, experimental in test_cases:
        journal = recommend_journal(research_type, novelty, experimental)
        print(f"{research_type} (novelty: {novelty}, experimental: {experimental}) → {journal}")
    

Code Example 2: Submission Priority Optimization

Scores multiple candidate journals and determines optimal submission order.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Scores multiple candidate journals and determines optimal su
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    
    # Candidate journal data
    journals = {
        'journal': ['npj Computational Materials', 'Computational Materials Science',
                    'Digital Discovery', 'Machine Learning: Sci. & Tech.'],
        'impact_factor': [9.0, 3.5, 5.0, 6.8],
        'review_time_months': [2.5, 3.0, 1.5, 2.5],
        'acceptance_rate': [0.25, 0.45, 0.35, 0.40],
        'relevance_score': [0.95, 0.85, 0.80, 0.90]  # 0-1 range
    }
    
    df = pd.DataFrame(journals)
    
    def calculate_priority_score(row, weights):
        """
        Calculate priority score
    
        Parameters:
        -----------
        row : Series
            Journal data row
        weights : dict
            Weight for each factor
        """
        # Normalization
        if_norm = row['impact_factor'] / df['impact_factor'].max()
        time_norm = 1 - (row['review_time_months'] / df['review_time_months'].max())
        acc_norm = row['acceptance_rate']
        rel_norm = row['relevance_score']
    
        # Weighted score
        score = (weights['if'] * if_norm +
                 weights['time'] * time_norm +
                 weights['acceptance'] * acc_norm +
                 weights['relevance'] * rel_norm)
    
        return score
    
    # Three strategy patterns
    strategies = {
        'Conservative': {'if': 0.2, 'time': 0.1, 'acceptance': 0.5, 'relevance': 0.2},
        'Balanced': {'if': 0.3, 'time': 0.2, 'acceptance': 0.3, 'relevance': 0.2},
        'High-IF': {'if': 0.6, 'time': 0.1, 'acceptance': 0.1, 'relevance': 0.2}
    }
    
    print("=== Submission Priority (by Strategy) ===\n")
    
    for strategy_name, weights in strategies.items():
        print(f"【{strategy_name}】")
        df['priority_score'] = df.apply(lambda row: calculate_priority_score(row, weights), axis=1)
        df_sorted = df.sort_values('priority_score', ascending=False)
    
        for idx, (i, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"  {idx}. {row['journal']}")
            print(f"       Score: {row['priority_score']:.3f} (IF: {row['impact_factor']}, "
                  f"Acceptance: {row['acceptance_rate']:.0%})")
        print()
    
    # Calculate expected time to acceptance
    df['expected_time_to_accept'] = df['review_time_months'] / df['acceptance_rate']
    print("=== Expected Time to Acceptance (Average) ===")
    for _, row in df.iterrows():
        print(f"{row['journal']}: {row['expected_time_to_accept']:.1f} months")
    

### Learning Objectives Confirmation

#### 📘 Level 1: Basic Understanding

  * ✓ Understood the 8 research type categories and recommended journals 
  * ✓ Understood the MI-PI boundary domain concept 

#### 📗 Level 2: Practical Skills

  * ✓ Able to select journals using decision trees 
  * ✓ Able to implement priority scoring 

### References

  1. Butler, K. T., et al. (2018). "Machine learning for molecular and materials science". _Nature_ , 559(7715), pp. 547-555.
  2. Himanen, L., et al. (2019). "Data-driven materials science: status, challenges, and perspectives". _Advanced Science_ , 6(21), pp. 1-23.
  3. Morgan, D., & Jacobs, R. (2020). "Opportunities and challenges for machine learning in materials science". _Annual Review of Materials Research_ , 50, pp. 71-103.
  4. Agrawal, A., & Choudhary, A. (2016). "Perspective: Materials informatics and big data". _APL Materials_ , 4(5), pp. 053208-1 to 053208-10.
  5. Murdock, R. J., et al. (2020). "Is domain knowledge necessary for machine learning materials properties?". _Integrating Materials and Manufacturing Innovation_ , 9, pp. 221-227.

[← Previous: 21 MI Specialized Journals](<chapter-2.html>) [Back to Contents](<index.html>) [Next: International & Domestic Conferences →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
