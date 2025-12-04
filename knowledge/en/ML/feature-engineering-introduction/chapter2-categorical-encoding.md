---
title: "Chapter 2: Categorical Variable Encoding"
chapter_title: "Chapter 2: Categorical Variable Encoding"
subtitle: Numerical transformation techniques for qualitative data in machine learning models
reading_time: 20-25 minutes
difficulty: Beginner to Intermediate
code_examples: 12
exercises: 5
version: 1.0
---

This chapter covers Categorical Variable Encoding. You will learn differences between Label Encoding and Appropriately use Target Encoding (Mean Encoding).

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the types and characteristics of categorical variables
  * ✅ Master the principles and implementation of One-Hot Encoding
  * ✅ Explain the differences between Label Encoding and Ordinal Encoding
  * ✅ Appropriately use Target Encoding (Mean Encoding)
  * ✅ Understand the concept and applications of Frequency Encoding
  * ✅ Implement Binary Encoding and Hashing Trick
  * ✅ Properly select among various encoding methods

* * *

## 2.1 What are Categorical Variables?

### Definition

**Categorical Variables** are variables that represent qualitative data with discrete categories or levels.

> "Variables that may be represented by numbers but have no mathematical meaning (such as magnitude relationships or addition) in those values themselves"

### Classification of Categorical Variables

Type | Description | Examples  
---|---|---  
**Nominal Variables** | Categories without order relationships | Color (red, blue, green), gender, country names  
**Ordinal Variables** | Categories with order relationships | Rating (low, medium, high), education level, size (S, M, L)  
  
### Why Encoding is Necessary

Many machine learning algorithms (linear regression, neural networks, SVM, etc.) only handle numerical data. Therefore, we need to convert categorical variables to numbers.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Many machine learning algorithms (linear regression, neural 
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    
    # Sample data with categorical variables
    data = {
        'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
        'size': ['S', 'M', 'L', 'M', 'S', 'L'],
        'rating': ['low', 'medium', 'high', 'medium', 'low', 'high'],
        'price': [100, 150, 200, 120, 90, 180]
    }
    
    df = pd.DataFrame(data)
    print("=== Sample Data with Categorical Variables ===")
    print(df)
    print("\nData types:")
    print(df.dtypes)
    
    # Identify categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical variables: {categorical_cols}")
    
    # Number of unique values in each categorical variable
    print("\nCardinality (number of categories) for each variable:")
    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()} categories -> {df[col].unique()}")
    

**Output** :
    
    
    === Sample Data with Categorical Variables ===
       color size rating  price
    0    red    S    low    100
    1   blue    M medium    150
    2  green    L   high    200
    3    red    M medium    120
    4   blue    S    low     90
    5  green    L   high    180
    
    Data types:
    color     object
    size      object
    rating    object
    price      int64
    dtype: object
    
    Categorical variables: ['color', 'size', 'rating']
    
    Cardinality (number of categories) for each variable:
      color: 3 categories -> ['red' 'blue' 'green']
      size: 3 categories -> ['S' 'M' 'L']
      rating: 3 categories -> ['low' 'medium' 'high']
    

### The Cardinality Problem

**Cardinality** is the number of unique values a categorical variable has.

  * **Low Cardinality** : About 2-10 categories → Most methods are applicable
  * **High Cardinality** : 100+ categories → Need to consider memory efficiency and overfitting

    
    
    ```mermaid
    graph TD
        A[Categorical Variable] --> B{Cardinality?}
        B -->|Low 2-10| C[One-Hot Encoding recommended]
        B -->|Medium 10-100| D[Compare multiple methods]
        B -->|High 100+| E[Target/Frequency/Hashing]
    
        C --> F[Apply each method]
        D --> F
        E --> F
    
        style A fill:#e3f2fd
        style C fill:#c8e6c9
        style D fill:#fff9c4
        style E fill:#ffccbc
    ```

* * *

## 2.2 One-Hot Encoding

### Overview

**One-Hot Encoding** is a technique that represents each category of a categorical variable as a binary vector of 0s and 1s.

### Principle

A variable with $n$ categories is converted into $n$ binary variables. For each sample, the column corresponding to the relevant category is 1, and all others are 0.

**Example** : color = {red, blue, green}

Original Data | color_red | color_blue | color_green  
---|---|---|---  
red | 1 | 0 | 0  
blue | 0 | 1 | 0  
green | 0 | 0 | 1  
  
### Implementation with pandas
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Implementation with pandas
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    
    # Sample data
    data = {
        'color': ['red', 'blue', 'green', 'red', 'blue'],
        'size': ['S', 'M', 'L', 'M', 'S'],
        'price': [100, 150, 200, 120, 90]
    }
    
    df = pd.DataFrame(data)
    print("=== Original Data ===")
    print(df)
    
    # One-Hot Encoding using pandas get_dummies
    df_encoded = pd.get_dummies(df, columns=['color', 'size'], drop_first=False)
    print("\n=== After One-Hot Encoding ===")
    print(df_encoded)
    
    # Avoid multicollinearity with drop_first=True
    df_encoded_drop = pd.get_dummies(df, columns=['color', 'size'], drop_first=True)
    print("\n=== drop_first=True (1 column dropped) ===")
    print(df_encoded_drop)
    

**Output** :
    
    
    === Original Data ===
       color size  price
    0    red    S    100
    1   blue    M    150
    2  green    L    200
    3    red    M    120
    4   blue    S     90
    
    === After One-Hot Encoding ===
       price  color_blue  color_green  color_red  size_L  size_M  size_S
    0    100           0            0          1       0       0       1
    1    150           1            0          0       0       1       0
    2    200           0            1          0       1       0       0
    3    120           0            0          1       0       1       0
    4     90           1            0          0       0       0       1
    
    === drop_first=True (1 column dropped) ===
       price  color_green  color_red  size_M  size_S
    0    100            0          1       0       1
    1    150            0          0       1       0
    2    200            1          0       0       0
    3    120            0          1       1       0
    4     90            0          0       0       1
    

### Implementation with scikit-learn
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation with scikit-learn
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    
    # Sample data
    X = np.array([['red', 'S'],
                  ['blue', 'M'],
                  ['green', 'L'],
                  ['red', 'M'],
                  ['blue', 'S']])
    
    print("=== Original Data ===")
    print(X)
    
    # Apply OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    X_encoded = encoder.fit_transform(X)
    
    print("\n=== After One-Hot Encoding ===")
    print(X_encoded)
    print(f"\nShape: {X_encoded.shape}")
    
    # Check categories
    print("\nCategories:")
    for i, categories in enumerate(encoder.categories_):
        print(f"  Feature {i}: {categories}")
    
    # Apply to new data
    X_new = np.array([['green', 'S'], ['red', 'L']])
    X_new_encoded = encoder.transform(X_new)
    print("\n=== Encoding New Data ===")
    print(X_new)
    print("↓")
    print(X_new_encoded)
    

### Utilizing Sparse Matrices

For high-cardinality categorical variables, One-Hot Encoding generates matrices with many zeros. Using **sparse matrices** can improve memory efficiency.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: For high-cardinality categorical variables, One-Hot Encoding
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.preprocessing import OneHotEncoder
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # High-cardinality sample
    np.random.seed(42)
    n_samples = 10000
    categories = [f'cat_{i}' for i in range(1000)]
    X = np.random.choice(categories, size=(n_samples, 1))
    
    print(f"Number of samples: {n_samples}")
    print(f"Number of categories: {len(categories)}")
    
    # Dense format
    encoder_dense = OneHotEncoder(sparse_output=False)
    X_dense = encoder_dense.fit_transform(X)
    dense_size = X_dense.nbytes / (1024 ** 2)  # MB
    
    # Sparse format
    encoder_sparse = OneHotEncoder(sparse_output=True)
    X_sparse = encoder_sparse.fit_transform(X)
    sparse_size = (X_sparse.data.nbytes + X_sparse.indices.nbytes +
                   X_sparse.indptr.nbytes) / (1024 ** 2)  # MB
    
    print("\n=== Memory Usage Comparison ===")
    print(f"Dense format: {dense_size:.2f} MB")
    print(f"Sparse format: {sparse_size:.2f} MB")
    print(f"Reduction rate: {(1 - sparse_size/dense_size) * 100:.1f}%")
    

### Advantages and Disadvantages of One-Hot Encoding

Advantages | Disadvantages  
---|---  
Does not assume ordering between categories | Dimensions increase proportionally to number of categories  
Simple implementation and easy to interpret | Inefficient for high cardinality  
Good compatibility with linear models | Sparsity issues  
Requires handling for new categories | Risk of multicollinearity  
  
* * *

## 2.3 Label Encoding and Ordinal Encoding

### Label Encoding

**Label Encoding** is a technique that converts each category to integers (0, 1, 2, ...).
    
    
    from sklearn.preprocessing import LabelEncoder
    
    # Sample data
    colors = ['red', 'blue', 'green', 'red', 'blue', 'green', 'red']
    
    # Apply LabelEncoder
    label_encoder = LabelEncoder()
    colors_encoded = label_encoder.fit_transform(colors)
    
    print("=== Label Encoding ===")
    print(f"Original data: {colors}")
    print(f"After encoding: {colors_encoded}")
    print(f"\nMapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")
    
    # Inverse transformation
    colors_decoded = label_encoder.inverse_transform(colors_encoded)
    print(f"\nInverse transformation: {colors_decoded}")
    

**Output** :
    
    
    === Label Encoding ===
    Original data: ['red', 'blue', 'green', 'red', 'blue', 'green', 'red']
    After encoding: [2 0 1 2 0 1 2]
    
    Mapping:
      blue -> 0
      green -> 1
      red -> 2
    
    Inverse transformation: ['red' 'blue' 'green' 'red' 'blue' 'green' 'red']
    

### Ordinal Encoding

**Ordinal Encoding** is a technique that assigns numerical values to categories with order relationships while preserving that order.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Ordinal Encodingis a technique that assigns numerical values
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.preprocessing import OrdinalEncoder
    import numpy as np
    
    # Sample with ordered categories
    data = {
        'size': ['S', 'M', 'L', 'XL', 'M', 'S', 'L'],
        'rating': ['low', 'medium', 'high', 'medium', 'low', 'high', 'medium']
    }
    
    df = pd.DataFrame(data)
    print("=== Original Data ===")
    print(df)
    
    # Define the order
    size_order = ['S', 'M', 'L', 'XL']
    rating_order = ['low', 'medium', 'high']
    
    # Apply OrdinalEncoder
    ordinal_encoder = OrdinalEncoder(categories=[size_order, rating_order])
    df_encoded = df.copy()
    df_encoded[['size', 'rating']] = ordinal_encoder.fit_transform(df[['size', 'rating']])
    
    print("\n=== After Ordinal Encoding ===")
    print(df_encoded)
    
    print("\nOrder mapping:")
    print("size: S(0) < M(1) < L(2) < XL(3)")
    print("rating: low(0) < medium(1) < high(2)")
    

**Output** :
    
    
    === Original Data ===
      size  rating
    0    S     low
    1    M  medium
    2    L    high
    3   XL  medium
    4    M     low
    5    S    high
    6    L  medium
    
    === After Ordinal Encoding ===
       size  rating
    0   0.0     0.0
    1   1.0     1.0
    2   2.0     2.0
    3   3.0     1.0
    4   1.0     0.0
    5   0.0     2.0
    6   2.0     1.0
    
    Order mapping:
    size: S(0) < M(1) < L(2) < XL(3)
    rating: low(0) < medium(1) < high(2)
    

### Differences Between Label Encoding and Ordinal Encoding

Feature | Label Encoding | Ordinal Encoding  
---|---|---  
**Purpose** | Encoding target variables | Encoding explanatory variables  
**Order consideration** | Not considered (alphabetical order, etc.) | Explicitly specified order  
**Implementation** | LabelEncoder (1D only) | OrdinalEncoder (supports multiple columns)  
**Target** | Classification problem labels | Ordered categorical features  
  
### Caution: Assuming Incorrect Order
    
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # Sample data with nominal variables (no order)
    np.random.seed(42)
    n_samples = 1000
    colors = np.random.choice(['red', 'blue', 'green'], size=n_samples)
    # y is more likely to be 1 when color is 'red'
    y = (colors == 'red').astype(int)
    
    # 1. Learn with Label Encoding (inappropriate)
    label_encoder = LabelEncoder()
    X_label = label_encoder.fit_transform(colors).reshape(-1, 1)
    
    clf_label = DecisionTreeClassifier(random_state=42)
    score_label = cross_val_score(clf_label, X_label, y, cv=5).mean()
    
    # 2. Learn with One-Hot Encoding (appropriate)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    X_onehot = onehot_encoder.fit_transform(colors.reshape(-1, 1))
    
    clf_onehot = DecisionTreeClassifier(random_state=42)
    score_onehot = cross_val_score(clf_onehot, X_onehot, y, cv=5).mean()
    
    print("=== Comparison of Encoding Methods ===")
    print(f"Label Encoding: {score_label:.4f}")
    print(f"One-Hot Encoding: {score_onehot:.4f}")
    print("\n⚠️ The difference is small for decision trees, but significant for linear models")
    

> **Important** : Applying Label Encoding to nominal variables causes the model to learn non-existent order relationships. One-Hot Encoding is recommended for linear models and neural networks.

* * *

## 2.4 Target Encoding (Mean Encoding)

### Overview

**Target Encoding** is a technique that replaces each category with the mean (or other statistics) of the target variable. Also known as **Mean Encoding**.

### Principle

Target Encoding value for category $c$:

$$ \text{TE}(c) = \frac{\sum_{i: x_i = c} y_i}{|i: x_i = c|} $$

In other words, it's the mean value of the target variable for samples belonging to that category.

### Overfitting Problem and Smoothing

Target Encoding directly uses the target variable, making it **prone to overfitting**. To prevent this, we apply **smoothing** :

$$ \text{TE}_{\text{smooth}}(c) = \frac{n_c \cdot \text{mean}_c + m \cdot \text{global_mean}}{n_c + m} $$

  * $n_c$: Number of samples in category $c$
  * $\text{mean}_c$: Mean of target variable in category $c$
  * $\text{global_mean}$: Overall mean of target variable
  * $m$: Smoothing parameter (typically 1-100)

### Scratch Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Scratch Implementation
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    
    # Sample data
    np.random.seed(42)
    data = {
        'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'] * 10,
        'target': np.random.randint(0, 2, 100)
    }
    
    # Intentionally set category A's target high
    data_list = list(zip(data['category'], data['target']))
    modified_data = []
    for cat, target in data_list:
        if cat == 'A':
            target = 1 if np.random.rand() < 0.8 else 0
        modified_data.append((cat, target))
    
    df = pd.DataFrame(modified_data, columns=['category', 'target'])
    
    print("=== Sample Data ===")
    print(df.head(10))
    print(f"\nMean target value by category:")
    print(df.groupby('category')['target'].mean())
    
    # Target Encoding (without smoothing)
    def target_encoding_simple(df, column, target_col):
        """Simple Target Encoding"""
        mean_encoding = df.groupby(column)[target_col].mean()
        return df[column].map(mean_encoding)
    
    # Target Encoding (with smoothing)
    def target_encoding_smoothed(df, column, target_col, m=10):
        """Target Encoding with smoothing"""
        global_mean = df[target_col].mean()
        category_stats = df.groupby(column)[target_col].agg(['mean', 'count'])
    
        smoothed = (category_stats['count'] * category_stats['mean'] +
                    m * global_mean) / (category_stats['count'] + m)
    
        return df[column].map(smoothed)
    
    # Apply
    df['te_simple'] = target_encoding_simple(df, 'category', 'target')
    df['te_smoothed'] = target_encoding_smoothed(df, 'category', 'target', m=10)
    
    print("\n=== Target Encoding Results ===")
    print(df.groupby('category')[['target', 'te_simple', 'te_smoothed']].mean())
    

### Cross-Validation Strategy

Computing Target Encoding on training data and applying it to the same training data causes **leakage (information leakage)**. To prevent this, we use an **Out-of-Fold** strategy.
    
    
    from sklearn.model_selection import KFold
    
    def target_encoding_cv(X, y, column, n_splits=5, m=10):
        """Target Encoding with Cross-Validation"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoded = np.zeros(len(X))
        global_mean = y.mean()
    
        for train_idx, val_idx in kfold.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    
            # Calculate statistics on training data
            category_stats = pd.DataFrame({
                'category': X_train[column],
                'target': y_train
            }).groupby('category')['target'].agg(['mean', 'count'])
    
            # Smoothing
            smoothed_means = (category_stats['count'] * category_stats['mean'] +
                              m * global_mean) / (category_stats['count'] + m)
    
            # Apply to validation data
            encoded[val_idx] = X.iloc[val_idx][column].map(smoothed_means)
    
            # Fill unmapped values with global_mean
            encoded[val_idx] = np.nan_to_num(encoded[val_idx], nan=global_mean)
    
        return encoded
    
    # Sample data
    np.random.seed(42)
    n_samples = 500
    X = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    })
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Set category A's target high
    y[X['category'] == 'A'] = np.random.choice([0, 1], size=(X['category'] == 'A').sum(), p=[0.2, 0.8])
    
    # Target Encoding with CV strategy
    X['te_cv'] = target_encoding_cv(X, y, 'category', n_splits=5, m=10)
    
    print("=== Target Encoding with Cross-Validation ===")
    print(X.groupby('category')['te_cv'].agg(['mean', 'std']))
    print(f"\nMean of target variable:")
    print(y.groupby(X['category']).mean())
    

### Using category_encoders Library
    
    
    import category_encoders as ce
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'category1': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'category2': np.random.choice(['X', 'Y', 'Z'], size=n_samples),
        'numeric': np.random.randn(n_samples)
    })
    
    # Target is more likely to be 1 when category1 is A and category2 is X
    y = ((X['category1'] == 'A') & (X['category2'] == 'X')).astype(int)
    y = np.where(np.random.rand(n_samples) < 0.3, 1 - y, y)  # Add noise
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. Learn with One-Hot Encoding
    X_train_onehot = pd.get_dummies(X_train, columns=['category1', 'category2'])
    X_test_onehot = pd.get_dummies(X_test, columns=['category1', 'category2'])
    
    # Align columns
    missing_cols = set(X_train_onehot.columns) - set(X_test_onehot.columns)
    for col in missing_cols:
        X_test_onehot[col] = 0
    X_test_onehot = X_test_onehot[X_train_onehot.columns]
    
    clf_onehot = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_onehot.fit(X_train_onehot, y_train)
    y_pred_onehot = clf_onehot.predict(X_test_onehot)
    acc_onehot = accuracy_score(y_test, y_pred_onehot)
    
    # 2. Learn with Target Encoding
    target_encoder = ce.TargetEncoder(cols=['category1', 'category2'], smoothing=10)
    X_train_te = target_encoder.fit_transform(X_train, y_train)
    X_test_te = target_encoder.transform(X_test)
    
    clf_te = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_te.fit(X_train_te, y_train)
    y_pred_te = clf_te.predict(X_test_te)
    acc_te = accuracy_score(y_test, y_pred_te)
    
    print("=== Performance Comparison of Encoding Methods ===")
    print(f"One-Hot Encoding: Accuracy = {acc_onehot:.4f}")
    print(f"Target Encoding:  Accuracy = {acc_te:.4f}")
    

### Advantages and Disadvantages of Target Encoding

Advantages | Disadvantages  
---|---  
Handles high cardinality | Prone to overfitting  
No dimension increase | CV strategy essential  
Directly captures relationship with target variable | Complex implementation  
Good compatibility with tree-based models | Limited effectiveness in regression problems in some cases  
  
* * *

## 2.5 Frequency Encoding

### Overview

**Frequency Encoding** is a technique that replaces each category with its frequency of occurrence (or proportion).

### Principle

Frequency Encoding value for category $c$:

$$ \text{FE}(c) = \frac{\text{count}(c)}{N} $$

where $N$ is the total number of samples.

### Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Data Processing
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    
    # Sample data
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D', 'E']
    # Category A appears most frequently
    probabilities = [0.5, 0.2, 0.15, 0.1, 0.05]
    
    data = {
        'category': np.random.choice(categories, size=1000, p=probabilities),
        'value': np.random.randn(1000)
    }
    
    df = pd.DataFrame(data)
    
    print("=== Category Occurrence Counts ===")
    print(df['category'].value_counts().sort_index())
    
    # Frequency Encoding (count-based)
    def frequency_encoding_count(df, column):
        """Count-based Frequency Encoding"""
        frequency = df[column].value_counts()
        return df[column].map(frequency)
    
    # Frequency Encoding (ratio-based)
    def frequency_encoding_ratio(df, column):
        """Ratio-based Frequency Encoding"""
        frequency = df[column].value_counts(normalize=True)
        return df[column].map(frequency)
    
    # Apply
    df['freq_count'] = frequency_encoding_count(df, 'category')
    df['freq_ratio'] = frequency_encoding_ratio(df, 'category')
    
    print("\n=== Frequency Encoding Results ===")
    print(df.groupby('category')[['freq_count', 'freq_ratio']].first().sort_index())
    print("\nSample data:")
    print(df.head(10))
    

**Output** :
    
    
    === Category Occurrence Counts ===
    A    492
    B    206
    C    163
    D     95
    E     44
    Name: category, dtype: int64
    
    === Frequency Encoding Results ===
              freq_count  freq_ratio
    category
    A                492       0.492
    B                206       0.206
    C                163       0.163
    D                 95       0.095
    E                 44       0.044
    
    Sample data:
      category     value  freq_count  freq_ratio
    0        C  0.496714         163       0.163
    1        A -0.138264         492       0.492
    2        A  0.647689         492       0.492
    3        A  1.523030         492       0.492
    4        B -0.234153         206       0.206
    

### Application Example of Frequency Encoding
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 2000
    
    # High-frequency categories are more likely to have target=1
    categories = np.random.choice(['A', 'B', 'C', 'D', 'E'],
                                  size=n_samples,
                                  p=[0.4, 0.25, 0.2, 0.1, 0.05])
    
    # Target is more likely to be 1 for 'A' and 'B'
    target = np.where(np.isin(categories, ['A', 'B']),
                      np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                      np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
    
    X = pd.DataFrame({'category': categories})
    y = target
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. Learn with Label Encoding
    label_encoder = LabelEncoder()
    X_train_label = label_encoder.fit_transform(X_train['category']).reshape(-1, 1)
    X_test_label = label_encoder.transform(X_test['category']).reshape(-1, 1)
    
    clf_label = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_label.fit(X_train_label, y_train)
    acc_label = accuracy_score(y_test, clf_label.predict(X_test_label))
    
    # 2. Learn with Frequency Encoding
    freq_map = X_train['category'].value_counts(normalize=True)
    X_train_freq = X_train['category'].map(freq_map).values.reshape(-1, 1)
    X_test_freq = X_test['category'].map(freq_map).fillna(0).values.reshape(-1, 1)
    
    clf_freq = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_freq.fit(X_train_freq, y_train)
    acc_freq = accuracy_score(y_test, clf_freq.predict(X_test_freq))
    
    # 3. Learn with One-Hot Encoding
    X_train_onehot = pd.get_dummies(X_train, columns=['category'])
    X_test_onehot = pd.get_dummies(X_test, columns=['category'])
    X_test_onehot = X_test_onehot.reindex(columns=X_train_onehot.columns, fill_value=0)
    
    clf_onehot = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_onehot.fit(X_train_onehot, y_train)
    acc_onehot = accuracy_score(y_test, clf_onehot.predict(X_test_onehot))
    
    print("=== Performance Comparison of Encoding Methods ===")
    print(f"Label Encoding:     Accuracy = {acc_label:.4f}")
    print(f"Frequency Encoding: Accuracy = {acc_freq:.4f}")
    print(f"One-Hot Encoding:   Accuracy = {acc_onehot:.4f}")
    

### When to Use Frequency Encoding

  * When category occurrence frequency correlates with the target variable
  * High-cardinality categorical variables
  * When dimension reduction is needed
  * When new categories (unknown categories) may appear

* * *

## 2.6 Binary Encoding and Hashing

### Binary Encoding

**Binary Encoding** is a technique that converts categories to integers and represents those integers in binary format. It can reduce dimensions compared to One-Hot Encoding.

### Principle

$n$ categories are represented by $\lceil \log_2 n \rceil$ binary columns.

**Example** : 8 categories → 3 columns ($\lceil \log_2 8 \rceil = 3$)

Category | Integer | bit_0 | bit_1 | bit_2  
---|---|---|---|---  
A | 0 | 0 | 0 | 0  
B | 1 | 0 | 0 | 1  
C | 2 | 0 | 1 | 0  
D | 3 | 0 | 1 | 1  
E | 4 | 1 | 0 | 0  
  
### Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Data Processing
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import category_encoders as ce
    import pandas as pd
    import numpy as np
    
    # Sample data
    np.random.seed(42)
    categories = [f'cat_{i}' for i in range(50)]
    data = {
        'category': np.random.choice(categories, size=200)
    }
    
    df = pd.DataFrame(data)
    
    print(f"=== Binary Encoding ===")
    print(f"Number of categories: {df['category'].nunique()}")
    print(f"Required number of bits: {int(np.ceil(np.log2(df['category'].nunique())))}")
    
    # Apply Binary Encoder
    binary_encoder = ce.BinaryEncoder(cols=['category'])
    df_encoded = binary_encoder.fit_transform(df)
    
    print(f"\nNumber of columns after encoding: {df_encoded.shape[1]}")
    print("\nSample:")
    print(df_encoded.head(10))
    
    # Dimension comparison
    print("\n=== One-Hot vs Binary Encoding ===")
    n_categories = 100
    onehot_dims = n_categories
    binary_dims = int(np.ceil(np.log2(n_categories)))
    
    print(f"Number of categories: {n_categories}")
    print(f"One-Hot Encoding: {onehot_dims} dimensions")
    print(f"Binary Encoding: {binary_dims} dimensions")
    print(f"Reduction rate: {(1 - binary_dims/onehot_dims) * 100:.1f}%")
    

### Hashing Trick

**Hashing Trick** is a technique that uses a hash function to convert categories to fixed-dimension vectors.

### Principle

  1. Map categories to integers using hash function $h$: $h(c) \in \\{0, 1, ..., m-1\\}$
  2. Set the position corresponding to that integer to 1

**Advantages** :

  * No need to know the number of categories in advance
  * New categories are automatically handled
  * Memory efficient

**Disadvantages** :

  * Hash collisions (different categories map to the same value)
  * Reduced interpretability

### Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Data Processing
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from sklearn.feature_extraction import FeatureHasher
    import pandas as pd
    import numpy as np
    
    # Sample data
    np.random.seed(42)
    categories = [f'cat_{i}' for i in range(1000)]
    data = {'category': np.random.choice(categories, size=5000)}
    df = pd.DataFrame(data)
    
    print("=== Hashing Trick ===")
    print(f"Number of unique categories: {df['category'].nunique()}")
    
    # Apply FeatureHasher
    n_features = 50  # Hash dimensions
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    
    # Convert categories to list of lists
    X_hashed = hasher.transform([[cat] for cat in df['category']])
    
    print(f"Dimensions after hashing: {X_hashed.shape[1]}")
    print(f"Sparsity: {(1 - X_hashed.nnz / (X_hashed.shape[0] * X_hashed.shape[1])) * 100:.1f}%")
    
    # Check hash collisions
    unique_hashes = set()
    collisions = 0
    
    for cat in df['category'].unique():
        hash_val = hash(cat) % n_features
        if hash_val in unique_hashes:
            collisions += 1
        unique_hashes.add(hash_val)
    
    print(f"\nNumber of hash collisions: {collisions}")
    print(f"Collision rate: {collisions / df['category'].nunique() * 100:.2f}%")
    
    # Relationship between dimensions and collision rate
    dimensions = [10, 20, 50, 100, 200, 500]
    collision_rates = []
    
    for dim in dimensions:
        unique_hashes = set()
        collisions = 0
        for cat in df['category'].unique():
            hash_val = hash(cat) % dim
            if hash_val in unique_hashes:
                collisions += 1
            unique_hashes.add(hash_val)
        collision_rate = collisions / df['category'].nunique() * 100
        collision_rates.append(collision_rate)
    
    print("\n=== Dimensions vs Collision Rate ===")
    for dim, rate in zip(dimensions, collision_rates):
        print(f"{dim} dimensions: collision rate {rate:.2f}%")
    

* * *

## 2.7 Comparison and Selection of Methods

### Comprehensive Comparison of Encoding Methods

Method | Dimension Increase | High Cardinality | Interpretability | Overfitting Risk  
---|---|---|---|---  
**One-Hot** | Large (n columns) | Not suitable | High | Low  
**Label/Ordinal** | None (1 column) | Applicable | Medium | Low  
**Target** | None (1 column) | Applicable | Medium | High (CV required)  
**Frequency** | None (1 column) | Applicable | High | Low  
**Binary** | Small (log n columns) | Applicable | Low | Low  
**Hashing** | Fixed (m columns) | Applicable | Low | Low  
  
### Selection Flowchart
    
    
    ```mermaid
    graph TD
        A[Categorical Variable] --> B{Cardinality?}
        B -->|Low 2-10| C{Ordered?}
        B -->|Medium 10-100| D[Try multiple methods]
        B -->|High 100+| E[Target/Frequency/Hashing]
    
        C -->|Yes| F[Ordinal Encoding]
        C -->|No| G[One-Hot Encoding]
    
        D --> H[One-Hot/Target/Frequency]
    
        E --> I{Correlation with target?}
        I -->|Strong| J[Target Encoding + CV]
        I -->|Weak| K[Frequency/Hashing]
    
        style A fill:#e3f2fd
        style G fill:#c8e6c9
        style F fill:#fff9c4
        style J fill:#ffccbc
    ```

### Practical Selection Guide

Situation | Recommended Method | Reason  
---|---|---  
Linear model + low cardinality | One-Hot | Linear models assume ordering  
Tree-based model + ordered | Ordinal | Order helps with tree splits  
High cardinality + classification | Target | Directly captures relationship with target  
Streaming data | Hashing | Automatically handles new categories  
Memory constraints | Binary/Hashing | Dimension reduction  
Interpretability focus | One-Hot/Frequency | Intuitive understanding possible  
  
### Example: Performance Comparison of All Methods
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Example: Performance Comparison of All Methods
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import category_encoders as ce
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 2000
    
    # Medium cardinality categories (20)
    categories = [f'cat_{i}' for i in range(20)]
    X_cat = np.random.choice(categories, size=n_samples)
    
    # Some categories more likely to have target=1
    high_target_cats = ['cat_0', 'cat_1', 'cat_5', 'cat_10']
    y = np.where(np.isin(X_cat, high_target_cats),
                 np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                 np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
    
    X = pd.DataFrame({'category': X_cat})
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = []
    
    # 1. One-Hot Encoding
    X_train_onehot = pd.get_dummies(X_train, columns=['category'])
    X_test_onehot = pd.get_dummies(X_test, columns=['category'])
    X_test_onehot = X_test_onehot.reindex(columns=X_train_onehot.columns, fill_value=0)
    
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    score_onehot = cross_val_score(clf_rf, X_train_onehot, y_train, cv=5).mean()
    results.append(('One-Hot', score_onehot, X_train_onehot.shape[1]))
    
    # 2. Label Encoding
    label_encoder = LabelEncoder()
    X_train_label = label_encoder.fit_transform(X_train['category']).reshape(-1, 1)
    X_test_label = label_encoder.transform(X_test['category']).reshape(-1, 1)
    
    score_label = cross_val_score(clf_rf, X_train_label, y_train, cv=5).mean()
    results.append(('Label', score_label, 1))
    
    # 3. Target Encoding
    target_encoder = ce.TargetEncoder(cols=['category'], smoothing=10)
    X_train_target = target_encoder.fit_transform(X_train, y_train)
    X_test_target = target_encoder.transform(X_test)
    
    score_target = cross_val_score(clf_rf, X_train_target, y_train, cv=5).mean()
    results.append(('Target', score_target, 1))
    
    # 4. Frequency Encoding
    freq_map = X_train['category'].value_counts(normalize=True)
    X_train_freq = X_train['category'].map(freq_map).values.reshape(-1, 1)
    X_test_freq = X_test['category'].map(freq_map).fillna(0).values.reshape(-1, 1)
    
    score_freq = cross_val_score(clf_rf, X_train_freq, y_train, cv=5).mean()
    results.append(('Frequency', score_freq, 1))
    
    # 5. Binary Encoding
    binary_encoder = ce.BinaryEncoder(cols=['category'])
    X_train_binary = binary_encoder.fit_transform(X_train)
    X_test_binary = binary_encoder.transform(X_test)
    
    score_binary = cross_val_score(clf_rf, X_train_binary, y_train, cv=5).mean()
    results.append(('Binary', score_binary, X_train_binary.shape[1]))
    
    # Display results
    print("=== Performance Comparison of Encoding Methods (Random Forest) ===")
    print(f"{'Method':<15} {'Accuracy':<10} {'Dimensions':<10}")
    print("-" * 35)
    for method, score, dims in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{method:<15} {score:.4f}    {dims:<10}")
    

**Sample Output** :
    
    
    === Performance Comparison of Encoding Methods (Random Forest) ===
    Method          Accuracy    Dimensions
    -----------------------------------
    Target          0.7531    1
    One-Hot         0.7469    20
    Binary          0.7419    5
    Frequency       0.6956    1
    Label           0.6894    1
    

* * *

## 2.8 Chapter Summary

### What We Learned

  1. **Basics of Categorical Variables**

     * Difference between nominal and ordinal variables
     * Concept and importance of cardinality
     * Why encoding is necessary
  2. **One-Hot Encoding**

     * Representation using binary vectors
     * When to use pandas get_dummies vs OneHotEncoder
     * Memory efficiency using sparse matrices
     * Avoiding multicollinearity with drop_first
  3. **Label Encoding and Ordinal Encoding**

     * Integer conversion techniques
     * Selection based on presence of order
     * Cautions with linear models
  4. **Target Encoding**

     * Transformation using target variable statistics
     * Smoothing as overfitting countermeasure
     * Importance of Cross-Validation strategy
     * Handling high cardinality
  5. **Frequency Encoding**

     * Transformation using occurrence frequency
     * Simple and effective method
     * Handling new categories
  6. **Binary Encoding and Hashing**

     * Techniques for dimension reduction
     * Handling high cardinality
     * Hash collision tradeoffs
  7. **Method Selection**

     * Selection based on cardinality
     * Compatibility with models
     * Balance between computational resources and accuracy

### Next Chapter

In Chapter 3, we will learn about **numerical feature transformation and scaling** :

  * Standardization and normalization
  * Log transformation and Box-Cox transformation
  * Binning (discretization)
  * Feature interactions

* * *

## Exercises

### Exercise 1 (Difficulty: easy)

Explain why `drop_first=True` is used in One-Hot Encoding from the perspective of multicollinearity.

Sample Answer

**Answer** :

**Multicollinearity** refers to a state where there is a strong correlation between explanatory variables.

In One-Hot Encoding, $n$ categories are converted to $n$ binary variables. At this time, the following relationship holds:

$$ \sum_{i=1}^{n} x_i = 1 $$

In other words, the value of one variable can be completely predicted from the other $n-1$ variables. This causes multicollinearity.

**Problems** :

  * Linear regression coefficients become unstable
  * Errors may occur in inverse matrix calculation
  * Statistical inference becomes difficult

**Solution** :

With `drop_first=True`, $n$ categories are represented by $n-1$ variables. The omitted category is represented by "all variables are 0".

**Example** :
    
    
    color = {red, blue, green}
    drop_first=False: color_red, color_blue, color_green (3 columns)
    drop_first=True:  color_blue, color_green (2 columns)
      - red: [0, 0]
      - blue: [1, 0]
      - green: [0, 1]
    

### Exercise 2 (Difficulty: medium)

Describe three strategies to prevent overfitting in Target Encoding.

Sample Answer

**Answer** :

**1\. Smoothing**

Regularize the statistics of categories with few samples with the global mean:

$$ \text{TE}_{\text{smooth}}(c) = \frac{n_c \cdot \text{mean}_c + m \cdot \text{global_mean}}{n_c + m} $$

  * Larger $m$ approaches global mean (conservative)
  * Smaller $m$ approaches category mean (overfitting risk)
  * Recommended value: $m = 1 \sim 100$

**2\. Cross-Validation Strategy (Out-of-Fold Encoding)**

  1. Divide data into K folds
  2. Calculate statistics for fold $k$ on other folds
  3. Separate training and validation data

This prevents **leakage** where statistics are calculated and used on the same data.

**3\. Noise Addition**

Add small noise to encoded values:

$$ \text{TE}_{\text{noise}}(c) = \text{TE}(c) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) $$

  * Suppresses overfitting
  * $\sigma$ is a small value (around 0.01-0.1)

**Implementation Example** :
    
    
    import category_encoders as ce
    
    # Target Encoding with smoothing
    target_encoder = ce.TargetEncoder(cols=['category'], smoothing=10)
    X_encoded = target_encoder.fit_transform(X_train, y_train)
    

### Exercise 3 (Difficulty: medium)

For the following categorical variables, select the optimal encoding method and explain your reasoning.

  1. Prefecture names (47 categories)
  2. Website visitor IDs (1 million categories)
  3. Customer satisfaction (1=low, 2=medium, 3=high)
  4. Product category (5 categories)

Sample Answer

**Answer** :

**1\. Prefecture Names (47 categories)**

**Recommended** : Target Encoding or One-Hot Encoding

**Reasoning** :

  * Cardinality: Medium (47)
  * One-Hot: Increases to 47 columns but acceptable
  * Target: High expressiveness with 1 column. Can capture relationship between region and target
  * No order relationship (nominal variable)

**Selection Criteria** :

  * Linear model → One-Hot
  * Tree-based model + classification problem → Target

**2\. Website Visitor IDs (1 million categories)**

**Recommended** : Frequency Encoding or Hashing

**Reasoning** :

  * Cardinality: Very high (1 million)
  * One-Hot: Practically impossible due to memory shortage
  * Frequency: Visit frequency may be a useful feature
  * Hashing: Fixed dimensions with automatic handling of new IDs

**3\. Customer Satisfaction (1=low, 2=medium, 3=high)**

**Recommended** : Ordinal Encoding

**Reasoning** :

  * Clear order relationship (ordinal variable)
  * Should preserve order: low(0) < medium(1) < high(2)
  * One-Hot loses order information
  * Can be treated as integer values directly

    
    
    from sklearn.preprocessing import OrdinalEncoder
    
    encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
    X_encoded = encoder.fit_transform(X)
    

**4\. Product Category (5 categories)**

**Recommended** : One-Hot Encoding

**Reasoning** :

  * Cardinality: Low (5)
  * No order relationship (nominal variable)
  * One-Hot increases to 5 columns with no problem
  * High interpretability
  * Good compatibility with linear models

### Exercise 4 (Difficulty: hard)

For high-cardinality categorical variables (1000 categories), write code to apply One-Hot Encoding, Target Encoding, Frequency Encoding, and Binary Encoding, and compare their performance using Random Forest.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: For high-cardinality categorical variables (1000 categories)
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import category_encoders as ce
    import time
    
    # Generate high-cardinality sample data
    np.random.seed(42)
    n_samples = 10000
    n_categories = 1000
    
    # Generate categories (realistic frequency distribution with power law)
    categories = [f'cat_{i}' for i in range(n_categories)]
    weights = np.array([1/(i+1)**0.8 for i in range(n_categories)])
    weights /= weights.sum()
    
    X_cat = np.random.choice(categories, size=n_samples, p=weights)
    
    # Target variable: top 50 categories more likely to have target=1
    high_target_cats = [f'cat_{i}' for i in range(50)]
    y = np.where(np.isin(X_cat, high_target_cats),
                 np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                 np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
    
    X = pd.DataFrame({'category': X_cat})
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"=== Data Overview ===")
    print(f"Number of samples: {n_samples}")
    print(f"Number of categories: {X['category'].nunique()}")
    print(f"Training data: {len(X_train)}, Test data: {len(X_test)}")
    
    results = []
    
    # Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 1. One-Hot Encoding (sparse matrix)
    print("\n1. One-Hot Encoding...")
    start_time = time.time()
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_train_onehot = onehot_encoder.fit_transform(X_train[['category']])
    X_test_onehot = onehot_encoder.transform(X_test[['category']])
    
    score_onehot = cross_val_score(clf, X_train_onehot, y_train, cv=3, n_jobs=-1).mean()
    time_onehot = time.time() - start_time
    results.append(('One-Hot', score_onehot, X_train_onehot.shape[1], time_onehot))
    
    # 2. Target Encoding
    print("2. Target Encoding...")
    start_time = time.time()
    target_encoder = ce.TargetEncoder(cols=['category'], smoothing=10)
    X_train_target = target_encoder.fit_transform(X_train, y_train)
    X_test_target = target_encoder.transform(X_test)
    
    score_target = cross_val_score(clf, X_train_target, y_train, cv=3, n_jobs=-1).mean()
    time_target = time.time() - start_time
    results.append(('Target', score_target, 1, time_target))
    
    # 3. Frequency Encoding
    print("3. Frequency Encoding...")
    start_time = time.time()
    freq_map = X_train['category'].value_counts(normalize=True)
    X_train_freq = X_train['category'].map(freq_map).values.reshape(-1, 1)
    X_test_freq = X_test['category'].map(freq_map).fillna(0).values.reshape(-1, 1)
    
    score_freq = cross_val_score(clf, X_train_freq, y_train, cv=3, n_jobs=-1).mean()
    time_freq = time.time() - start_time
    results.append(('Frequency', score_freq, 1, time_freq))
    
    # 4. Binary Encoding
    print("4. Binary Encoding...")
    start_time = time.time()
    binary_encoder = ce.BinaryEncoder(cols=['category'])
    X_train_binary = binary_encoder.fit_transform(X_train)
    X_test_binary = binary_encoder.transform(X_test)
    
    score_binary = cross_val_score(clf, X_train_binary, y_train, cv=3, n_jobs=-1).mean()
    time_binary = time.time() - start_time
    results.append(('Binary', score_binary, X_train_binary.shape[1], time_binary))
    
    # Display results
    print("\n" + "="*70)
    print("=== Performance Comparison of Encoding Methods (1000 categories) ===")
    print("="*70)
    print(f"{'Method':<15} {'Accuracy':<10} {'Dimensions':<10} {'Execution Time (sec)':<15}")
    print("-"*70)
    
    for method, score, dims, exec_time in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{method:<15} {score:.4f}    {dims:<10} {exec_time:.2f}")
    
    print("\n" + "="*70)
    print("Observations:")
    print("- Target Encoding: High accuracy + 1 dimension + fast")
    print("- One-Hot: High accuracy but large memory usage")
    print("- Binary: Dimension reduction with balanced performance")
    print("- Frequency: Simple but insufficient information")
    print("="*70)
    

**Sample Output** :
    
    
    === Data Overview ===
    Number of samples: 10000
    Number of categories: 1000
    Training data: 8000, Test data: 2000
    
    1. One-Hot Encoding...
    2. Target Encoding...
    3. Frequency Encoding...
    4. Binary Encoding...
    
    ======================================================================
    === Performance Comparison of Encoding Methods (1000 categories) ===
    ======================================================================
    Method          Accuracy    Dimensions  Execution Time (sec)
    ----------------------------------------------------------------------
    Target          0.8125    1          2.45
    One-Hot         0.8031    1000       5.67
    Binary          0.7794    10         3.12
    Frequency       0.7031    1          1.89
    
    ======================================================================
    Observations:
    - Target Encoding: High accuracy + 1 dimension + fast
    - One-Hot: High accuracy but large memory usage
    - Binary: Dimension reduction with balanced performance
    - Frequency: Simple but insufficient information
    ======================================================================
    

### Exercise 5 (Difficulty: hard)

Explain how each encoding method should handle cases where new categories (unknown categories) may appear.

Sample Answer

**Answer** :

Handling new categories is very important in practical operation. Here's how each method handles them.

**1\. One-Hot Encoding**

**Countermeasures** :

  * `handle_unknown='ignore'`: Set all unknown categories to 0
  * Add an "Other" category

    
    
    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train)
    X_test_encoded = encoder.transform(X_test)  # Unknown categories become [0,0,0,...]
    

**2\. Label Encoding / Ordinal Encoding**

**Countermeasures** :

  * Assign a special value (-1, etc.) to unknown categories
  * Replace with the most frequent category

    
    
    from sklearn.preprocessing import LabelEncoder
    
    encoder = LabelEncoder()
    encoder.fit(X_train)
    
    # Handle unknown categories with -1
    X_test_encoded = []
    for x in X_test:
        if x in encoder.classes_:
            X_test_encoded.append(encoder.transform([x])[0])
        else:
            X_test_encoded.append(-1)  # Unknown category
    

**3\. Target Encoding**

**Countermeasures** :

  * Replace with global mean
  * Use the same value as the global mean in smoothing

    
    
    import category_encoders as ce
    
    target_encoder = ce.TargetEncoder(cols=['category'],
                                      smoothing=10,
                                      handle_unknown='value',
                                      handle_missing='value')
    
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)  # Unknown → global mean
    

**4\. Frequency Encoding**

**Countermeasures** :

  * Assign frequency 0 (or minimum frequency)
  * Treat as rare category

    
    
    freq_map = X_train['category'].value_counts(normalize=True)
    min_freq = freq_map.min()
    
    # Unknown categories get minimum frequency
    X_test_encoded = X_test['category'].map(freq_map).fillna(min_freq)
    

**5\. Binary Encoding**

**Countermeasures** :

  * Assign a special code (all 0s, etc.) to unknown categories

    
    
    import category_encoders as ce
    
    binary_encoder = ce.BinaryEncoder(cols=['category'], handle_unknown='value')
    X_train_encoded = binary_encoder.fit_transform(X_train)
    X_test_encoded = binary_encoder.transform(X_test)
    

**6\. Hashing**

**Countermeasures** :

  * Automatically handled (hash function converts to fixed dimensions)
  * New categories are also mapped to existing hash values

    
    
    from sklearn.feature_extraction import FeatureHasher
    
    hasher = FeatureHasher(n_features=50, input_type='string')
    X_train_hashed = hasher.transform([[cat] for cat in X_train['category']])
    X_test_hashed = hasher.transform([[cat] for cat in X_test['category']])
    # Unknown categories are automatically hashed
    

**Recommended Strategy** :

Situation | Recommended Method  
---|---  
Frequent unknown categories | Hashing  
Rare unknown categories | One-Hot (ignore) or Target (global mean)  
Interpretability focus | Frequency (minimum frequency)  
High accuracy priority | Target (global mean)  
  
* * *

## References

  1. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. _ACM SIGKDD Explorations Newsletter_ , 3(1), 27-32.
  2. Weinberger, K., et al. (2009). Feature hashing for large scale multitask learning. _Proceedings of the 26th Annual International Conference on Machine Learning_.
  3. Pargent, F., et al. (2022). Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features. _Computational Statistics_ , 37(5), 2671-2692.
  4. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  5. Kuhn, M., & Johnson, K. (2019). _Feature Engineering and Selection: A Practical Approach for Predictive Models_. CRC Press.
