---
title: "Chapter 1: Model Interpretability Basics"
chapter_title: "Chapter 1: Model Interpretability Basics"
subtitle: Understanding Interpretability for Building Trustworthy AI Systems
reading_time: 30-35 minutes
difficulty: Beginner
code_examples: 8
exercises: 6
---

This chapter introduces the basics of Model Interpretability Basics. You will learn why model interpretability is important, taxonomy of interpretability, and characteristics of interpretable models.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand why model interpretability is important
  * ✅ Grasp the taxonomy of interpretability
  * ✅ Learn the characteristics of interpretable models
  * ✅ Understand an overview of major interpretation techniques
  * ✅ Learn criteria for evaluating interpretability
  * ✅ Implement practical interpretable models

* * *

## 1.1 Why Model Interpretability Matters

### Trust and Accountability

To trust machine learning model predictions, we need to understand "why the model made that prediction." Especially for high-risk decision-making (medical diagnosis, loan approval, criminal justice, etc.), accountability is essential.

Application Domain | Why Interpretability is Needed | Risks  
---|---|---  
**Medical Diagnosis** | Doctors need to understand diagnostic reasoning and explain to patients | Life-threatening misdiagnosis  
**Loan Approval** | Obligation to explain rejection reasons, ensure fairness | Discriminatory decisions, legal litigation  
**Criminal Justice** | Need to show basis for recidivism risk assessment | Unjust verdicts, human rights violations  
**Autonomous Vehicles** | Accountability in accidents, safety verification | Loss of life, legal liability  
  
> **Important** : "High prediction accuracy" alone is insufficient. For stakeholders to trust and properly use models, they need to understand the basis for predictions.

### Regulatory Requirements (GDPR, AI Regulations)

Regulations regarding machine learning model transparency are being strengthened worldwide:

  * **GDPR (General Data Protection Regulation)** : Stipulates "right to explanation" regarding automated decision-making (Article 22)
  * **EU AI Act** : Transparency and explainability requirements for high-risk AI systems
  * **U.S. Fair Credit Reporting Act** : Obligation to provide "adverse action notice" regarding credit scores
  * **Japan's Personal Information Protection Act** : Information provision to individuals regarding automated decision-making

### Debugging and Model Improvement

Interpretability is also essential for improving model performance:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Diagnosing unexpected model predictions
    
    Problem: Customer churn prediction model performs poorly in production
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 120, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 10000, n_samples),
        'num_support_calls': np.random.poisson(2, n_samples),
        'contract_type': np.random.choice(['month', 'year', '2year'], n_samples),
        'customer_id': np.arange(n_samples)  # Data leak!
    })
    
    # Target variable (churn)
    data['churn'] = ((data['num_support_calls'] > 3) |
                     (data['monthly_charges'] > 100)).astype(int)
    
    # Train model
    X = data.drop('churn', axis=1)
    X_encoded = pd.get_dummies(X, columns=['contract_type'])
    y = data['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Diagnose with Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance.head(10))
    
    # Problem discovered: customer_id has highest importance (data leak)
    print("\n⚠️ Abnormally high importance for customer_id → Possible data leakage")
    

### Bias Detection

Interpretability allows us to discover unfair patterns learned by the model:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Bias detection in hiring screening model
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Sample data with bias
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'years_experience': np.random.randint(0, 20, n_samples),
        'education_level': np.random.randint(1, 5, n_samples),
        'skills_score': np.random.uniform(0, 100, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.randint(22, 65, n_samples)
    })
    
    # Biased target (includes gender discrimination)
    data['hired'] = (
        (data['years_experience'] > 5) &
        (data['skills_score'] > 60) &
        (data['gender'] == 'M')  # Gender bias
    ).astype(int)
    
    # Train model
    X = pd.get_dummies(data.drop('hired', axis=1), columns=['gender'])
    y = data['hired']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Check coefficients to detect bias
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    print("Model Coefficients:")
    print(coefficients)
    
    # Abnormally high coefficient for gender_M → Detect gender bias
    print("\n⚠️ High coefficient for gender_M → Possible gender discrimination")
    print("📊 Fairness evaluation required")
    

* * *

## 1.2 Classification of Interpretability

### Global Interpretation vs Local Interpretation

Classification | Description | Question | Example Methods  
---|---|---|---  
**Global Interpretation** | Understanding overall model behavior | "How does the model predict in general?" | Feature Importance, Partial Dependence  
**Local Interpretation** | Explaining individual predictions | "Why was this customer predicted to churn?" | LIME, SHAP, Counterfactual  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Global Interpretation vs Local Interpretation
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.uniform(20000, 150000, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'credit_history_months': np.random.randint(0, 360, n_samples)
    })
    
    # Target: Loan approval
    data['approved'] = (
        (data['income'] > 50000) &
        (data['debt_ratio'] < 0.5) &
        (data['credit_history_months'] > 24)
    ).astype(int)
    
    X = data.drop('approved', axis=1)
    y = data['approved']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # --- Global Interpretation: Feature Importance ---
    print("=== Global Interpretation ===")
    print("Most important features for the overall model:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # --- Local Interpretation: Individual prediction explanation ---
    print("\n=== Local Interpretation ===")
    # Select one sample from test data
    sample_idx = 0
    sample = X_test.iloc[sample_idx:sample_idx+1]
    prediction = model.predict(sample)[0]
    prediction_proba = model.predict_proba(sample)[0]
    
    print(f"Features of sample {sample_idx}:")
    print(sample.T)
    print(f"\nPrediction: {'Approved' if prediction == 1 else 'Rejected'}")
    print(f"Probability: {prediction_proba[1]:.2%}")
    
    # Simple local importance (tree-based)
    # In practice, using SHAP or LIME is recommended
    print("\nFeatures contributing to this prediction (approximate):")
    for feature in X.columns:
        print(f"  {feature}: {sample[feature].values[0]:.2f}")
    

### Model-Specific vs Model-Agnostic

Classification | Description | Advantages | Disadvantages  
---|---|---|---  
**Model-Specific** | Interpretation specific to particular models | Accurate, efficient | Cannot be applied to other models  
**Model-Agnostic** | Applicable to any model | High versatility | May have high computational cost  
  
### Intrinsic Interpretability vs Post-hoc Interpretability

  * **Intrinsic Interpretability** : The model itself is interpretable (linear regression, decision trees)
  * **Post-hoc Interpretability** : Interpreting black-box models after the fact (SHAP, LIME)

### Taxonomy of Interpretability
    
    
    ```mermaid
    graph TB
        A[Model Interpretability] --> B[Scope]
        A --> C[Dependency]
        A --> D[Timing]
    
        B --> B1[Global InterpretationOverall model behavior]
        B --> B2[Local InterpretationIndividual prediction explanation]
    
        C --> C1[Model-SpecificFor specific models]
        C --> C2[Model-AgnosticGeneral purpose]
    
        D --> D1[Intrinsic InterpretabilityInherently interpretable]
        D --> D2[Post-hoc InterpretabilityAfter-the-fact explanation]
    
        style A fill:#7b2cbf,color:#fff
        style B1 fill:#e3f2fd
        style B2 fill:#e3f2fd
        style C1 fill:#fff3e0
        style C2 fill:#fff3e0
        style D1 fill:#c8e6c9
        style D2 fill:#c8e6c9
    ```

* * *

## 1.3 Interpretable Models

### Linear Regression

Linear regression is one of the most easily interpretable models. The coefficient of each feature directly indicates its influence.

**Formula** :

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$$

$\beta_i$ indicates the change in predicted value for a one-unit change in feature $x_i$.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Housing price prediction with linear regression
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        'square_feet': np.random.randint(500, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'distance_to_city': np.random.uniform(0, 50, n_samples)
    })
    
    # Target: Price (in ten thousands)
    data['price'] = (
        data['square_feet'] * 0.5 +
        data['bedrooms'] * 50 -
        data['age_years'] * 5 -
        data['distance_to_city'] * 10 +
        np.random.normal(0, 100, n_samples)
    )
    
    X = data.drop('price', axis=1)
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardization (for comparing coefficients)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Interpret coefficients
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Linear regression model coefficients:")
    print(coefficients)
    print(f"\nIntercept: {model.intercept_:.2f}")
    
    print("\nInterpretation:")
    print("- square_feet has the largest coefficient → Area most influences price")
    print("- age_years has a negative coefficient → Older properties have lower prices")
    print("- Coefficients are standardized, enabling direct comparison")
    
    # Prediction example
    sample = X_test_scaled[0:1]
    prediction = model.predict(sample)[0]
    print(f"\nSample predicted price: {prediction:.2f} (in ten thousands)")
    

### Decision Trees

Decision trees have rule-based branching structures that are easy for humans to understand.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Iris classification with decision tree
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Decision tree model (limit depth for interpretability)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Extract rules (text format)
    from sklearn.tree import export_text
    tree_rules = export_text(model, feature_names=list(iris.feature_names))
    print("\nDecision tree rules:")
    print(tree_rules[:500] + "...")  # Display first 500 characters only
    
    # Interpretation example
    print("\nInterpretation:")
    print("- petal width (cm) <= 0.8 → classified as setosa")
    print("- Otherwise, petal width or petal length determines versicolor/virginica")
    print("- Decision boundaries are clear and understandable even for non-experts")
    

### Rule-Based Models

Models composed of IF-THEN rules can be directly used as business rules.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Simple rule-based classifier
    """
    
    import numpy as np
    import pandas as pd
    
    class SimpleRuleClassifier:
        """Interpretable rule-based classifier"""
    
        def __init__(self):
            self.rules = []
    
        def add_rule(self, condition, prediction, description=""):
            """Add a rule"""
            self.rules.append({
                'condition': condition,
                'prediction': prediction,
                'description': description
            })
    
        def predict(self, X):
            """Make predictions"""
            predictions = []
            for _, row in X.iterrows():
                prediction = None
                for rule in self.rules:
                    if rule['condition'](row):
                        prediction = rule['prediction']
                        break
                predictions.append(prediction if prediction is not None else 0)
            return np.array(predictions)
    
        def explain(self):
            """Explain rules"""
            print("Classification rules:")
            for i, rule in enumerate(self.rules, 1):
                print(f"  Rule {i}: {rule['description']} → {rule['prediction']}")
    
    # Usage example: Loan approval rules
    classifier = SimpleRuleClassifier()
    
    # Rule 1: High income and low debt
    classifier.add_rule(
        condition=lambda row: row['income'] > 100000 and row['debt_ratio'] < 0.3,
        prediction=1,
        description="High income (>100K) and low debt ratio (<30%)"
    )
    
    # Rule 2: Medium income with good credit history
    classifier.add_rule(
        condition=lambda row: row['income'] > 50000 and row['credit_history_months'] > 36,
        prediction=1,
        description="Medium income (>50K) and credit history >3 years"
    )
    
    # Rule 3: Reject all other cases
    classifier.add_rule(
        condition=lambda row: True,
        prediction=0,
        description="All other cases"
    )
    
    # Test data
    test_data = pd.DataFrame({
        'income': [120000, 60000, 30000],
        'debt_ratio': [0.2, 0.4, 0.6],
        'credit_history_months': [48, 40, 12]
    })
    
    predictions = classifier.predict(test_data)
    classifier.explain()
    
    print("\nPrediction results:")
    for i, (pred, income) in enumerate(zip(predictions, test_data['income'])):
        print(f"  Applicant {i+1} (Income: ${income:,.0f}): {'Approved' if pred == 1 else 'Rejected'}")
    

### GAM (Generalized Additive Models)

GAMs are interpretable models that can visualize the nonlinear effect of each feature.

**Formula** :

$$g(\mathbb{E}[y]) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_n(x_n)$$

$f_i$ is a nonlinear function of feature $x_i$.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Modeling nonlinear relationships with GAM
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    # Generate sample data (nonlinear relationships)
    np.random.seed(42)
    n_samples = 300
    
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    
    # Nonlinear relationships: sine function and quadratic function
    y = np.sin(x1) + x2**2 + np.random.normal(0, 0.2, n_samples)
    
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    
    # Feature engineering: Add polynomial features (GAM approximation)
    from sklearn.preprocessing import PolynomialFeatures
    
    X = data[['x1', 'x2']]
    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_features=False)
    X_poly = poly.fit_transform(X)
    
    feature_names = poly.get_feature_names_out(['x1', 'x2'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, data['y'], test_size=0.2, random_state=42
    )
    
    # Train with Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    print(f"Test R² score: {model.score(X_test, y_test):.3f}")
    
    # Visualize the effect of each feature
    print("\nPolynomial coefficients for each feature:")
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    })
    print(coef_df)
    
    print("\nInterpretation:")
    print("- Odd-degree terms of x1 are important → sine function-like nonlinearity")
    print("- Quadratic term of x2 is important → quadratic function relationship")
    print("- Effects of each variable can be interpreted individually")
    

* * *

## 1.4 Overview of Interpretation Techniques

### Feature Importance

A method to quantify the importance of features. Frequently used in tree-based models.

  * **Mean Decrease Impurity** : Measures importance by decrease in impurity
  * **Permutation Importance** : Measures by performance degradation when features are shuffled

### Partial Dependence Plot (PDP)

Visualizes the relationship between a specific feature and model predictions.

**Formula** :

$$\text{PDP}(x_s) = \mathbb{E}_{x_c}[f(x_s, x_c)]$$

$x_s$ is the target feature, $x_c$ are the other features.

### SHAP (SHapley Additive exPlanations)

Uses Shapley values from game theory to calculate the contribution of each feature.

**Characteristics** :

  * Consistent explanations
  * Enables both local and global interpretation
  * Model-agnostic

### LIME (Local Interpretable Model-agnostic Explanations)

Explains individual predictions by approximating them locally with a linear model.

**Procedure** :

  1. Generate samples in the neighborhood of the instance to be predicted
  2. Obtain predictions from the black-box model
  3. Locally approximate with an interpretable model (such as linear regression)
  4. Interpret the coefficients of the approximate model

### Saliency Maps

In image classification, visualizes which pixels are important for predictions.

**Calculation Method** :

$$S(x) = \left| \frac{\partial f(x)}{\partial x} \right|$$

Computes gradients with respect to the input image and highlights important regions.

* * *

## 1.5 Evaluating Interpretability

### Fidelity

Measures how accurately the interpretation method explains the behavior of the original model.

Evaluation Metric | Description | Calculation Method  
---|---|---  
**R² Score** | Agreement between explanation model and original model | $R^2 = 1 - \frac{\sum(y_{\text{true}} - y_{\text{approx}})^2}{\sum(y_{\text{true}} - \bar{y})^2}$  
**Local Fidelity** | Agreement of local predictions | Prediction error on neighborhood samples  
  
### Consistency

Evaluates whether similar explanations are obtained for similar instances.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Evaluating consistency of interpretation
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Sample data
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    data['target'] = (data['feature1'] + data['feature2'] > 0).astype(int)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Compare Feature Importance for similar samples
    sample1 = X_test.iloc[0:1]
    sample2 = X_test.iloc[1:2]  # Similar sample
    
    # Simple local importance using tree paths
    # (Using SHAP is recommended in practice)
    
    print("Sample 1 features:")
    print(sample1.values)
    print(f"Prediction: {model.predict(sample1)[0]}")
    
    print("\nSample 2 features:")
    print(sample2.values)
    print(f"Prediction: {model.predict(sample2)[0]}")
    
    # Calculate distance
    distance = np.linalg.norm(sample1.values - sample2.values)
    print(f"\nDistance between samples: {distance:.3f}")
    print("Consistency evaluation: Need to verify if explanations for similar samples are similar")
    

### Stability

Evaluates whether interpretations change significantly with minor changes in input data.

### Comprehensibility

Evaluates how easily humans can understand the explanation. Since quantification is difficult, user studies are common.

Evaluation Method | Description  
---|---  
**Number of Rules** | Number of rules in decision trees or rule sets (fewer is more understandable)  
**Number of Features** | Number of features used in explanation (fewer is better)  
**User Study** | Comprehension test by actual users  
  
* * *

## Practice Problems

Problem 1: Need for Model Interpretability

**Problem** : Explain why model interpretability is particularly important in the following scenarios.

  1. Bank loan approval system
  2. Medical image diagnosis support system
  3. Recommendation system

**Sample Answer** :

  1. **Loan Approval** : Obligation to explain rejection reasons (legal requirement), ensuring fairness, preventing discriminatory decisions
  2. **Medical Diagnosis** : Doctors' understanding of diagnostic reasoning, explanation to patients, reducing misdiagnosis risk, responding to medical malpractice litigation
  3. **Recommendation** : Improving user trust, transparency of recommendation reasons, bias detection (avoiding filter bubbles)

Problem 2: Global Interpretation and Local Interpretation

**Problem** : For a "customer churn prediction model," provide examples of information you would want to know from global interpretation and local interpretation respectively.

**Sample Answer** :

  * **Global Interpretation** : 
    * "Number of support inquiries" is the most influential feature for churn
    * Relationship between "contract duration" and churn probability (longer duration tends to have lower churn rate)
    * Top 5 most important features overall in the model
  * **Local Interpretation** : 
    * Reason why customer A (ID=12345) was predicted to churn (10+ support inquiries, contract duration less than 3 months, etc.)
    * Factors that should be improved to reduce this customer's churn probability

Problem 3: Choosing Interpretable Models

**Problem** : For the following scenarios, choose which interpretable model is appropriate and explain the reason.

  1. Housing price prediction (features: area, number of rooms, building age, etc.)
  2. Spam email classification (features: word frequency)
  3. Patient readmission risk prediction (features: age, diagnosis history, test values, etc.)

**Sample Answer** :

  1. **Linear Regression** : Coefficients of each feature directly indicate influence on price, making it understandable for real estate agents and customers
  2. **Decision Tree or Rule-Based** : Rules like "if the word 'free' appears 5+ times → spam" are intuitive
  3. **GAM or Decision Tree** : Can visualize nonlinear relationships (e.g., U-shaped relationship between age and readmission risk). Easy for doctors to understand diagnostic logic

Problem 4: Detecting Data Leakage

**Problem** : Explain how to detect data leakage using Feature Importance and provide a code example.

**Sample Answer** :
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Method for detecting data leakage
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    # Checklist of suspicious features
    suspicious_features = [
        'id', 'timestamp', 'created_at', 'updated_at',
        'target', 'label', 'outcome'  # Target variable itself or its leakage
    ]
    
    # Calculate Feature Importance
    model = RandomForestClassifier()
    # model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Check top features
    top_features = feature_importance.head(5)
    for _, row in top_features.iterrows():
        feature = row['feature']
        importance = row['importance']
    
        # Check if suspicious features are in top ranks
        if any(suspect in feature.lower() for suspect in suspicious_features):
            print(f"⚠️ Possible data leakage: {feature} (importance: {importance:.3f})")
    
        # Check if importance is abnormally high (>0.9)
        if importance > 0.9:
            print(f"⚠️ Abnormally high importance: {feature} (importance: {importance:.3f})")
    

Problem 5: Evaluating Interpretability

**Problem** : What metrics or methods can be used to evaluate "Fidelity" of interpretation techniques?

**Sample Answer** :

  * **R² Score** : Agreement between explanation model (such as LIME's linear approximation) and original black-box model predictions
  * **Prediction Error** : Mean absolute error (MAE) between explanation model and original model predictions
  * **Classification Accuracy Comparison** : How accurately the explanation model can reproduce the original model's predictions
  * **Local Fidelity** : How accurate the explanation model is in the neighborhood of a specific instance

Problem 6: Implementation Challenge

**Problem** : Using scikit-learn's Titanic dataset (or any dataset), implement the following.

  1. Train a logistic regression model and interpret its coefficients
  2. Train a decision tree model and extract its rules
  3. Train a random forest model and visualize Feature Importance
  4. Compare the interpretability of the three models

**Hint** :
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Hint:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.datasets import fetch_openml
    import pandas as pd
    
    # Load data
    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = titanic.frame
    
    # Preprocessing (missing value handling, categorical encoding, etc.)
    # ...
    
    # Train and interpret models
    # ...
    

* * *

## Summary

In this chapter, we learned the basics of model interpretability:

  * ✅ **Importance** : Essential for trustworthiness, regulatory compliance, debugging, and bias detection
  * ✅ **Classification** : Global/local, model-specific/agnostic, intrinsic/post-hoc interpretability
  * ✅ **Interpretable Models** : Linear regression, decision trees, rule-based, GAM
  * ✅ **Interpretation Techniques** : Feature Importance, PDP, SHAP, LIME, Saliency Maps
  * ✅ **Evaluation Criteria** : Fidelity, Consistency, Stability, Comprehensibility

In the next chapter, we will learn in detail about Feature Importance and Permutation Importance.

* * *

## References

  * Molnar, C. (2022). _Interpretable Machine Learning: A Guide for Making Black Box Models Explainable_. <https://christophm.github.io/interpretable-ml-book/>
  * Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." _NIPS 2017_.
  * Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." _KDD 2016_.
  * Rudin, C. (2019). "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead." _Nature Machine Intelligence_.
  * European Union. (2016). _General Data Protection Regulation (GDPR)_. Article 22.
  * Doshi-Velez, F., & Kim, B. (2017). "Towards A Rigorous Science of Interpretable Machine Learning." _arXiv:1702.08608_.
