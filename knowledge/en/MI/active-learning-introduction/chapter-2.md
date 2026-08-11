---
title: "Chapter 2: Uncertainty Estimation Techniques"
chapter_title: "Chapter 2: Uncertainty Estimation Techniques"
subtitle: Prediction Confidence Intervals with Ensemble, Dropout, and Gaussian Process
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 2: Uncertainty Estimation Techniques

This chapter covers Uncertainty Estimation Techniques. You will learn principles of three uncertainty estimation methods, Ensemble methods (Random Forest), and MC Dropout to neural networks.

**Prediction Confidence Intervals with Ensemble, Dropout, and Gaussian Process**

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the principles of three uncertainty estimation methods
  * ✅ Implement Ensemble methods (Random Forest)
  * ✅ Apply MC Dropout to neural networks
  * ✅ Calculate prediction variance with Gaussian Process
  * ✅ Explain the criteria for selecting among these methods

**Reading Time** : 25-30 minutes **Code Examples** : 8 examples **Exercises** : 3 problems

* * *

## 2.1 Uncertainty Estimation with Ensemble Methods

### Why Uncertainty Estimation is Important

In active learning, it is necessary to quantify "how confident the model is in its predictions." Uncertainty estimation is a core technology for query strategies.

**Two Types of Uncertainty** :

  1. **Aleatoric Uncertainty** \- Noise inherent in the data itself \- Measurement errors, environmental variations, etc. \- Does not decrease even with more data

  2. **Epistemic Uncertainty** \- Uncertainty due to the model's lack of knowledge \- Caused by insufficient data \- Decreases with more data

**Uncertainty Focused on by Active Learning** : → **Epistemic Uncertainty** (can be improved by adding data)

### Principle of Ensemble Methods

**Basic Idea** : Measure uncertainty by the variation in predictions from multiple models

**Formula** : $$ \mu(x) = \frac{1}{M} \sum_{m=1}^M f_m(x) $$

$$ \sigma^2(x) = \frac{1}{M} \sum_{m=1}^M (f_m(x) - \mu(x))^2 $$

  * $f_m(x)$: Prediction from the m-th model
  * $M$: Number of models (ensemble size)
  * $\mu(x)$: Prediction mean
  * $\sigma^2(x)$: Prediction variance (uncertainty)

### Implementation with Random Forest

**Code Example 1: Uncertainty Estimation with Random Forest**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    # Generate data
    np.random.seed(42)
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=10,
        random_state=42
    )
    
    # Split into training and test data
    train_size = 50
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Uncertainty estimation with Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Get predictions from each decision tree
    tree_predictions = np.array([
        tree.predict(X_test)
        for tree in rf.estimators_
    ])
    
    # Prediction mean and standard deviation
    mean_prediction = np.mean(tree_predictions, axis=0)
    std_prediction = np.std(tree_predictions, axis=0)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Left panel: Prediction vs true value
    plt.subplot(1, 2, 1)
    plt.errorbar(
        y_test,
        mean_prediction,
        yerr=1.96 * std_prediction,  # 95% confidence interval
        fmt='o',
        alpha=0.6,
        capsize=5
    )
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        label='Perfect prediction'
    )
    plt.xlabel('True Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title('Random Forest: Prediction with Uncertainty', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right panel: Distribution of uncertainty
    plt.subplot(1, 2, 2)
    plt.hist(std_prediction, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Standard Deviation (Uncertainty)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Uncertainty', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('rf_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("Random Forest uncertainty estimation results:")
    print(f"Mean uncertainty: {std_prediction.mean():.2f}")
    print(f"Minimum uncertainty: {std_prediction.min():.2f}")
    print(f"Maximum uncertainty: {std_prediction.max():.2f}")
    print(f"Standard deviation of uncertainty: {std_prediction.std():.2f}")

**OutputExample** :
    
    
    Random Forest uncertainty estimation results:
    Mean uncertainty: 5.23
    Minimum uncertainty: 2.14
    Maximum uncertainty: 12.45
    Standard deviation of uncertainty: 2.18

### Implementation with LightGBM

**Code Example 2: Uncertainty Estimation with LightGBM**
    
    
    import lightgbm as lgb
    
    # Train multiple models with LightGBM (Bagging)
    n_models = 100
    lgb_predictions = []
    
    for i in range(n_models):
        # Bootstrap sampling
        indices = np.random.choice(
            len(X_train),
            len(X_train),
            replace=True
        )
        X_boot = X_train[indices]
        y_boot = y_train[indices]
    
        # Train LightGBM
        train_data = lgb.Dataset(X_boot, label=y_boot)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
    
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100
        )
    
        # Predict
        pred = model.predict(X_test)
        lgb_predictions.append(pred)
    
    lgb_predictions = np.array(lgb_predictions)
    
    # Compute uncertainty
    lgb_mean = np.mean(lgb_predictions, axis=0)
    lgb_std = np.std(lgb_predictions, axis=0)
    
    print("\nLightGBM uncertainty estimation results:")
    print(f"Mean uncertainty: {lgb_std.mean():.2f}")
    print(f"Correlation with Random Forest: "
          f"{np.corrcoef(std_prediction, lgb_std)[0,1]:.3f}")

**Advantages** : \- ✅ Simple to implement \- ✅ Relatively low computational cost \- ✅ Easy to interpret \- ✅ Strong performance on tabular data

**Disadvantages** : \- ⚠️ Depends on ensemble size \- ⚠️ Difficult to apply to deep learning \- ⚠️ May require uncertainty calibration

* * *

## 2.2 Uncertainty Estimation with Dropout Methods

### MC Dropout (Monte Carlo Dropout)

**Principle** : Apply dropout during inference as well and measure variation through multiple predictions

**Regular Dropout** (training only):
    
    
    # During training
    model.train()  # Dropout enabled
    output = model(x)
    
    # During inference
    model.eval()  # Dropout disabled
    output = model(x)  # Deterministic prediction

**MC Dropout** (dropout during inference too):
    
    
    # Enable dropout during inference as well
    model.train()  # Dropout stays enabled
    predictions = [model(x) for _ in range(T)]  # T predictions
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)

### Implementation Example

**Code Example 3: MC Dropout with PyTorch**
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MCDropoutNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=50, dropout_rate=0.5):
            super(MCDropoutNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(p=dropout_rate)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)  # Apply dropout
            x = F.relu(self.fc2(x))
            x = self.dropout(x)  # Apply dropout
            x = self.fc3(x)
            return x
    
    # Initialize model
    model = MCDropoutNet(input_dim=5, hidden_dim=50, dropout_rate=0.3)
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')
    
    # Uncertainty estimation with MC Dropout
    def mc_dropout_predict(model, x, n_samples=100):
        """
        Prediction and uncertainty estimation with MC Dropout
    
        Parameters:
        -----------
        model : nn.Module
            Trained model
        x : Tensor
            Input data
        n_samples : int
            Number of sampling iterations
    
        Returns:
        --------
        mean : array
            Prediction mean
        std : array
            Prediction standard deviation (uncertainty)
        """
        model.train()  # Enable dropout
        predictions = []
    
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(x).numpy()
                predictions.append(pred)
    
        predictions = np.array(predictions).squeeze()
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
    
        return mean, std
    
    # Predict with MC Dropout
    mc_mean, mc_std = mc_dropout_predict(
        model,
        X_test_tensor,
        n_samples=100
    )
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        y_test,
        mc_mean,
        yerr=1.96 * mc_std,
        fmt='o',
        alpha=0.6,
        capsize=5,
        color='purple'
    )
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        label='Perfect prediction'
    )
    plt.xlabel('True Value', fontsize=12)
    plt.ylabel('Predicted Value (MC Dropout)', fontsize=12)
    plt.title('MC Dropout: Uncertainty Estimation', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mc_dropout_uncertainty.png', dpi=150)
    plt.show()
    
    print("\nMC Dropout uncertainty estimation results:")
    print(f"Mean uncertainty: {mc_std.mean():.2f}")
    print(f"Minimum uncertainty: {mc_std.min():.2f}")
    print(f"Maximum uncertainty: {mc_std.max():.2f}")

**OutputExample** :
    
    
    Epoch [50/200], Loss: 145.2341
    Epoch [100/200], Loss: 98.5632
    Epoch [150/200], Loss: 67.8921
    Epoch [200/200], Loss: 52.1234
    
    MC Dropout uncertainty estimation results:
    Mean uncertainty: 4.87
    Minimum uncertainty: 1.92
    Maximum uncertainty: 11.23

**Advantages** : \- ✅ Easy to apply to existing neural networks \- ✅ No additional training required (dropout only) \- ✅ Well-suited for deep learning

**Disadvantages** : \- ⚠️ Computational cost depends on sampling count (T) \- ⚠️ Choice of dropout rate is important \- ⚠️ May require uncertainty calibration

* * *

## 2.3 Uncertainty Estimation with Gaussian Process (GP)

### Fundamentals of GP

Gaussian Process is a powerful method for defining probability distributions over functions.

**Definition** : $$ f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) $$

  * $\mu(\mathbf{x})$: Mean function (usually 0)
  * $k(\mathbf{x}, \mathbf{x}')$: Kernel function (covariance function)

**Predictive Distribution** : $$ p(f^* | \mathbf{X}, \mathbf{y}, \mathbf{x}^*) = \mathcal{N}(\mu^*, \sigma^{*2}) $$

$$ \mu^* = k(\mathbf{x}^*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} \mathbf{y} $$

$$ \sigma^{*2} = k(\mathbf{x}^*, \mathbf{x}^*) - k(\mathbf{x}^*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} k(\mathbf{X}, \mathbf{x}^*) $$

### Kernel Functions

**RBF (Radial Basis Function) Kernel** : $$ k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \exp\left(-\frac{|\mathbf{x}_i - \mathbf{x}_j|^2}{2\ell^2}\right) $$

  * $\sigma_f^2$: Signal variance
  * $\ell$: Length scale (smoothness)

**Matérn Kernel** : $$ k(\mathbf{x}_i, \mathbf{x}_j) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} r}{\ell}\right) $$

### Implementation with GPyTorch

**Code Example 4: Uncertainty Estimation with GPyTorch**
    
    
    import gpytorch
    import torch
    
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # Convert data to tensors
    train_x = torch.FloatTensor(X_train)
    train_y = torch.FloatTensor(y_train)
    test_x = torch.FloatTensor(X_test)
    
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Training mode
    model.train()
    likelihood.train()
    
    # Configure optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Loss function (Marginal Log Likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    n_iterations = 100
    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
    
        if (i + 1) % 20 == 0:
            print(f'Iteration {i+1}/{n_iterations} - Loss: {loss.item():.3f}')
    
        optimizer.step()
    
    # Inference mode
    model.eval()
    likelihood.eval()
    
    # Prediction (with uncertainty)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        gp_mean = observed_pred.mean.numpy()
        gp_std = observed_pred.stddev.numpy()
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        y_test,
        gp_mean,
        yerr=1.96 * gp_std,
        fmt='o',
        alpha=0.6,
        capsize=5,
        color='green'
    )
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        label='Perfect prediction'
    )
    plt.xlabel('True Value', fontsize=12)
    plt.ylabel('Predicted Value (GP)', fontsize=12)
    plt.title('Gaussian Process: Uncertainty Estimation', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gp_uncertainty.png', dpi=150)
    plt.show()
    
    print("\nGaussian Process uncertainty estimation results:")
    print(f"Mean uncertainty: {gp_std.mean():.2f}")
    print(f"Minimum uncertainty: {gp_std.min():.2f}")
    print(f"Maximum uncertainty: {gp_std.max():.2f}")
    
    # Learned hyperparameters
    print("\nLearned hyperparameters:")
    print(f"Length scale: "
          f"{model.covar_module.base_kernel.lengthscale.item():.3f}")
    print(f"Signal variance: "
          f"{model.covar_module.outputscale.item():.3f}")
    print(f"Noise variance: "
          f"{likelihood.noise.item():.3f}")

**OutputExample** :
    
    
    Iteration 20/100 - Loss: 145.234
    Iteration 40/100 - Loss: 98.567
    Iteration 60/100 - Loss: 67.891
    Iteration 80/100 - Loss: 52.123
    Iteration 100/100 - Loss: 45.678
    
    Gaussian Process uncertainty estimation results:
    Mean uncertainty: 5.12
    Minimum uncertainty: 2.34
    Maximum uncertainty: 10.87
    
    Learned hyperparameters:
    Length scale: 1.234
    Signal variance: 45.678
    Noise variance: 3.456

**Advantages** : \- ✅ Rigorous uncertainty quantification \- ✅ High accuracy with small datasets \- ✅ Flexibility through kernel selection \- ✅ Strong theoretical foundation

**Disadvantages** : \- ⚠️ Not suitable for large-scale data (O(n³)) \- ⚠️ Kernel and hyperparameter selection is important \- ⚠️ Performance degrades with high-dimensional data

* * *

## 2.4 Case Study: Band Gap Prediction

### Problem Setup

**Objective** : Predict the band gap of inorganic materials and prioritize calculations for samples with high uncertainty

**Dataset** : Materials Project (DFT calculations completed) \- Number of samples: 5,000 materials \- Features: Compositional descriptors (20-dimensional) \- Target variable: Band Gap (eV)

### Comparison of Three Methods

**Code Example 5: Comparison of Uncertainty Estimation for Band Gap Prediction**
    
    
    """
    Comparison of three uncertainty estimation methods for band gap prediction
    
    Data: Materials Project-style synthetic dataset
    Goal: Compare the performance of Random Forest, MC Dropout, and Gaussian Process
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import gpytorch
    
    
    # ============================================
    # Data generation and preprocessing
    # ============================================
    def generate_bandgap_dataset(n_samples=5000, n_features=20,
                                  random_state=42):
        """
        Generate a Materials Project-style band gap dataset
    
        Parameters:
        -----------
        n_samples : int
            Number of samples (number of materials)
        n_features : int
            Feature dimension (compositional descriptors)
        random_state : int
            Random seed
    
        Returns:
        --------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix (compositional descriptors)
        y : ndarray, shape (n_samples,)
            Band gap (eV)
        """
        np.random.seed(random_state)
    
        # Simulate compositional descriptors (normal distribution)
        X = np.random.randn(n_samples, n_features)
    
        # Generate the band gap with a nonlinear function
        # Real band gaps are roughly 0-8 eV
        true_weights = np.random.randn(n_features) * 0.3
        y = (
            2.5  # Base value
            + X @ true_weights  # Linear component
            + 0.5 * np.sin(X[:, 0])  # Nonlinear component
            + 0.3 * (X[:, 1] ** 2)
            + np.random.randn(n_samples) * 0.2  # Noise
        )
    
        # Clip the band gap to a physically reasonable range
        y = np.clip(y, 0.0, 8.0)
    
        return X, y
    
    
    # Generate data
    print("Generating band gap dataset...")
    X, y = generate_bandgap_dataset(n_samples=500, n_features=20)
    
    # Split into training and test data (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )
    
    # Standardization (for GP and NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"Band gap range: {y.min():.2f} - {y.max():.2f} eV\n")
    
    
    # ============================================
    # Method 1: Random Forest
    # ============================================
    print("=" * 50)
    print("Estimating uncertainty with Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Get predictions from each decision tree
    rf_tree_preds = np.array([
        tree.predict(X_test) for tree in rf_model.estimators_
    ])
    
    # Prediction mean and standard deviation
    rf_mean = np.mean(rf_tree_preds, axis=0)
    rf_std = np.std(rf_tree_preds, axis=0)
    rf_time = time.time() - start_time
    
    print(f"Done ({rf_time:.2f}s)")
    print(f"RMSE: {np.sqrt(np.mean((rf_mean - y_test) ** 2)):.3f} eV")
    
    
    # ============================================
    # Method 2: MC Dropout
    # ============================================
    print("=" * 50)
    print("Estimating uncertainty with MC Dropout...")
    
    # Define the MC Dropout model
    class BandgapMCDropout(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
            super(BandgapMCDropout, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(p=dropout_rate)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.fc4(x)
            return x
    
    
    start_time = time.time()
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Train the model
    mc_model = BandgapMCDropout(input_dim=20, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mc_model.parameters(), lr=0.01)
    
    mc_model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        outputs = mc_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Predict with MC Dropout (100 samples)
    mc_model.train()  # Enable dropout
    mc_predictions = []
    with torch.no_grad():
        for _ in range(100):
            pred = mc_model(X_test_tensor).numpy().flatten()
            mc_predictions.append(pred)
    
    mc_predictions = np.array(mc_predictions)
    mc_mean = np.mean(mc_predictions, axis=0)
    mc_std = np.std(mc_predictions, axis=0)
    mc_time = time.time() - start_time
    
    print(f"Done ({mc_time:.2f}s)")
    print(f"RMSE: {np.sqrt(np.mean((mc_mean - y_test) ** 2)):.3f} eV")
    
    
    # ============================================
    # Method 3: Gaussian Process
    # ============================================
    print("=" * 50)
    print("Estimating uncertainty with Gaussian Process...")
    
    # Define the GP
    class BandgapGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(BandgapGP, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(
                mean_x, covar_x
            )
    
    
    start_time = time.time()
    
    # Convert data to tensors
    gp_train_x = torch.FloatTensor(X_train_scaled)
    gp_train_y = torch.FloatTensor(y_train)
    gp_test_x = torch.FloatTensor(X_test_scaled)
    
    # Train the GP
    gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = BandgapGP(gp_train_x, gp_train_y, gp_likelihood)
    gp_model.train()
    gp_likelihood.train()
    
    gp_optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
    gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        gp_likelihood, gp_model
    )
    
    for i in range(100):
        gp_optimizer.zero_grad()
        output = gp_model(gp_train_x)
        loss = -gp_mll(output, gp_train_y)
        loss.backward()
        gp_optimizer.step()
    
    # GP prediction
    gp_model.eval()
    gp_likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_pred = gp_likelihood(gp_model(gp_test_x))
        gp_mean = gp_pred.mean.numpy()
        gp_std = gp_pred.stddev.numpy()
    
    gp_time = time.time() - start_time
    
    print(f"Done ({gp_time:.2f}s)")
    print(f"RMSE: {np.sqrt(np.mean((gp_mean - y_test) ** 2)):.3f} eV")
    
    
    # ============================================
    # Computing the Calibration Curve
    # ============================================
    def compute_calibration_curve(y_true, y_pred, y_std, n_bins=10):
        """
        Compute the calibration curve of the predictions
    
        Parameters:
        -----------
        y_true : array
            True values
        y_pred : array
            Prediction mean
        y_std : array
            Prediction standard deviation
        n_bins : int
            Number of bins
    
        Returns:
        --------
        expected_freq : array
            Expected frequency
        observed_freq : array
            Observed frequency
        """
        # Compute normalized residuals
        residuals = (y_true - y_pred) / y_std
    
        # Define confidence levels (-3 sigma to +3 sigma)
        confidence_levels = np.linspace(0.01, 0.99, n_bins)
        expected_freq = confidence_levels
        observed_freq = []
    
        for conf in confidence_levels:
            # Compute the bounds of the confidence interval
            z_score = np.abs(
                np.percentile(np.random.randn(10000), conf * 100)
            )
            # Compute the fraction that falls inside the interval
            in_interval = np.abs(residuals) <= z_score
            observed_freq.append(np.mean(in_interval))
    
        return expected_freq, np.array(observed_freq)
    
    
    # ============================================
    # Comparison visualization
    # ============================================
    methods = {
        'Random Forest': (rf_mean, rf_std, 'blue'),
        'MC Dropout': (mc_mean, mc_std, 'purple'),
        'Gaussian Process': (gp_mean, gp_std, 'green')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) Distribution of uncertainty
    ax = axes[0, 0]
    for method_name, (_, std_values, color) in methods.items():
        ax.hist(
            std_values,
            bins=30,
            alpha=0.5,
            label=method_name,
            color=color
        )
    ax.set_xlabel('Uncertainty (Standard Deviation)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Uncertainty', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # (2) Uncertainty vs prediction error
    ax = axes[0, 1]
    for method_name, (pred_mean, std_values, color) in methods.items():
        errors = np.abs(y_test - pred_mean)
        ax.scatter(
            std_values,
            errors,
            alpha=0.5,
            label=method_name,
            s=30,
            color=color
        )
    
        # Compute the correlation coefficient
        corr = np.corrcoef(std_values, errors)[0, 1]
        print(f"\n{method_name} - Correlation between uncertainty and error: {corr:.3f}")
    
    ax.set_xlabel('Uncertainty', fontsize=12)
    ax.set_ylabel('Prediction Error (|True - Pred|)', fontsize=12)
    ax.set_title('Uncertainty vs Error', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (3) Calibration Curve
    ax = axes[1, 0]
    for method_name, (pred_mean, std_values, color) in methods.items():
        expected, observed = compute_calibration_curve(
            y_test, pred_mean, std_values, n_bins=10
        )
        ax.plot(
            expected,
            observed,
            marker='o',
            label=method_name,
            color=color,
            linewidth=2
        )
    
    # Perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        label='Perfect calibration',
        linewidth=2
    )
    ax.set_xlabel('Expected Confidence Level', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (4) Comparison of computation time
    ax = axes[1, 1]
    computation_times = [rf_time, mc_time, gp_time]
    colors = ['blue', 'purple', 'green']
    bars = ax.bar(
        ['Random\nForest', 'MC\nDropout', 'Gaussian\nProcess'],
        computation_times,
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )
    
    # Display the time on each bar
    for bar, time_val in zip(bars, computation_times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{time_val:.2f}s',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Computational Cost', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('uncertainty_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Overall comparison of uncertainty estimation")
    print("=" * 50)
    for method_name, (pred_mean, std_values, _) in methods.items():
        rmse = np.sqrt(np.mean((pred_mean - y_test) ** 2))
        print(f"\n{method_name}:")
        print(f"  RMSE: {rmse:.3f} eV")
        print(f"  Mean uncertainty: {std_values.mean():.3f}")
        print(f"  Uncertainty range: [{std_values.min():.3f}, "
              f"{std_values.max():.3f}]")

**OutputExample** :
    
    
    Generating band gap dataset...
    Training data: 350 samples
    Test data: 150 samples
    Band gap range: 0.00 - 7.92 eV
    
    ==================================================
    Estimating uncertainty with Random Forest...
    Done (0.58s)
    RMSE: 0.423 eV
    ==================================================
    Estimating uncertainty with MC Dropout...
    Done (3.21s)
    RMSE: 0.387 eV
    ==================================================
    Estimating uncertainty with Gaussian Process...
    Done (1.87s)
    RMSE: 0.356 eV
    
    Random Forest - Correlation between uncertainty and error: 0.621
    MC Dropout - Correlation between uncertainty and error: 0.684
    Gaussian Process - Correlation between uncertainty and error: 0.743
    
    ==================================================
    Overall comparison of uncertainty estimation
    ==================================================
    
    Random Forest:
      RMSE: 0.423 eV
      Mean uncertainty: 0.287
      Uncertainty range: [0.134, 0.612]
    
    MC Dropout:
      RMSE: 0.387 eV
      Mean uncertainty: 0.312
      Uncertainty range: [0.156, 0.698]
    
    Gaussian Process:
      RMSE: 0.356 eV
      Mean uncertainty: 0.298
      Uncertainty range: [0.142, 0.721]

* * *

## 2.5 Chapter Summary

### What We Learned

  1. **Ensemble Methods** \- Uncertainty estimation with Random Forest and LightGBM \- Quantify uncertainty through prediction variance \- Simple to implement, moderate computational cost

  2. **MC Dropout** \- Apply dropout during inference as well \- Easy to implement with neural networks \- Sampling count and dropout rate are important

  3. **Gaussian Process** \- Rigorous uncertainty quantification \- Flexibility through kernel functions \- High accuracy with small data, not suitable for large-scale data

### Selecting the Right Method

Method | Recommended Case | Data Size | Computational Cost  
---|---|---|---  
Random Forest | Tabular data, medium-scale | 100-10,000 | Low to Medium  
MC Dropout | Deep learning, images/text | 1,000-100,000 | Medium to High  
Gaussian Process | Small datasets, rigorous uncertainty | 10-1,000 | Medium to High  
  
### Next Chapter

In Chapter 3, we will learn about **acquisition function design** that leverages uncertainty: \- Expected Improvement (EI) \- Probability of Improvement (PI) \- Upper Confidence Bound (UCB) \- Multi-objective and constrained acquisition functions

**[Chapter 3: Acquisition Function Design →](<chapter-3.html>)**

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

(Omitted: Detailed implementation of exercises)

### Problem 2 (Difficulty: Medium)

(Omitted: Detailed implementation of exercises)

### Problem 3 (Difficulty: Hard)

(Omitted: Detailed implementation of exercises)

* * *

## References

  1. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning." _ICML_ , 1050-1059.

  2. Rasmussen, C. E., & Williams, C. K. I. (2006). _Gaussian Processes for Machine Learning_. MIT Press.

  3. Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." _NeurIPS_.

* * *

## Navigation

### Previous Chapter

**[← Chapter 1: The Need for Active Learning](<chapter-1.html>)**

### Next Chapter

**[Chapter 3: Acquisition Function Design →](<chapter-3.html>)**

### Series Index

**[← Back to Series Index](<./index.html>)**

* * *

**Let's learn about acquisition function design in the next chapter!**
