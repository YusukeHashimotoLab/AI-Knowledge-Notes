---
title: "Chapter 1: Fundamentals of Probability and Statistics"
chapter_title: "Chapter 1: Fundamentals of Probability and Statistics"
---

This chapter covers the fundamentals of Fundamentals of Probability and Statistics, which 1. fundamentals of probability. You will learn essential concepts and techniques.

**Deeply understand the probability and statistics that form the foundation of machine learning through both theory and implementation**

**What You Will Learn in This Chapter**

  * Mathematical understanding of Bayes' theorem and conditional probability
  * Properties and implementation of normal distributions and multivariate normal distributions
  * Calculation and geometric interpretation of expectation, variance, and covariance
  * Theoretical differences between maximum likelihood estimation and Bayesian estimation
  * Applications of probability and statistics to machine learning algorithms

## 1\. Fundamentals of Probability

### 1.1 Conditional Probability and Bayes' Theorem

Conditional probability represents the probability of event A occurring given that event B has occurred.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$ 

Bayes' theorem is the fundamental theorem for calculating posterior probability from prior probability and likelihood.

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{P(B|A)P(A)}{\sum_{i} P(B|A_i)P(A_i)}$$ 

Where:

  * **P(A)** : Prior probability - probability of the hypothesis before observation
  * **P(B|A)** : Likelihood - probability of observing data B given hypothesis A
  * **P(A|B)** : Posterior probability - probability of the hypothesis after observing data
  * **P(B)** : Marginal likelihood (evidence) - normalization constant for the data

**Applications in Machine Learning** Bayes' theorem forms the foundation for many machine learning algorithms, including Naive Bayes classifiers, Bayesian linear regression, and Bayesian optimization. 

### Implementation Example 1: Naive Bayes Classifier

This is an implementation example of document classification using Bayes' theorem, assuming each word occurrence is independent.
    
    
    import numpy as np
    from collections import defaultdict
    
    class NaiveBayesClassifier:
        """Implementation of Naive Bayes Classifier"""
    
        def __init__(self, alpha=1.0):
            """
            Parameters:
            -----------
            alpha : float
                Laplace smoothing parameter (additive smoothing)
            """
            self.alpha = alpha
            self.class_priors = {}
            self.word_probs = defaultdict(dict)
            self.vocab = set()
    
        def fit(self, X, y):
            """
            Learn probabilities from training data
    
            Parameters:
            -----------
            X : list of list
                List of words for each document
            y : list
                Class label for each document
            """
            n_docs = len(X)
            class_counts = defaultdict(int)
            word_counts = defaultdict(lambda: defaultdict(int))
    
            # Count documents per class and word occurrences
            for doc, label in zip(X, y):
                class_counts[label] += 1
                for word in doc:
                    self.vocab.add(word)
                    word_counts[label][word] += 1
    
            # Calculate prior probability P(class)
            for label, count in class_counts.items():
                self.class_priors[label] = count / n_docs
    
            # Calculate likelihood P(word|class) with Laplace smoothing
            vocab_size = len(self.vocab)
            for label in class_counts:
                total_words = sum(word_counts[label].values())
                for word in self.vocab:
                    word_count = word_counts[label].get(word, 0)
                    # P(word|class) with Laplace smoothing
                    self.word_probs[label][word] = (
                        (word_count + self.alpha) /
                        (total_words + self.alpha * vocab_size)
                    )
    
        def predict(self, X):
            """
            Calculate posterior probability using Bayes' theorem and predict the most probable class
    
            log P(class|doc) = log P(class) + Σ log P(word|class)
            """
            predictions = []
            for doc in X:
                class_scores = {}
                for label in self.class_priors:
                    # Calculate log posterior probability (for numerical stability)
                    score = np.log(self.class_priors[label])
                    for word in doc:
                        if word in self.vocab:
                            score += np.log(self.word_probs[label][word])
                    class_scores[label] = score
                predictions.append(max(class_scores, key=class_scores.get))
            return predictions
    
    # Usage example
    X_train = [
        ['machine', 'learning', 'deep', 'learning'],
        ['statistics', 'probability', 'distribution'],
        ['deep', 'neural', 'network'],
        ['probability', 'Bayes', 'statistics']
    ]
    y_train = ['ML', 'Stats', 'ML', 'Stats']
    
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(X_train, y_train)
    
    X_test = [['machine', 'learning'], ['Bayes', 'probability']]
    predictions = nb.predict(X_test)
    print(f"Predictions: {predictions}")  # ['ML', 'Stats']
    

## 2\. Probability Distributions

### 2.1 Normal Distribution (Gaussian Distribution)

The normal distribution is the most important continuous probability distribution observed in many natural phenomena and measurement errors.

$$\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$ 

Where:

  * **μ** : Mean (expectation) - center position of the distribution
  * **σ²** : Variance - degree of data dispersion
  * **σ** : Standard deviation - square root of variance

**Central Limit Theorem** The sum of independent and identically distributed random variables approaches a normal distribution as the sample size increases, regardless of the shape of the original distribution. This is one of the reasons why the normal distribution is so important. 

### 2.2 Multivariate Normal Distribution

The multivariate normal distribution, which describes the probability distribution of multidimensional data, is frequently used in machine learning.

$$\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$ 

Where:

  * **μ** : D-dimensional mean vector
  * **Σ** : D×D covariance matrix (symmetric positive definite matrix)
  * **|Σ|** : Determinant of the covariance matrix

### Implementation Example 2: Visualization of Multivariate Normal Distribution
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    
    def plot_multivariate_gaussian():
        """Visualization of 2D multivariate normal distribution"""
    
        # Define mean vector and covariance matrices
        mu = np.array([0, 0])
    
        # Different covariance matrix cases
        covariances = [
            np.array([[1, 0], [0, 1]]),           # Independent, equal variance
            np.array([[2, 0], [0, 0.5]]),         # Independent, different variance
            np.array([[1, 0.8], [0.8, 1]]),       # Positive correlation
            np.array([[1, -0.8], [-0.8, 1]])      # Negative correlation
        ]
    
        titles = ['Independent, Equal Variance', 'Independent, Different Variance', 'Positive Correlation', 'Negative Correlation']
    
        # Generate grid points
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
    
        for idx, (cov, title) in enumerate(zip(covariances, titles)):
            # Calculate multivariate normal distribution
            rv = multivariate_normal(mu, cov)
            Z = rv.pdf(pos)
    
            # Contour plot
            axes[idx].contour(X, Y, Z, levels=10, cmap='viridis')
            axes[idx].set_title(f'{title}\nΣ = {cov.tolist()}')
            axes[idx].set_xlabel('x₁')
            axes[idx].set_ylabel('x₂')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axis('equal')
    
        plt.tight_layout()
        plt.savefig('multivariate_gaussian.png', dpi=150, bbox_inches='tight')
        print("Saved multivariate normal distribution visualization")
    
    # Execute
    plot_multivariate_gaussian()
    
    # Analyze covariance matrix through eigenvalue decomposition
    cov = np.array([[2, 1], [1, 2]])
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    print(f"\nCovariance matrix:\n{cov}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    print(f"Direction of principal axis (first eigenvector): {eigenvectors[:, 0]}")
    

## 3\. Expectation and Variance

### 3.1 Expectation

The expectation (mean) represents the "central value" of a random variable.

$$\mathbb{E}[X] = \sum_{x} x \cdot P(X=x) \quad \text{(discrete)}$$ $$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) dx \quad \text{(continuous)}$$ 

**Properties of Expectation:**

  * Linearity: \\(\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]\\)
  * Product of independent variables: \\(\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]\\) (when X and Y are independent)

### 3.2 Variance and Covariance

Variance is a measure of data dispersion.

$$\text{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$ 

Covariance represents the joint variation of two random variables.

$$\text{Cov}[X, Y] = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$ 

The correlation coefficient is the normalized covariance, taking values between -1 and 1.

$$\rho_{X,Y} = \frac{\text{Cov}[X, Y]}{\sqrt{\text{Var}[X]\text{Var}[Y]}}$$ 

### Implementation Example 3: Calculation of Expectation, Variance, and Covariance
    
    
    import numpy as np
    
    class StatisticsCalculator:
        """Class for calculating basic probability and statistics quantities"""
    
        @staticmethod
        def expectation(X, P=None):
            """
            Calculate expectation
    
            Parameters:
            -----------
            X : array-like
                Values of random variable
            P : array-like, optional
                Probability of each value (assumes uniform distribution if None)
    
            Returns:
            --------
            float : Expectation
            """
            X = np.array(X)
            if P is None:
                return np.mean(X)
            else:
                P = np.array(P)
                assert abs(np.sum(P) - 1.0) < 1e-10, "Sum of probabilities must be 1"
                return np.sum(X * P)
    
        @staticmethod
        def variance(X, P=None):
            """
            Calculate variance: Var[X] = E[X²] - (E[X])²
            """
            X = np.array(X)
            E_X = StatisticsCalculator.expectation(X, P)
            E_X2 = StatisticsCalculator.expectation(X**2, P)
            return E_X2 - E_X**2
    
        @staticmethod
        def covariance(X, Y):
            """
            Calculate covariance: Cov[X,Y] = E[XY] - E[X]E[Y]
            """
            X, Y = np.array(X), np.array(Y)
            assert len(X) == len(Y), "X and Y must have the same length"
    
            E_X = np.mean(X)
            E_Y = np.mean(Y)
            E_XY = np.mean(X * Y)
    
            return E_XY - E_X * E_Y
    
        @staticmethod
        def correlation(X, Y):
            """
            Calculate correlation coefficient: ρ = Cov[X,Y] / (σ_X * σ_Y)
            """
            cov_XY = StatisticsCalculator.covariance(X, Y)
            std_X = np.sqrt(StatisticsCalculator.variance(X))
            std_Y = np.sqrt(StatisticsCalculator.variance(Y))
    
            return cov_XY / (std_X * std_Y)
    
        @staticmethod
        def covariance_matrix(data):
            """
            Calculate covariance matrix
    
            Parameters:
            -----------
            data : ndarray of shape (n_samples, n_features)
                Data matrix
    
            Returns:
            --------
            ndarray : Covariance matrix (n_features, n_features)
            """
            data = np.array(data)
            n_samples, n_features = data.shape
    
            # Calculate mean of each feature
            means = np.mean(data, axis=0)
    
            # Centering
            centered_data = data - means
    
            # Covariance matrix: (1/n) * X^T X
            cov_matrix = (centered_data.T @ centered_data) / n_samples
    
            return cov_matrix
    
    # Usage example
    calc = StatisticsCalculator()
    
    # Discrete random variable (dice)
    X = [1, 2, 3, 4, 5, 6]
    P = [1/6] * 6
    print(f"Expectation of dice: {calc.expectation(X, P):.2f}")
    print(f"Variance of dice: {calc.variance(X, P):.2f}")
    
    # Continuous data
    np.random.seed(42)
    data = np.random.randn(1000, 3)  # 3-dimensional data
    cov_matrix = calc.covariance_matrix(data)
    print(f"\nCovariance matrix:\n{cov_matrix}")
    
    # Compare with NumPy function
    cov_numpy = np.cov(data.T)
    print(f"\nNumPy covariance matrix:\n{cov_numpy}")
    print(f"Maximum difference: {np.max(np.abs(cov_matrix - cov_numpy)):.10f}")
    

## 4\. Maximum Likelihood Estimation and Bayesian Estimation

### 4.1 Maximum Likelihood Estimation (MLE)

Maximum likelihood estimation is a method to find parameters that maximize the probability (likelihood) of obtaining the observed data.

$$\hat{\theta}_{ML} = \arg\max_{\theta} P(D|\theta) = \arg\max_{\theta} \prod_{i=1}^{N} P(x_i|\theta)$$ 

Using log-likelihood simplifies the calculation:

$$\hat{\theta}_{ML} = \arg\max_{\theta} \log P(D|\theta) = \arg\max_{\theta} \sum_{i=1}^{N} \log P(x_i|\theta)$$ 

**MLE for Normal Distribution** The MLE estimates of the normal distribution parameters μ and σ² coincide with the sample mean and sample variance: $$\hat{\mu}_{ML} = \frac{1}{N}\sum_{i=1}^{N}x_i, \quad \hat{\sigma}^2_{ML} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{\mu})^2$$ 

### 4.2 Bayesian Estimation and MAP Estimation

In Bayesian estimation, we calculate the posterior distribution from the prior distribution and the likelihood of the data.

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} \propto P(D|\theta)P(\theta)$$ 

MAP estimation (Maximum A Posteriori) maximizes the posterior probability:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} P(D|\theta)P(\theta)$$ 

### Implementation Example 4: MLE and MAP Estimation for Normal Distribution
    
    
    import numpy as np
    from scipy.stats import norm
    
    class GaussianEstimator:
        """Parameter estimation for normal distribution"""
    
        @staticmethod
        def mle(data):
            """
            Maximum Likelihood Estimation (MLE)
    
            Parameters:
            -----------
            data : array-like
                Observed data
    
            Returns:
            --------
            tuple : (mean estimate, variance estimate)
            """
            data = np.array(data)
            n = len(data)
    
            # MLE: sample mean and sample variance
            mu_mle = np.mean(data)
            sigma2_mle = np.mean((data - mu_mle)**2)  # 1/n * Σ(x - μ)²
    
            return mu_mle, sigma2_mle
    
        @staticmethod
        def map_estimation(data, prior_mu=0, prior_sigma=1, prior_alpha=1, prior_beta=1):
            """
            MAP Estimation (Maximum A Posteriori)
    
            Prior distributions:
            - μ ~ N(prior_mu, prior_sigma²)
            - σ² ~ InverseGamma(prior_alpha, prior_beta)
    
            Parameters:
            -----------
            data : array-like
                Observed data
            prior_mu : float
                Mean of prior distribution for mean
            prior_sigma : float
                Standard deviation of prior distribution for mean
            prior_alpha, prior_beta : float
                Parameters of prior distribution (inverse gamma) for variance
    
            Returns:
            --------
            tuple : (MAP estimate of mean, MAP estimate of variance)
            """
            data = np.array(data)
            n = len(data)
            sample_mean = np.mean(data)
            sample_var = np.mean((data - sample_mean)**2)
    
            # MAP estimate of mean (when prior distribution is normal)
            # Posterior mean is precision-weighted average of prior and likelihood
            precision_prior = 1 / prior_sigma**2
            precision_likelihood = n / sample_var
    
            mu_map = (precision_prior * prior_mu + precision_likelihood * sample_mean) / \
                     (precision_prior + precision_likelihood)
    
            # MAP estimate of variance (when prior distribution is inverse gamma)
            # Simplified estimate considering prior distribution influence
            alpha_post = prior_alpha + n / 2
            beta_post = prior_beta + 0.5 * np.sum((data - mu_map)**2)
    
            sigma2_map = beta_post / (alpha_post + 1)
    
            return mu_map, sigma2_map
    
        @staticmethod
        def compare_estimators(data, true_mu=0, true_sigma=1):
            """Compare MLE and MAP estimation"""
    
            mu_mle, sigma2_mle = GaussianEstimator.mle(data)
            mu_map, sigma2_map = GaussianEstimator.map_estimation(
                data, prior_mu=true_mu, prior_sigma=true_sigma
            )
    
            print(f"True values: μ={true_mu:.3f}, σ²={true_sigma**2:.3f}")
            print(f"\nMLE estimation:")
            print(f"  μ̂_MLE = {mu_mle:.3f}, σ̂²_MLE = {sigma2_mle:.3f}")
            print(f"  Error: |μ-μ̂|={abs(true_mu-mu_mle):.3f}, |σ²-σ̂²|={abs(true_sigma**2-sigma2_mle):.3f}")
    
            print(f"\nMAP estimation:")
            print(f"  μ̂_MAP = {mu_map:.3f}, σ̂²_MAP = {sigma2_map:.3f}")
            print(f"  Error: |μ-μ̂|={abs(true_mu-mu_map):.3f}, |σ²-σ̂²|={abs(true_sigma**2-sigma2_map):.3f}")
    
            return (mu_mle, sigma2_mle), (mu_map, sigma2_map)
    
    # Usage example
    np.random.seed(42)
    
    # Small sample size case (MAP estimation advantage)
    print("=" * 50)
    print("Comparison with small sample (n=10)")
    print("=" * 50)
    data_small = np.random.normal(0, 1, size=10)
    GaussianEstimator.compare_estimators(data_small)
    
    # Large sample size case (MLE and MAP converge)
    print("\n" + "=" * 50)
    print("Comparison with large sample (n=1000)")
    print("=" * 50)
    data_large = np.random.normal(0, 1, size=1000)
    GaussianEstimator.compare_estimators(data_large)
    

### 4.3 Comparison of MLE and Bayesian Estimation

Aspect | Maximum Likelihood Estimation (MLE) | Bayesian Estimation  
---|---|---  
Parameter Treatment | Fixed value (point estimate) | Random variable (distribution estimate)  
Prior Knowledge | Not used | Expressed through prior distribution  
With Limited Data | Prone to overfitting | Supplemented with prior knowledge  
Computational Cost | Low | High (requires integration)  
Uncertainty Representation | Point estimate only | Entire posterior distribution  
  
## 5\. Practical Applications

### 5.1 Gaussian Mixture Model (GMM)

A Gaussian Mixture Model represents complex distributions as a weighted sum of multiple normal distributions.

$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$ 

Where \\(\pi_k\\) is the mixture coefficient for each Gaussian component (\\(\sum_k \pi_k = 1\\)).

### Implementation Example 5: Clustering with GMM
    
    
    import numpy as np
    from scipy.stats import multivariate_normal
    
    class GaussianMixtureModel:
        """Gaussian Mixture Model (learning by EM algorithm)"""
    
        def __init__(self, n_components=2, max_iter=100, tol=1e-4):
            """
            Parameters:
            -----------
            n_components : int
                Number of Gaussian components
            max_iter : int
                Maximum number of iterations
            tol : float
                Convergence threshold
            """
            self.n_components = n_components
            self.max_iter = max_iter
            self.tol = tol
    
        def initialize_parameters(self, X):
            """Initialize parameters"""
            n_samples, n_features = X.shape
    
            # Randomly select data points as initial means
            random_idx = np.random.choice(n_samples, self.n_components, replace=False)
            self.means = X[random_idx]
    
            # Initialize covariance matrices as identity matrices
            self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
            # Initialize mixture coefficients uniformly
            self.weights = np.ones(self.n_components) / self.n_components
    
        def e_step(self, X):
            """
            E-step: Calculate responsibilities (which Gaussian component each data belongs to)
    
            γ(z_nk) = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
            """
            n_samples = X.shape[0]
            responsibilities = np.zeros((n_samples, self.n_components))
    
            for k in range(self.n_components):
                # Calculate likelihood for each component
                rv = multivariate_normal(self.means[k], self.covariances[k])
                responsibilities[:, k] = self.weights[k] * rv.pdf(X)
    
            # Normalize to calculate responsibilities
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    
            return responsibilities
    
        def m_step(self, X, responsibilities):
            """
            M-step: Update parameters using responsibilities
            """
            n_samples, n_features = X.shape
    
            # Effective number of samples for each component
            N_k = responsibilities.sum(axis=0)
    
            # Update parameters
            for k in range(self.n_components):
                # Update mixture coefficient
                self.weights[k] = N_k[k] / n_samples
    
                # Update mean
                self.means[k] = (responsibilities[:, k].reshape(-1, 1) * X).sum(axis=0) / N_k[k]
    
                # Update covariance matrix
                diff = X - self.means[k]
                self.covariances[k] = (responsibilities[:, k].reshape(-1, 1, 1) *
                                      (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / N_k[k]
    
                # Add small value for numerical stability
                self.covariances[k] += np.eye(n_features) * 1e-6
    
        def compute_log_likelihood(self, X):
            """Calculate log-likelihood"""
            n_samples = X.shape[0]
            log_likelihood = 0
    
            for i in range(n_samples):
                likelihood = 0
                for k in range(self.n_components):
                    rv = multivariate_normal(self.means[k], self.covariances[k])
                    likelihood += self.weights[k] * rv.pdf(X[i])
                log_likelihood += np.log(likelihood)
    
            return log_likelihood
    
        def fit(self, X):
            """Learn parameters with EM algorithm"""
            self.initialize_parameters(X)
    
            prev_log_likelihood = -np.inf
    
            for iteration in range(self.max_iter):
                # E-step
                responsibilities = self.e_step(X)
    
                # M-step
                self.m_step(X, responsibilities)
    
                # Calculate log-likelihood
                log_likelihood = self.compute_log_likelihood(X)
    
                # Convergence check
                if abs(log_likelihood - prev_log_likelihood) < self.tol:
                    print(f"Converged ({iteration+1} iterations)")
                    break
    
                prev_log_likelihood = log_likelihood
    
                if (iteration + 1) % 10 == 0:
                    print(f"Iteration {iteration+1}: log-likelihood = {log_likelihood:.4f}")
    
            return self
    
        def predict(self, X):
            """Predict cluster with highest responsibility"""
            responsibilities = self.e_step(X)
            return np.argmax(responsibilities, axis=1)
    
    # Usage example
    np.random.seed(42)
    
    # Data generated from two Gaussian distributions
    data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 150)
    data2 = np.random.multivariate_normal([4, 4], [[1, 0.5], [0.5, 1]], 150)
    X = np.vstack([data1, data2])
    
    # Train GMM
    gmm = GaussianMixtureModel(n_components=2, max_iter=50)
    gmm.fit(X)
    
    # Clustering results
    labels = gmm.predict(X)
    print(f"\nLearned parameters:")
    for k in range(gmm.n_components):
        print(f"Component {k+1}: weight={gmm.weights[k]:.3f}, mean={gmm.means[k]}")
    

### 5.2 Bayesian Linear Regression

In Bayesian linear regression, we consider a probability distribution over parameters and can also estimate prediction uncertainty.

$$P(\mathbf{w}|D) = \frac{P(D|\mathbf{w})P(\mathbf{w})}{P(D)}$$ 

The predictive distribution is obtained by marginalizing over the posterior distribution:

$$P(y^*|x^*, D) = \int P(y^*|x^*, \mathbf{w})P(\mathbf{w}|D)d\mathbf{w}$$ 

### Implementation Example 6: Bayesian Linear Regression
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class BayesianLinearRegression:
        """Implementation of Bayesian Linear Regression"""
    
        def __init__(self, alpha=1.0, beta=1.0):
            """
            Parameters:
            -----------
            alpha : float
                Precision of weight prior distribution (λ = α * I)
            beta : float
                Precision of observation noise (1/σ²)
            """
            self.alpha = alpha  # Prior precision of weights
            self.beta = beta    # Noise precision
    
        def fit(self, X, y):
            """
            Calculate posterior distribution from training data
    
            Posterior distribution: P(w|D) = N(w|m_N, S_N)
            S_N = (α*I + β*X^T*X)^(-1)
            m_N = β*S_N*X^T*y
            """
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
    
            n_samples, n_features = X.shape
    
            # Prior precision matrix
            prior_precision = self.alpha * np.eye(n_features)
    
            # Posterior covariance matrix (inverse of precision matrix)
            posterior_precision = prior_precision + self.beta * (X.T @ X)
            self.posterior_cov = np.linalg.inv(posterior_precision)
    
            # Posterior mean
            self.posterior_mean = self.beta * (self.posterior_cov @ X.T @ y)
    
            return self
    
        def predict(self, X_test, return_std=False):
            """
            Calculate predictive distribution
    
            Predictive distribution: P(y*|x*, D) = N(y*|m_N^T*x*, σ_N²(x*))
            σ_N²(x*) = 1/β + x*^T*S_N*x*
            """
            X_test = np.array(X_test)
    
            # Predictive mean
            y_pred = X_test @ self.posterior_mean
    
            if return_std:
                # Predictive variance (data noise + parameter uncertainty)
                y_var = 1/self.beta + np.sum(X_test @ self.posterior_cov * X_test, axis=1, keepdims=True)
                y_std = np.sqrt(y_var)
                return y_pred.flatten(), y_std.flatten()
            else:
                return y_pred.flatten()
    
        def sample_weights(self, n_samples=10):
            """Sample parameters from posterior distribution"""
            return np.random.multivariate_normal(
                self.posterior_mean.flatten(),
                self.posterior_cov,
                size=n_samples
            )
    
    # Usage example and visualization of Bayesian estimation
    np.random.seed(42)
    
    # True parameters: y = 2x + 1 + noise
    def true_function(x):
        return 2 * x + 1
    
    # Generate training data
    X_train = np.linspace(0, 1, 10).reshape(-1, 1)
    X_train = np.hstack([np.ones_like(X_train), X_train])  # Add bias term
    y_train = true_function(X_train[:, 1]) + np.random.randn(10) * 0.3
    
    # Train Bayesian linear regression
    bayesian_lr = BayesianLinearRegression(alpha=2.0, beta=10.0)
    bayesian_lr.fit(X_train, y_train)
    
    # Test data
    X_test = np.linspace(-0.2, 1.2, 100).reshape(-1, 1)
    X_test_with_bias = np.hstack([np.ones_like(X_test), X_test])
    
    # Prediction (mean and standard deviation)
    y_pred, y_std = bayesian_lr.predict(X_test_with_bias, return_std=True)
    
    # Sample parameters from posterior distribution
    weight_samples = bayesian_lr.sample_weights(n_samples=20)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Left plot: Predictive distribution and confidence interval
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 1], y_train, c='red', s=50, label='Training data', zorder=3)
    plt.plot(X_test, true_function(X_test), 'g--', linewidth=2, label='True function', zorder=2)
    plt.plot(X_test, y_pred, 'b-', linewidth=2, label='Predictive mean', zorder=2)
    plt.fill_between(X_test.flatten(), y_pred - 2*y_std, y_pred + 2*y_std,
                     alpha=0.3, label='95% confidence interval', zorder=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesian Linear Regression: Predictive Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right plot: Functions sampled from posterior distribution
    plt.subplot(1, 2, 2)
    plt.scatter(X_train[:, 1], y_train, c='red', s=50, label='Training data', zorder=3)
    for i, w in enumerate(weight_samples):
        y_sample = X_test_with_bias @ w
        plt.plot(X_test, y_sample, 'b-', alpha=0.3, linewidth=1,
                 label='Sample' if i == 0 else '')
    plt.plot(X_test, true_function(X_test), 'g--', linewidth=2, label='True function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Functions Sampled from Posterior Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_regression.png', dpi=150, bbox_inches='tight')
    print("Saved Bayesian linear regression visualization")
    
    # Posterior distribution of parameters
    print(f"\nPosterior distribution of parameters:")
    print(f"Mean: {bayesian_lr.posterior_mean.flatten()}")
    print(f"Standard deviation: {np.sqrt(np.diag(bayesian_lr.posterior_cov))}")
    

## Summary

In this chapter, we learned the fundamentals of probability and statistics that form the foundation of machine learning.

**What We Learned**

  * **Bayes' Theorem** : The fundamental principle for calculating posterior probability from prior knowledge and data
  * **Probability Distributions** : Properties and implementation of normal distributions and multivariate normal distributions
  * **Statistical Quantities** : Calculation and interpretation of expectation, variance, and covariance
  * **Parameter Estimation** : Differences and applications of maximum likelihood estimation and Bayesian estimation
  * **Practical Applications** : Naive Bayes, GMM, and Bayesian linear regression

**Preparation for Next Chapter** In Chapter 2, we will learn the fundamentals of linear algebra. In particular, matrix decomposition (eigenvalue decomposition, SVD) and PCA are deeply related to the covariance matrices of multivariate normal distributions learned in this chapter. 

### Exercise Problems

  1. Calculate the accuracy of a spam email detector using Bayes' theorem
  2. Generate and visualize data for a 2D normal distribution with correlation coefficients of 0.9 and -0.9
  3. Compare the differences between MLE and MAP estimation with small samples (n=5) and large samples (n=1000)
  4. Extend GMM to 3 components and verify its operation with data having three clusters
  5. Experiment with how predictions change in Bayesian linear regression when varying the prior precision parameter α

### References

  * C.M. Bishop, "Pattern Recognition and Machine Learning" (2006)
  * Kevin P. Murphy, "Machine Learning: A Probabilistic Perspective" (2012)
  * Masashi Sugiyama, "100 Mathematical Problems in Statistical Machine Learning with Python" (2020)

[← Back to Series Contents](<./index.html>) [Chapter 2: Fundamentals of Linear Algebra →](<./chapter2-linear-algebra.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
