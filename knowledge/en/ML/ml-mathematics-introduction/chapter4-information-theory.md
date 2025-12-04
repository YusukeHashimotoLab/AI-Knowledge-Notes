---
title: "Chapter 4: Information Theory"
chapter_title: "Chapter 4: Information Theory"
---

This chapter covers Information Theory. You will learn essential concepts and techniques.

**Deeply understand information theory supporting machine learning, from entropy to VAE, through theory and implementation**

**What You'll Learn in This Chapter**

  * Mathematical definition and intuitive understanding of entropy and information content
  * Relationship between KL divergence and cross-entropy
  * Theoretical foundation of feature selection using mutual information
  * Information-theoretic interpretation of VAE and information bottleneck
  * Information-theoretic meaning of loss functions in machine learning

## 1\. Entropy

### 1.1 Information Content and Shannon Entropy

The foundation of information theory begins with quantifying "the amount of information in an event." When the probability of event x occurring is P(x), its **self-information** is defined as follows.

$$I(x) = -\log_2 P(x) \quad \text{[bits]}$$ 

This definition has deep meaning:

  * **Events with lower probability have greater information content** : Rare events are more surprising
  * **Certain events (P(x)=1) have zero information content** : There is no surprise in predictable events
  * **Information content of independent events is additive** : I(x,y) = I(x) + I(y)

**Shannon entropy** represents the average information content of an entire probability distribution.

$$H(X) = -\sum_{x} P(x) \log_2 P(x) = \mathbb{E}_{x \sim P}[-\log P(x)]$$ 

**Intuitive Understanding of Entropy** Entropy is a measure of "uncertainty" or "randomness." A uniform distribution has maximum entropy (most difficult to predict), while a deterministic distribution has minimum entropy (entropy=0). 

### 1.2 Conditional Entropy

The entropy of X under the condition that variable Y is given is called **conditional entropy**.

$$H(X|Y) = \sum_{y} P(y) H(X|Y=y) = -\sum_{x,y} P(x,y) \log P(x|y)$$ 

As an important property, the **chain rule** holds:

$$H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$ 

### Implementation Example 1: Entropy Calculation and Visualization
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class InformationMeasures:
        """Class for calculating fundamental quantities in information theory"""
    
        @staticmethod
        def entropy(p, base=2):
            """
            Calculate Shannon entropy
    
            Parameters:
            -----------
            p : array-like
                Probability distribution (must sum to 1)
            base : int
                Logarithm base (2: bits, e: nats)
    
            Returns:
            --------
            float : Entropy
            """
            p = np.array(p)
            # Define 0 * log(0) = 0 (numerically stable implementation)
            p = p[p > 0]  # Consider only positive probabilities
    
            if base == 2:
                return -np.sum(p * np.log2(p))
            elif base == np.e:
                return -np.sum(p * np.log(p))
            else:
                return -np.sum(p * np.log(p)) / np.log(base)
    
        @staticmethod
        def conditional_entropy(joint_p, axis=1):
            """
            Calculate conditional entropy H(X|Y)
    
            Parameters:
            -----------
            joint_p : ndarray
                Joint probability distribution P(X,Y)
            axis : int
                Axis of the conditioning variable (0: H(Y|X), 1: H(X|Y))
    
            Returns:
            --------
            float : Conditional entropy
            """
            joint_p = np.array(joint_p)
    
            # Calculate marginal probability
            marginal_p = np.sum(joint_p, axis=axis)
    
            # Calculate conditional entropy
            h_cond = 0
            for i, p_y in enumerate(marginal_p):
                if p_y > 0:
                    if axis == 1:
                        conditional_p = joint_p[:, i] / p_y
                    else:
                        conditional_p = joint_p[i, :] / p_y
                    h_cond += p_y * InformationMeasures.entropy(conditional_p)
    
            return h_cond
    
        @staticmethod
        def joint_entropy(joint_p):
            """
            Calculate joint entropy H(X,Y)
            """
            joint_p = np.array(joint_p).flatten()
            return InformationMeasures.entropy(joint_p)
    
    # Usage Example 1: Entropy of binary variable
    print("=" * 50)
    print("Entropy of Binary Variable")
    print("=" * 50)
    
    # Coin probability distribution
    probs = np.linspace(0.01, 0.99, 99)
    entropies = [InformationMeasures.entropy([p, 1-p]) for p in probs]
    
    plt.figure(figsize=(10, 5))
    plt.plot(probs, entropies, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Maximum entropy (p=0.5)')
    plt.xlabel('Probability P(X=1)')
    plt.ylabel('Entropy H(X) [bits]')
    plt.title('Entropy of Binary Random Variable')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('entropy_binary.png', dpi=150, bbox_inches='tight')
    print(f"Maximum entropy: {max(entropies):.4f} bits (p=0.5)")
    
    # Usage Example 2: Joint probability distribution and conditional entropy
    print("\n" + "=" * 50)
    print("Example of Conditional Entropy")
    print("=" * 50)
    
    # Joint probability distribution P(X,Y)
    joint_prob = np.array([
        [0.2, 0.1],  # P(X=0, Y=0), P(X=0, Y=1)
        [0.15, 0.55] # P(X=1, Y=0), P(X=1, Y=1)
    ])
    
    # Marginal probabilities
    p_x = np.sum(joint_prob, axis=1)
    p_y = np.sum(joint_prob, axis=0)
    
    # Various entropies
    h_x = InformationMeasures.entropy(p_x)
    h_y = InformationMeasures.entropy(p_y)
    h_xy = InformationMeasures.joint_entropy(joint_prob)
    h_x_given_y = InformationMeasures.conditional_entropy(joint_prob, axis=1)
    h_y_given_x = InformationMeasures.conditional_entropy(joint_prob, axis=0)
    
    print(f"H(X) = {h_x:.4f} bits")
    print(f"H(Y) = {h_y:.4f} bits")
    print(f"H(X,Y) = {h_xy:.4f} bits")
    print(f"H(X|Y) = {h_x_given_y:.4f} bits")
    print(f"H(Y|X) = {h_y_given_x:.4f} bits")
    
    # Verify chain rule: H(X,Y) = H(X) + H(Y|X)
    print(f"\nChain rule verification:")
    print(f"H(X) + H(Y|X) = {h_x + h_y_given_x:.4f}")
    print(f"H(X,Y) = {h_xy:.4f}")
    print(f"Difference: {abs(h_xy - (h_x + h_y_given_x)):.10f}")
    

## 2\. KL Divergence and Cross-Entropy

### 2.1 KL Divergence (Kullback-Leibler Divergence)

KL divergence is a metric that measures the "difference" between two probability distributions P(x) and Q(x).

$$D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$ 

**Important properties:**

  * **Asymmetry** : \\(D_{KL}(P||Q) \neq D_{KL}(Q||P)\\) (not a distance metric)
  * **Non-negativity** : \\(D_{KL}(P||Q) \geq 0\\), equality holds when \\(P = Q\\)
  * **Gibbs' inequality** : \\(\mathbb{E}_P[\log P(x)] \geq \mathbb{E}_P[\log Q(x)]\\)

### 2.2 Cross-Entropy

Cross-entropy is the average number of bits needed to encode events under true distribution P using model distribution Q.

$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = H(P) + D_{KL}(P||Q)$$ 

From this relationship, **minimizing cross-entropy is equivalent to minimizing KL divergence** (since H(P) is constant).

**Applications in Machine Learning** Cross-entropy is widely used as a loss function in classification problems. Learning progresses by minimizing the difference between the true label distribution P (one-hot) and the model's predicted distribution Q (softmax output). 

### Implementation Example 2: KL Divergence and Cross-Entropy
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import softmax
    
    class DivergenceMeasures:
        """Calculation of divergence metrics"""
    
        @staticmethod
        def kl_divergence(p, q, epsilon=1e-10):
            """
            Calculate KL divergence D_KL(P||Q)
    
            Parameters:
            -----------
            p, q : array-like
                Probability distributions (must be normalized)
            epsilon : float
                Small value for numerical stability
    
            Returns:
            --------
            float : KL divergence [nats or bits]
            """
            p = np.array(p)
            q = np.array(q)
    
            # Prevent division by zero
            q = np.clip(q, epsilon, 1.0)
            p = np.clip(p, epsilon, 1.0)
    
            return np.sum(p * np.log(p / q))
    
        @staticmethod
        def cross_entropy(p, q, epsilon=1e-10):
            """
            Calculate cross-entropy H(P,Q)
    
            Returns:
            --------
            float : Cross-entropy
            """
            p = np.array(p)
            q = np.array(q)
    
            q = np.clip(q, epsilon, 1.0)
    
            return -np.sum(p * np.log(q))
    
        @staticmethod
        def js_divergence(p, q):
            """
            Jensen-Shannon divergence (symmetric version of KL)
    
            JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
            where M = 0.5 * (P + Q)
            """
            p = np.array(p)
            q = np.array(q)
            m = 0.5 * (p + q)
    
            return 0.5 * DivergenceMeasures.kl_divergence(p, m) + \
                   0.5 * DivergenceMeasures.kl_divergence(q, m)
    
    # Usage Example 1: KL divergence of Gaussian distributions
    print("=" * 50)
    print("KL Divergence of Gaussian Distributions")
    print("=" * 50)
    
    # Two normal distributions
    x = np.linspace(-5, 5, 1000)
    p = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)  # N(0, 1)
    q = np.exp(-0.5 * (x - 1)**2) / np.sqrt(2 * np.pi)  # N(1, 1)
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    kl_pq = DivergenceMeasures.kl_divergence(p, q)
    kl_qp = DivergenceMeasures.kl_divergence(q, p)
    js = DivergenceMeasures.js_divergence(p, q)
    
    print(f"D_KL(P||Q) = {kl_pq:.4f}")
    print(f"D_KL(Q||P) = {kl_qp:.4f}")
    print(f"JS(P||Q) = {js:.4f}")
    print(f"Asymmetry: |D_KL(P||Q) - D_KL(Q||P)| = {abs(kl_pq - kl_qp):.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(x, p * 1000, 'b-', linewidth=2, label='P: N(0,1)')
    plt.plot(x, q * 1000, 'r-', linewidth=2, label='Q: N(1,1)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'KL Divergence: D_KL(P||Q) = {kl_pq:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kl_divergence.png', dpi=150, bbox_inches='tight')
    
    # Usage Example 2: Cross-entropy loss in classification
    print("\n" + "=" * 50)
    print("Cross-Entropy Loss in Classification")
    print("=" * 50)
    
    # True label (one-hot encoding)
    true_label = np.array([0, 1, 0, 0])  # Class 1 is correct
    
    # Model predictions (logits)
    logits_good = np.array([1.0, 3.5, 0.5, 0.8])   # Good prediction
    logits_bad = np.array([2.0, 0.5, 1.5, 1.0])    # Poor prediction
    
    # Convert to probabilities with softmax
    pred_good = softmax(logits_good)
    pred_bad = softmax(logits_bad)
    
    # Cross-entropy loss
    ce_good = DivergenceMeasures.cross_entropy(true_label, pred_good)
    ce_bad = DivergenceMeasures.cross_entropy(true_label, pred_bad)
    
    print(f"Good prediction probability distribution: {pred_good}")
    print(f"Cross-entropy loss: {ce_good:.4f}\n")
    
    print(f"Poor prediction probability distribution: {pred_bad}")
    print(f"Cross-entropy loss: {ce_bad:.4f}\n")
    
    print(f"Loss difference: {ce_bad - ce_good:.4f}")
    print("→ Good prediction has lower loss")
    

## 3\. Mutual Information

### 3.1 Definition of Mutual Information

**Mutual Information** is a metric that measures the statistical dependence between two random variables X and Y.

$$I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} = D_{KL}(P(X,Y)||P(X)P(Y))$$ 

Mutual information can also be expressed using entropy as follows:

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$ 

**Important properties:**

  * **Symmetry** : \\(I(X;Y) = I(Y;X)\\)
  * **Non-negativity** : \\(I(X;Y) \geq 0\\), equality holds when X and Y are independent
  * **Information reduction** : \\(I(X;Y) \leq \min(H(X), H(Y))\\)

**Intuitive Understanding** Mutual information represents "how much information about Y can be obtained by observing X." The larger I(X;Y) is, the more strongly X and Y depend on each other. 

### 3.2 Application to Feature Selection

In machine learning, mutual information can be used to select important features. By calculating the mutual information I(X_i;Y) between target variable Y and each feature X_i, features with large values are selected.

### Implementation Example 3: Feature Selection Using Mutual Information
    
    
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.datasets import make_classification
    
    class MutualInformation:
        """Mutual information calculation and feature selection"""
    
        @staticmethod
        def mutual_information_discrete(x, y):
            """
            Calculate mutual information for discrete variables
    
            I(X;Y) = H(X) + H(Y) - H(X,Y)
    
            Parameters:
            -----------
            x, y : array-like
                Discrete-valued random variables
    
            Returns:
            --------
            float : Mutual information
            """
            x = np.array(x)
            y = np.array(y)
    
            # Create joint frequency matrix
            xy = np.c_[x, y]
            unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
            joint_prob = counts_xy / len(x)
    
            # Marginal probabilities
            unique_x, counts_x = np.unique(x, return_counts=True)
            p_x = counts_x / len(x)
    
            unique_y, counts_y = np.unique(y, return_counts=True)
            p_y = counts_y / len(y)
    
            # Calculate entropies
            from scipy.stats import entropy
            h_x = entropy(p_x, base=2)
            h_y = entropy(p_y, base=2)
            h_xy = entropy(joint_prob, base=2)
    
            # Mutual information
            mi = h_x + h_y - h_xy
    
            return mi
    
        @staticmethod
        def feature_selection_by_mi(X, y, n_features=5):
            """
            Feature selection using mutual information
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                Feature matrix
            y : array-like
                Target variable
            n_features : int
                Number of features to select
    
            Returns:
            --------
            selected_indices : array
                Indices of selected features
            mi_scores : array
                Mutual information scores for each feature
            """
            # scikit-learn's mutual information calculation
            mi_scores = mutual_info_classif(X, y, random_state=42)
    
            # Sort in descending order of scores
            selected_indices = np.argsort(mi_scores)[::-1][:n_features]
    
            return selected_indices, mi_scores
    
    # Usage Example 1: Mutual information of discrete variables
    print("=" * 50)
    print("Mutual Information of Discrete Variables")
    print("=" * 50)
    
    # Example: Weather (X) and umbrella use (Y)
    # 0: sunny/not used, 1: rainy/used
    np.random.seed(42)
    
    # Case with strong correlation
    weather = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 10)
    umbrella_corr = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1] * 10)  # Correlated with weather
    umbrella_rand = np.random.randint(0, 2, 100)  # Random
    
    mi_corr = MutualInformation.mutual_information_discrete(weather, umbrella_corr)
    mi_rand = MutualInformation.mutual_information_discrete(weather, umbrella_rand)
    
    print(f"Mutual information of weather and umbrella (correlated): {mi_corr:.4f} bits")
    print(f"Mutual information of weather and umbrella (random): {mi_rand:.4f} bits")
    print(f"→ Mutual information is larger in the correlated case")
    
    # Usage Example 2: Feature selection
    print("\n" + "=" * 50)
    print("Feature Selection Using Mutual Information")
    print("=" * 50)
    
    # Generate synthetic data (20 features, 5 of which are useful)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,  # Useful features
        n_redundant=5,    # Redundant features
        n_repeated=0,
        n_classes=2,
        random_state=42
    )
    
    # Feature selection with mutual information
    selected_idx, mi_scores = MutualInformation.feature_selection_by_mi(X, y, n_features=5)
    
    print(f"Total number of features: {X.shape[1]}")
    print(f"\nMutual information scores (top 10 features):")
    for i in range(10):
        print(f"  Feature {i}: MI = {mi_scores[i]:.4f}")
    
    print(f"\nSelected features (top 5): {selected_idx}")
    print(f"MI scores of selected features: {mi_scores[selected_idx]}")
    
    # Visualization
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(mi_scores)), mi_scores, color='steelblue')
    plt.bar(selected_idx, mi_scores[selected_idx], color='crimson', label='Selected features')
    plt.xlabel('Feature Index')
    plt.ylabel('Mutual Information Score')
    plt.title('Mutual Information of Each Feature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_idx = np.argsort(mi_scores)[::-1]
    plt.plot(range(1, len(mi_scores)+1), mi_scores[sorted_idx], 'o-', linewidth=2)
    plt.axvline(x=5, color='r', linestyle='--', label='Number of selections=5')
    plt.xlabel('Rank')
    plt.ylabel('Mutual Information Score')
    plt.title('Rank of Mutual Information Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mutual_information.png', dpi=150, bbox_inches='tight')
    print("\nVisualization of mutual information saved")
    

## 4\. Information Theory and Machine Learning

### 4.1 Variational Autoencoder (VAE)

VAE can be understood from an information-theoretic perspective. When learning the relationship between latent variable z and data x, the following objective function is maximized.

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x)||p(z))$$ 

The right-hand side is called **ELBO (Evidence Lower BOund)** and consists of:

  * **First term (reconstruction term)** : Data reconstruction quality
  * **Second term (KL term)** : Closeness between latent distribution and prior distribution

### 4.2 Information Bottleneck Theory

Information bottleneck theory formulates representation learning in information-theoretic terms. For input X and label Y, representation Z should satisfy:

$$\min_{Z} I(X;Z) - \beta I(Z;Y)$$ 

This expresses the tradeoff of "compressing input information (minimizing I(X;Z)) while retaining label information (maximizing I(Z;Y))."

### Implementation Example 4: ELBO Calculation in VAE
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    
    class VAE(nn.Module):
        """Variational Autoencoder"""
    
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            """
            Parameters:
            -----------
            input_dim : int
                Input dimension (e.g., 28x28=784 for MNIST)
            hidden_dim : int
                Hidden layer dimension
            latent_dim : int
                Latent variable dimension
            """
            super(VAE, self).__init__()
    
            # Encoder q(z|x)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
            # Decoder p(x|z)
            self.fc3 = nn.Linear(latent_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, input_dim)
    
        def encode(self, x):
            """
            Encoder: x → (μ, log σ²)
    
            Returns:
            --------
            mu, logvar : Parameters of latent distribution
            """
            h = F.relu(self.fc1(x))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            """
            Reparameterization trick: z = μ + σ * ε, ε ~ N(0,1)
    
            This makes stochastic sampling differentiable
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
    
        def decode(self, z):
            """
            Decoder: z → x̂
            """
            h = F.relu(self.fc3(z))
            x_recon = torch.sigmoid(self.fc4(h))
            return x_recon
    
        def forward(self, x):
            """
            Forward pass: x → z → x̂
            """
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar
    
    def vae_loss(x, x_recon, mu, logvar, beta=1.0):
        """
        VAE loss function (negative ELBO)
    
        ELBO = E[log p(x|z)] - β * D_KL(q(z|x)||p(z))
    
        Parameters:
        -----------
        x : Tensor
            Original data
        x_recon : Tensor
            Reconstructed data
        mu, logvar : Tensor
            Parameters of latent distribution
        beta : float
            Weight of KL term (β-VAE)
    
        Returns:
        --------
        loss, recon_loss, kl_loss : Total loss, reconstruction loss, KL loss
        """
        # Reconstruction loss (negative log likelihood)
        # For binary data: BCE loss
        recon_loss = F.binary_cross_entropy(
            x_recon, x.view(-1, 784), reduction='sum'
        )
    
        # KL divergence loss
        # D_KL(N(μ,σ²)||N(0,1)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        # Total loss (negative ELBO)
        total_loss = recon_loss + beta * kl_loss
    
        return total_loss, recon_loss, kl_loss
    
    # Usage example
    print("=" * 50)
    print("ELBO Calculation in VAE")
    print("=" * 50)
    
    # Model initialization
    vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    
    # Dummy data (batch size 32, 28x28 images)
    torch.manual_seed(42)
    x = torch.rand(32, 1, 28, 28)
    
    # Forward pass
    x_recon, mu, logvar = vae(x)
    
    # Loss calculation
    loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, beta=1.0)
    
    print(f"Total loss (-ELBO): {loss.item():.2f}")
    print(f"  Reconstruction loss: {recon_loss.item():.2f}")
    print(f"  KL loss: {kl_loss.item():.2f}")
    
    # Check the effect of β
    print("\n" + "=" * 50)
    print("β-VAE: Effect of β Parameter")
    print("=" * 50)
    
    betas = [0.5, 1.0, 2.0, 5.0]
    for beta in betas:
        loss, recon, kl = vae_loss(x, x_recon, mu, logvar, beta=beta)
        print(f"β={beta:.1f}: Total loss={loss.item():.2f}, "
              f"Reconstruction={recon.item():.2f}, KL={kl.item():.2f}")
    
    print("\nInterpretation:")
    print("- Large β: Emphasizes KL term → Latent space approaches normal distribution")
    print("- Small β: Emphasizes reconstruction → Reconstruction quality improves")
    

## 5\. Practical Applications

### 5.1 Cross-Entropy Loss Function

The most common loss function in classification problems is cross-entropy loss.

$$\mathcal{L}_{CE} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}$$ 

Here, y_{i,c} is the true label (one-hot), and \\(\hat{y}_{i,c}\\) is the model's predicted probability.

### 5.2 KL Loss and Label Smoothing

Label smoothing is a regularization technique that prevents overconfidence. It transforms hard labels [0,1,0] to [ε/K, 1-ε+ε/K, ε/K].

### Implementation Example 5: Cross-Entropy and KL Loss
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LossFunctions:
        """Loss functions based on information theory"""
    
        @staticmethod
        def cross_entropy_loss(logits, targets):
            """
            Cross-entropy loss
    
            Parameters:
            -----------
            logits : Tensor of shape (batch_size, num_classes)
                Model output (before softmax)
            targets : Tensor of shape (batch_size,)
                True class labels
    
            Returns:
            --------
            loss : Tensor
                Cross-entropy loss
            """
            return F.cross_entropy(logits, targets)
    
        @staticmethod
        def kl_div_loss(logits, target_dist):
            """
            KL divergence loss
    
            Calculate D_KL(target || pred)
    
            Parameters:
            -----------
            logits : Tensor
                Model output
            target_dist : Tensor
                Target distribution (probability distribution)
    
            Returns:
            --------
            loss : Tensor
                KL loss
            """
            log_pred = F.log_softmax(logits, dim=-1)
            return F.kl_div(log_pred, target_dist, reduction='batchmean')
    
        @staticmethod
        def label_smoothing_loss(logits, targets, smoothing=0.1):
            """
            Cross-entropy with label smoothing
    
            Parameters:
            -----------
            smoothing : float
                Smoothing parameter (0: none, 1: completely uniform)
            """
            n_classes = logits.size(-1)
            log_pred = F.log_softmax(logits, dim=-1)
    
            # Label smoothing
            # True class: 1 - ε + ε/K
            # Other classes: ε/K
            with torch.no_grad():
                true_dist = torch.zeros_like(log_pred)
                true_dist.fill_(smoothing / (n_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    
            return torch.mean(torch.sum(-true_dist * log_pred, dim=-1))
    
    # Usage Example 1: Comparison of loss functions
    print("=" * 50)
    print("Comparison of Loss Functions")
    print("=" * 50)
    
    torch.manual_seed(42)
    
    # Data preparation
    batch_size = 4
    num_classes = 3
    
    logits = torch.randn(batch_size, num_classes) * 2
    targets = torch.tensor([0, 1, 2, 1])
    
    # Calculate each loss
    ce_loss = LossFunctions.cross_entropy_loss(logits, targets)
    
    # Label smoothing loss
    ls_loss_01 = LossFunctions.label_smoothing_loss(logits, targets, smoothing=0.1)
    ls_loss_03 = LossFunctions.label_smoothing_loss(logits, targets, smoothing=0.3)
    
    print(f"Cross-entropy loss: {ce_loss.item():.4f}")
    print(f"Label smoothing loss (ε=0.1): {ls_loss_01.item():.4f}")
    print(f"Label smoothing loss (ε=0.3): {ls_loss_03.item():.4f}")
    
    # Display prediction probabilities
    probs = F.softmax(logits, dim=-1)
    print(f"\nPrediction probabilities:")
    for i in range(batch_size):
        print(f"  Sample {i} (true label={targets[i]}): {probs[i].numpy()}")
    
    # Usage Example 2: Relationship between confidence and loss
    print("\n" + "=" * 50)
    print("Relationship Between Model Confidence and Loss")
    print("=" * 50)
    
    # Vary confidence
    confidences = np.linspace(0.1, 0.99, 50)
    losses = []
    
    for conf in confidences:
        # Create logits with probability conf for the correct class
        # [conf, (1-conf)/2, (1-conf)/2]
        logits_conf = torch.tensor([[
            np.log(conf),
            np.log((1-conf)/2),
            np.log((1-conf)/2)
        ]])
        target = torch.tensor([0])
    
        loss = LossFunctions.cross_entropy_loss(logits_conf, target)
        losses.append(loss.item())
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(confidences, losses, 'b-', linewidth=2)
    plt.xlabel('Predicted Probability for Correct Class')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Relationship Between Prediction Confidence and Loss')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('confidence_loss.png', dpi=150, bbox_inches='tight')
    print("Visualized the relationship between confidence and loss")
    
    print(f"\nObservations:")
    print(f"- Loss at confidence 0.5: {losses[20]:.4f}")
    print(f"- Loss at confidence 0.9: {losses[40]:.4f}")
    print(f"- Loss at confidence 0.99: {losses[-1]:.4f}")
    print("→ Loss decreases as confidence increases (exponential decrease)")
    

### 5.3 ELBO (Evidence Lower Bound)

ELBO, which is important in learning generative models, provides a lower bound on the log marginal likelihood.

$$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x)||p(z))$$ 

### Implementation Example 6: Detailed ELBO Calculation
    
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    class ELBOAnalysis:
        """Detailed analysis of ELBO"""
    
        @staticmethod
        def compute_elbo_components(x, x_recon, mu, logvar, n_samples=1000):
            """
            Calculate each component of ELBO in detail
    
            Parameters:
            -----------
            x : Tensor
                Original data
            x_recon : Tensor
                Reconstructed data
            mu, logvar : Tensor
                Encoder output (parameters of latent distribution)
            n_samples : int
                Number of Monte Carlo samples
    
            Returns:
            --------
            dict : Components of ELBO
            """
            batch_size = x.size(0)
            latent_dim = mu.size(1)
    
            # 1. Reconstruction term: E_q[log p(x|z)]
            # Log likelihood for binary data
            recon_term = -F.binary_cross_entropy(
                x_recon, x.view(batch_size, -1), reduction='sum'
            ) / batch_size
    
            # 2. KL term (analytical calculation): D_KL(q(z|x)||p(z))
            # q(z|x) = N(μ, σ²), p(z) = N(0, I)
            kl_term = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / batch_size
    
            # 3. ELBO calculation
            elbo = recon_term - kl_term
    
            # 4. Verification by Monte Carlo estimation
            # Sample estimation of E_q[log p(x|z)]
            std = torch.exp(0.5 * logvar)
            recon_mc = 0
            for _ in range(n_samples):
                eps = torch.randn_like(std)
                z = mu + eps * std
                # Simplified reconstruction likelihood
                recon_mc += -F.binary_cross_entropy(
                    x_recon, x.view(batch_size, -1), reduction='sum'
                )
            recon_mc = recon_mc / (n_samples * batch_size)
    
            return {
                'elbo': elbo.item(),
                'reconstruction': recon_term.item(),
                'kl_divergence': kl_term.item(),
                'reconstruction_mc': recon_mc.item(),
                'log_marginal_lower_bound': elbo.item()
            }
    
        @staticmethod
        def analyze_latent_distribution(mu, logvar):
            """
            Analyze statistics of latent distribution
    
            Returns:
            --------
            dict : Statistics
            """
            std = torch.exp(0.5 * logvar)
    
            return {
                'mean_mu': mu.mean().item(),
                'std_mu': mu.std().item(),
                'mean_sigma': std.mean().item(),
                'std_sigma': std.std().item(),
                'min_sigma': std.min().item(),
                'max_sigma': std.max().item()
            }
    
    # Usage example
    print("=" * 50)
    print("Detailed Analysis of ELBO")
    print("=" * 50)
    
    # Dummy VAE model output
    torch.manual_seed(42)
    batch_size = 16
    input_dim = 784
    latent_dim = 20
    
    x = torch.rand(batch_size, 1, 28, 28)
    mu = torch.randn(batch_size, latent_dim) * 0.5
    logvar = torch.randn(batch_size, latent_dim) * 0.5
    
    # Reparameterization
    std = torch.exp(0.5 * logvar)
    z = mu + torch.randn_like(std) * std
    x_recon = torch.sigmoid(torch.randn(batch_size, input_dim))
    
    # ELBO calculation
    elbo_components = ELBOAnalysis.compute_elbo_components(
        x, x_recon, mu, logvar, n_samples=100
    )
    
    print("ELBO components:")
    for key, value in elbo_components.items():
        print(f"  {key}: {value:.4f}")
    
    # Analysis of latent distribution
    latent_stats = ELBOAnalysis.analyze_latent_distribution(mu, logvar)
    
    print("\nStatistics of latent distribution:")
    for key, value in latent_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualization: μ and σ for each latent dimension
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution of mean μ
    axes[0].hist(mu.detach().numpy().flatten(), bins=30, alpha=0.7, color='blue')
    axes[0].set_xlabel('μ')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Latent Variable Mean (μ)')
    axes[0].axvline(x=0, color='r', linestyle='--', label='Prior mean')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of standard deviation σ
    axes[1].hist(std.detach().numpy().flatten(), bins=30, alpha=0.7, color='green')
    axes[1].set_xlabel('σ')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Latent Variable Standard Deviation (σ)')
    axes[1].axvline(x=1, color='r', linestyle='--', label='Prior standard deviation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elbo_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization of ELBO analysis")
    
    print("\n" + "=" * 50)
    print("Information-Theoretic Interpretation")
    print("=" * 50)
    print("ELBO = Reconstruction term - KL term")
    print("  Reconstruction term: Ability to explain data with latent variables")
    print("  KL term: How far the latent distribution deviates from the prior")
    print("  → Tradeoff: Good reconstruction vs regularized latent space")
    

## Summary

In this chapter, we learned the fundamentals of information theory that support machine learning.

**What We Learned**

  * **Entropy** : Quantification of uncertainty and conditional entropy
  * **KL Divergence** : Asymmetric metric measuring the difference between probability distributions
  * **Cross-Entropy** : Theoretical foundation of loss functions in classification problems
  * **Mutual Information** : Dependence between variables and application to feature selection
  * **VAE and ELBO** : Information-theoretic interpretation of generative models

**Preparation for Next Chapter** In Chapter 5, we will learn about learning theory in machine learning. KL divergence and mutual information learned in this chapter play important roles in understanding generalization error and model complexity. 

### Exercises

  1. Calculate and compare the entropy of a die (6 sides) and a biased coin (P(heads)=0.8)
  2. Analytically calculate the KL divergence between two Gaussian distributions N(0,1) and N(2,2)
  3. Explain what kind of relationship exists between X and Y when mutual information I(X;Y)=0
  4. Experimentally verify the difference in latent space when β=0.5, 1.0, 2.0 in β-VAE
  5. Explain from an information-theoretic perspective why label smoothing prevents model overconfidence

### References

  * Claude E. Shannon, "A Mathematical Theory of Communication" (1948)
  * Thomas M. Cover and Joy A. Thomas, "Elements of Information Theory" (2006)
  * D.P. Kingma and M. Welling, "Auto-Encoding Variational Bayes" (2013)
  * Naftali Tishby et al., "The Information Bottleneck Method" (2000)

[← Chapter 3: Optimization Theory](<./chapter3-optimization.html>) [Chapter 5: Learning Theory →](<./chapter5-learning-theory.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
