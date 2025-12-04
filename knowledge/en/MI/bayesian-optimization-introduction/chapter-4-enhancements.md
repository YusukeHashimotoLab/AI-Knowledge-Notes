---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25 minutes
difficulty: Beginner
code_examples: 0
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 4 Quality Enhancements

This chapter covers Chapter 4 Quality Enhancements. You will learn essential concepts and techniques.

This file contains enhancements to be integrated into chapter-4.md

## Code Reproducibility Section (add after section 4.1)

### Ensuring Code Reproducibility

**Environment Setup** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # Chapter 4: Active Learning Strategies
    # Required Library Versions
    """
    Python: 3.8+
    numpy: 1.21.0
    scikit-learn: 1.0.0
    scipy: 1.7.0
    matplotlib: 3.5.0
    """
    
    import numpy as np
    import random
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    
    # Ensure reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Recommended kernel configuration (for Active Learning)
    kernel_default = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(length_scale=0.2, length_scale_bounds=(1e-2, 1e0), nu=2.5)
    
    print("Environment setup complete (for Active Learning)")
    

* * *

## Practical Pitfalls Section (add after section 4.2)

### 4.3 Practical Pitfalls and Solutions

#### Pitfall 1: Bias in Uncertainty Sampling

**Problem** : Uncertainty sampling becomes too concentrated at the edges of the search space

**Symptoms** : \- Sampling concentrated near boundaries \- Insufficient information in interior regions \- Uneven prediction accuracy

**Solution** : Combination with epsilon-greedy method
    
    
    def epsilon_greedy_uncertainty_sampling(gp, X_candidate, epsilon=0.1):
        """
        Uncertainty sampling with epsilon-greedy strategy
    
        Parameters:
        -----------
        gp : GaussianProcessRegressor
            Trained GP model
        X_candidate : array (n_candidates, n_features)
            Candidate points
        epsilon : float
            Probability of random search (0~1)
    
        Returns:
        --------
        next_x : array
            Next sampling point
        """
        if np.random.rand() < epsilon:
            # Random sampling with epsilon probability
            next_idx = np.random.randint(len(X_candidate))
            print(f"  random search (ε={epsilon})")
        else:
            # Uncertainty sampling with (1-epsilon) probability
            _, sigma = gp.predict(X_candidate, return_std=True)
            next_idx = np.argmax(sigma)
            print(f"  uncertainty sampling (σ={sigma[next_idx]:.4f})")
    
        next_x = X_candidate[next_idx]
        return next_x, next_idx
    
    # Usage example
    np.random.seed(42)
    X_train = np.array([[0.1], [0.5], [0.9]])
    y_train = np.sin(5 * X_train).ravel()
    
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.15)
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X_train, y_train)
    
    X_candidate = np.linspace(0, 1, 100).reshape(-1, 1)
    
    # Epsilon-greedy uncertainty sampling
    for i in range(5):
        print(f"\nIteration {i+1}:")
        next_x, idx = epsilon_greedy_uncertainty_sampling(
            gp, X_candidate, epsilon=0.2  # 20% random
        )
        print(f"  Selected point: x={next_x[0]:.3f}")
    

* * *

#### Pitfall 2: Computational Cost of Diversity Sampling

**Problem** : Distance calculations are slow with large-scale data

**Symptoms** : \- Time-consuming sampling \- High memory usage \- Does not scale

**Solution** : Approximation using k-means clustering
    
    
    from sklearn.cluster import KMeans
    
    def fast_diversity_sampling(X_sampled, X_candidate, n_clusters=10):
        """
        Fast diversity sampling using k-means clustering
    
        Parameters:
        -----------
        X_sampled : array (n_sampled, n_features)
            Existing samples
        X_candidate : array (n_candidates, n_features)
            Candidate points
        n_clusters : int
            Number of clusters
    
        Returns:
        --------
        next_x : array
            Next sampling point
        """
        # Cluster candidate points
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_candidate)
    
        # Select the farthest candidate point from each cluster center
        cluster_centers = kmeans.cluster_centers_
        distances_from_sampled = np.min(
            np.linalg.norm(
                cluster_centers[:, np.newaxis, :] -
                X_sampled[np.newaxis, :, :],
                axis=2
            ),
            axis=1
        )
    
        # Select the representative point of the farthest cluster
        farthest_cluster = np.argmax(distances_from_sampled)
        cluster_mask = (kmeans.labels_ == farthest_cluster)
        candidates_in_cluster = X_candidate[cluster_mask]
    
        # Select the point closest to the cluster center within the cluster
        distances_to_center = np.linalg.norm(
            candidates_in_cluster - cluster_centers[farthest_cluster],
            axis=1
        )
        next_idx_in_cluster = np.argmin(distances_to_center)
        next_x = candidates_in_cluster[next_idx_in_cluster]
    
        return next_x
    
    # Benchmark
    import time
    
    n_sampled = 100
    n_candidates = 10000
    X_sampled = np.random.rand(n_sampled, 4)
    X_candidate = np.random.rand(n_candidates, 4)
    
    # Traditional method (full distance calculation)
    start = time.time()
    from scipy.spatial.distance import cdist
    distances = cdist(X_candidate, X_sampled)
    min_distances = np.min(distances, axis=1)
    next_idx_naive = np.argmax(min_distances)
    time_naive = time.time() - start
    
    # k-means approximation method
    start = time.time()
    next_x_fast = fast_diversity_sampling(X_sampled, X_candidate, n_clusters=20)
    time_fast = time.time() - start
    
    print(f"Traditional method: {time_naive:.4f} seconds")
    print(f"k-means method: {time_fast:.4f} seconds")
    print(f"Speedup ratio: {time_naive/time_fast:.1f}x")
    

* * *

#### Pitfall 3: Handling Experimental Failures in Closed-Loop Systems

**Problem** : Experimental failures are not considered

**Symptoms** : \- Loop stops due to experimental failures \- Cannot exploit failure data \- Low robustness

**Solution** : Active Learning considering failures
    
    
    class RobustClosedLoopOptimizer:
        """
        Closed-loop optimization handling experimental failures
        """
    
        def __init__(self, objective_function, total_budget=50, failure_rate=0.1):
            """
            Parameters:
            -----------
            objective_function : callable
                Objective function (experiment simulator)
            total_budget : int
                Total experiment budget
            failure_rate : float
                Experimental failure rate (0~1)
            """
            self.objective_function = objective_function
            self.total_budget = total_budget
            self.failure_rate = failure_rate
    
            self.X_sampled = []
            self.y_observed = []
            self.failures = []
    
        def execute_experiment(self, x):
            """
            Execute experiment (with possibility of failure)
    
            Returns:
            --------
            success : bool
                Experiment success flag
            result : float or None
                Measured value on success, None on failure
            """
            # Failure simulation
            if np.random.rand() < self.failure_rate:
                print(f"  Experiment failed: x={x}")
                return False, None
    
            # Evaluate objective function on success
            y = self.objective_function(x)
            return True, y
    
        def run(self):
            """Execute closed-loop optimization"""
            # Initialization
            X_init = np.random.uniform(0, 1, (5, 1))
            for x in X_init:
                success, y = self.execute_experiment(x)
                if success:
                    self.X_sampled.append(x)
                    self.y_observed.append(y)
                    self.failures.append(False)
                else:
                    self.failures.append(True)
    
            # Main loop
            experiments_done = len(X_init)
    
            while len(self.y_observed) < self.total_budget:
                if experiments_done >= self.total_budget * 1.5:
                    print("Experiment budget exceeded (many failures)")
                    break
    
                # Train GP model
                if len(self.y_observed) < 3:
                    # Random sampling when data is insufficient
                    next_x = np.random.uniform(0, 1, (1, 1))
                    print(f"Insufficient data: random sampling")
                else:
                    kernel = ConstantKernel(1.0) * RBF(length_scale=0.15)
                    gp = GaussianProcessRegressor(kernel=kernel)
                    X_array = np.array(self.X_sampled)
                    y_array = np.array(self.y_observed)
                    gp.fit(X_array, y_array)
    
                    # Maximize EI
                    X_candidate = np.linspace(0, 1, 500).reshape(-1, 1)
                    mu, sigma = gp.predict(X_candidate, return_std=True)
                    f_best = np.max(y_array)
    
                    from scipy.stats import norm
                    improvement = mu - f_best - 0.01
                    Z = improvement / (sigma + 1e-9)
                    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
                    next_idx = np.argmax(ei)
                    next_x = X_candidate[next_idx:next_idx+1]
    
                # Execute experiment
                success, y = self.execute_experiment(next_x)
                experiments_done += 1
    
                if success:
                    self.X_sampled.append(next_x)
                    self.y_observed.append(y)
                    self.failures.append(False)
                    print(f"Success {len(self.y_observed)}/{self.total_budget}: "
                          f"x={next_x[0][0]:.3f}, y={y:.3f}")
                else:
                    self.failures.append(True)
                    print(f"Failed: Retrying")
    
            # Results summary
            success_rate = len(self.y_observed) / experiments_done
            print(f"\nFinal results:")
            print(f"  Total experiments: {experiments_done}")
            print(f"  Successful experiments: {len(self.y_observed)}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Best value: {np.max(self.y_observed):.4f}")
    
    # Usage example
    def noisy_objective(x):
        """Noisy objective function"""
        return np.sin(5 * x[0]) * np.exp(-x[0]) + 0.1 * np.random.randn()
    
    np.random.seed(42)
    optimizer = RobustClosedLoopOptimizer(
        objective_function=noisy_objective,
        total_budget=20,
        failure_rate=0.2  # 20% failure rate
    )
    optimizer.run()
    

* * *

## End-of-Chapter Checklist (add before "Exercises")

### 4.7 End-of-Chapter Checklist

#### ✅ Understanding Active Learning

  * [ ] Can explain the difference between Active Learning and Bayesian Optimization
  * [ ] Understand the three main strategies (uncertainty, diversity, model change)
  * [ ] Can explain the advantages and disadvantages of each strategy
  * [ ] Can select strategies according to the problem
  * [ ] Know how to combine strategies

**Selection Guide** :
    
    
    Understanding search space            → Diversity sampling
    Improving prediction accuracy         → Uncertainty sampling
    Improving model generalization        → Expected model change
    Finding optimal solutions             → Bayesian Optimization (EI/UCB)
    Discovering diverse candidate materials → Combination of diversity + uncertainty
    

* * *

#### ✅ Uncertainty Sampling

  * [ ] Understand the meaning of prediction standard deviation σ
  * [ ] Know how to identify regions with high uncertainty
  * [ ] Can implement combination with epsilon-greedy method
  * [ ] Understand application to classification problems (margin, entropy)
  * [ ] Know the limitations of uncertainty sampling

**Implementation Check** :
    
    
    # Can you complete this code?
    def uncertainty_sampling(gp, X_candidate):
        """
        Select the point with maximum uncertainty
    
        Returns:
        --------
        next_x : array
            Next sampling point
        uncertainty : float
            Uncertainty at that point
        """
        # Your implementation
        _, sigma = gp.predict(X_candidate, return_std=True)
        next_idx = np.argmax(sigma)
        next_x = X_candidate[next_idx]
        uncertainty = sigma[next_idx]
    
        return next_x, uncertainty
    
    # Correct!
    

* * *

#### ✅ Diversity Sampling

  * [ ] Understand the concept of MaxMin distance
  * [ ] Can implement approximation using k-means clustering
  * [ ] Know the basics of Determinantal Point Process (DPP)
  * [ ] Can evaluate search space coverage
  * [ ] Know speedup methods for large-scale data

**Diversity Evaluation Metrics** :
    
    
    def evaluate_diversity(X_sampled, bounds):
        """
        Evaluate sampling diversity
    
        Returns:
        --------
        coverage_score : float
            Search space coverage (0~1)
        """
        # Divide search space into 10 parts and calculate coverage
        n_dims = X_sampled.shape[1]
        n_bins = 10
    
        coverage_count = 0
        total_bins = n_bins ** n_dims
    
        # Simplified version: coverage per dimension
        for dim in range(n_dims):
            hist, _ = np.histogram(
                X_sampled[:, dim],
                bins=n_bins,
                range=(bounds[dim, 0], bounds[dim, 1])
            )
            coverage_count += np.sum(hist > 0)
    
        coverage_score = coverage_count / (n_bins * n_dims)
        return coverage_score
    
    # Usage example
    bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
    X_sampled = np.random.rand(20, 4)
    coverage = evaluate_diversity(X_sampled, bounds)
    print(f"Coverage: {coverage:.1%}")
    

* * *

#### ✅ Closed-Loop Optimization

  * [ ] Understand the components of a closed-loop system
  * [ ] Know how to integrate AI engine, experimental equipment, and data management
  * [ ] Can implement methods for handling experimental failures
  * [ ] Can design real-time monitoring
  * [ ] Understand the role of human researchers

**System Design Checklist** :
    
    
    □ Definition of objective function and evaluation method
    □ Explicit constraints
    □ Initial sampling strategy
    □ Selection of acquisition function
    □ Determination of batch size
    □ Retry logic for experimental failures
    □ Anomaly detection and human notification
    □ Automatic data saving and backup
    □ Progress visualization
    □ Setting termination conditions
    

* * *

#### ✅ Understanding Real-World Applications

  * [ ] Can explain the achievements of Berkeley A-Lab
  * [ ] Understand the approach of RoboRXN
  * [ ] Know the features of Materials Acceleration Platform
  * [ ] Can evaluate ROI of industrial applications
  * [ ] Can analyze success factors and challenges

**ROI Calculation Template** :
    
    
    Traditional method:
      Number of experiments: ________ times
      Experiment time: ________ hours/time
      Labor cost: ________ $/hour
      Total cost: ________ $
      Development period: ________ months
    
    AI-driven method (closed-loop):
      Number of experiments: ________ times (__% reduction)
      Experiment time: ________ hours/time (automated)
      Labor cost: ________ $/hour (monitoring only)
      System construction: ________ $ (initial investment)
      Total cost: ________ $
      Development period: ________ months (__% reduction)
    
    Payback period: ________ months
    

* * *

#### ✅ Human-AI Collaboration

  * [ ] Understand human intuition and AI strengths
  * [ ] Can design hybrid approaches
  * [ ] Can determine when humans should intervene
  * [ ] Can build decision support systems
  * [ ] Can design feedback loops

**Collaboration Protocol** :
    
    
    Phase 1: Problem formulation (human-led)
      → Define objective function, constraints, and search space
      → AI checks feasibility
    
    Phase 2: Initial exploration (AI-led)
      → AI explores data-efficiently
      → Human validates anomalies
    
    Phase 3: Refinement (hybrid)
      → AI proposes
      → Human evaluates physical validity
      → Collaborative decision-making
    
    Phase 4: Implementation (human-led)
      → Human selects final candidates
      → AI quantifies uncertainty
    

* * *

### ✅ Understanding Career Paths

  * [ ] Understand the academic researcher path
  * [ ] Know the industry R&D engineer path
  * [ ] Can consider the autonomous experimentation specialist path
  * [ ] Can identify skills to learn next
  * [ ] Have clarified your own career goals

**Next Steps Selection Guide** :
    
    
    Theory research orientation
    → GNN Beginner + Reinforcement Learning Beginner
    → Paper writing, conference presentations
    
    Implementation/application orientation
    → Robotics Experiment Automation Beginner
    → Original projects, portfolio creation
    
    Industrial application orientation
    → Deep dive into industrial case studies
    → Internships, practical experience
    
    System building orientation
    → Closed-loop system construction
    → API design, hardware integration
    

* * *

### Pass Criteria

If you have achieved the following, you have completed the series:

  1. **Theoretical Understanding** : Clear 80% or more of each checklist item
  2. **Implementation Skills** : Can solve all exercises
  3. **Application Ability** : Can formulate new materials exploration problems
  4. **Career** : Next steps are clear

**Final Confirmation Questions** : 1\. Can you implement and compare the performance of three Active Learning strategies? 2\. Can you design a closed-loop optimization system? 3\. Can you extract learnings from real-world application success stories? 4\. Can you explain the next steps toward your career goals?

If all are YES, congratulations! You have completed the Bayesian Optimization & Active Learning Beginner series!

**To the Next Series** : \- Robotics Experiment Automation Beginner \- Reinforcement Learning Beginner (Materials Science Specialized) \- GNN Beginner

**Continuous Learning** : \- Paper reading (1 per week) \- Open source contributions \- Community participation \- Application to real projects

We wish you success!
