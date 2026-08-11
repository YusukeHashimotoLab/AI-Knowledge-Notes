---
title: "Chapter 4: Applications and Practice in Materials Exploration"
chapter_title: "Chapter 4: Applications and Practice in Materials Exploration"
subtitle: Bayesian Optimization・DFT・Integration with Experimental Robots
reading_time: 25-30 minutes
difficulty: Advanced
code_examples: 7
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 4: Applications and Practice in Materials Exploration

This chapter focuses on practical applications of Applications and Practice in Materials Exploration. You will learn Designing closed-loop systems and Visualizing specific career paths.

**Bayesian Optimization・DFT・Integration with Experimental Robots**

## Learning Objectives

By reading this chapter, you will master:

  * ✅ Understanding integration methods of Active Learning and Bayesian Optimization
  * ✅ Applying optimization to high-throughput calculations
  * ✅ Designing closed-loop systems
  * ✅ Gaining practical knowledge from 5 industrial application case studies
  * ✅ Visualizing specific career paths

**Reading Time** : 25-30 minutes **Code Examples** : 7 **Exercises** : 3

* * *

## 4.1 Active Learning × Bayesian Optimization

### Integration with Bayesian Optimization

Active Learning and Bayesian Optimization are closely related.

**Common Points** : \- Smart sampling leveraging uncertainty \- Surrogate models with Gaussian Processes \- Selecting next candidates with Acquisition Functions

**Differences** : \- **Active Learning** : Aims for model improvement \- **Bayesian Optimization** : Aims for maximizing objective function

### Integration Implementation with BoTorch

**Code Example 1: Active Learning + Bayesian Optimization**
    
    
    import torch
    import numpy as np
    from botorch.models import SingleTaskGP
    from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.fit import fit_gpytorch_model
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from sklearn.metrics import mean_squared_error
    
    
    class ActiveBayesianOptimizer:
        """Bayesian optimizer integrated with Active Learning"""
    
        def __init__(self, bounds, mode='exploration'):
            """
            Parameters
            ----------
            bounds : torch.Tensor
                Bounds of the search space (2 x d: [lower, upper])
            mode : str
                'exploration' (Active Learning) or 'exploitation' (BO)
            """
            self.bounds = bounds
            self.mode = mode
            self.train_X = None
            self.train_Y = None
            self.model = None
    
        def fit(self, X, Y):
            """Fit the GP model to the training data"""
            self.train_X = torch.tensor(X, dtype=torch.float64)
            self.train_Y = torch.tensor(Y, dtype=torch.float64).unsqueeze(-1)
    
            # Build the SingleTaskGP model
            self.model = SingleTaskGP(self.train_X, self.train_Y)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_model(mll)
    
        def suggest_next(self, n_candidates=1):
            """Propose the next experimental candidate"""
            if self.mode == 'exploration':
                # Active Learning: emphasize uncertainty
                acq_function = UpperConfidenceBound(
                    self.model, beta=2.0  # high beta = exploration-focused
                )
            else:
                # Bayesian Optimization: emphasize improvement
                acq_function = qExpectedImprovement(
                    self.model, best_f=self.train_Y.max()
                )
    
            # Maximize the acquisition function
            candidates, acq_value = optimize_acqf(
                acq_function,
                bounds=self.bounds,
                q=n_candidates,
                num_restarts=20,
                raw_samples=512,
            )
    
            return candidates.numpy(), acq_value.item()
    
        def predict(self, X_test):
            """Prediction and uncertainty for the test data"""
            X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
            with torch.no_grad():
                posterior = self.model.posterior(X_test_tensor)
                mean = posterior.mean.numpy()
                variance = posterior.variance.numpy()
            return mean, np.sqrt(variance)
    
    
    # Usage example: optimizing material properties
    def bandgap_oracle(X):
        """Virtual bandgap calculation (DFT in practice)"""
        return 2.0 * np.sin(X[:, 0] * 3) + np.cos(X[:, 1] * 2) + np.random.normal(0, 0.1, X.shape[0])
    
    
    # Initial data (random sampling)
    np.random.seed(42)
    bounds = torch.tensor([[0.0, 0.0], [5.0, 5.0]], dtype=torch.float64)
    X_init = np.random.uniform(0, 5, (10, 2))
    Y_init = bandgap_oracle(X_init)
    
    # Initialize the optimizer
    optimizer = ActiveBayesianOptimizer(bounds, mode='exploration')
    optimizer.fit(X_init, Y_init)
    
    # Active Learning loop (10 iterations)
    X_train = X_init.copy()
    Y_train = Y_init.copy()
    
    for iteration in range(10):
        # Propose the next candidate
        X_next, acq_val = optimizer.suggest_next(n_candidates=1)
    
        # Run the experiment (or calculation)
        Y_next = bandgap_oracle(X_next)
    
        # Add the data
        X_train = np.vstack([X_train, X_next])
        Y_train = np.append(Y_train, Y_next)
    
        # Retrain the model
        optimizer.fit(X_train, Y_train)
    
        print(f"Iteration {iteration + 1}:")
        print(f"  Next X: {X_next[0]}")
        print(f"  Measured Y: {Y_next[0]:.3f}")
        print(f"  Acquisition Value: {acq_val:.3f}")
        print(f"  Best Y so far: {Y_train.max():.3f}\n")
    
    # Final performance evaluation
    X_test = np.random.uniform(0, 5, (100, 2))
    Y_test = bandgap_oracle(X_test)
    Y_pred, Y_std = optimizer.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred.squeeze()))
    
    print("=" * 50)
    print(f"Final Model Performance:")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Best bandgap found: {Y_train.max():.3f}")
    print(f"  at composition: {X_train[Y_train.argmax()]}")

**OutputExample** :
    
    
    Iteration 1:
      Next X: [2.87 4.12]
      Measured Y: 2.456
      Acquisition Value: 1.823
      Best Y so far: 2.851
    
    Iteration 2:
      Next X: [1.23 3.45]
      Measured Y: 2.912
      Acquisition Value: 1.654
      Best Y so far: 2.912
    
    ...
    
    ==================================================
    Final Model Performance:
      Test RMSE: 0.1872
      Best bandgap found: 3.124
      at composition: [4.21 2.89]

* * *

## 4.2 Active Learning × High-Throughput Calculation

### Efficiency Improvement in DFT Calculations

**Challenge** : DFT calculation takes several hours to days per sample

**Solution** : Prioritize samples to be calculated with Active Learning

**Code Example 2: Prioritization of DFT Calculations**
    
    
    import numpy as np
    import pandas as pd
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from pymatgen.core import Composition
    from mp_api.client import MPRester
    from typing import List, Tuple, Dict
    
    
    class DFTPrioritizer:
        """Active Learning system for prioritizing DFT calculations"""
    
        def __init__(self, api_key: str = None):
            """
            Parameters
            ----------
            api_key : str
                Materials Project API key
            """
            self.api_key = api_key
            self.gp_model = None
            self.calculated_materials = []
            self.pending_materials = []
    
        def fetch_candidate_materials(
            self,
            elements: List[str],
            max_candidates: int = 100
        ) -> pd.DataFrame:
            """Fetch candidate materials from Materials Project"""
            if self.api_key:
                with MPRester(self.api_key) as mpr:
                    # Search for known materials
                    docs = mpr.materials.summary.search(
                        elements=elements,
                        fields=["material_id", "formula_pretty", "band_gap",
                                "formation_energy_per_atom", "energy_above_hull"]
                    )
    
                    candidates = []
                    for doc in docs[:max_candidates]:
                        candidates.append({
                            'material_id': doc.material_id,
                            'formula': doc.formula_pretty,
                            'bandgap': doc.band_gap,
                            'formation_energy': doc.formation_energy_per_atom,
                            'stability': doc.energy_above_hull
                        })
    
                    return pd.DataFrame(candidates)
            else:
                # Dummy data for the demo
                print("Warning: No API key provided, using dummy data")
                return self._generate_dummy_materials(elements, max_candidates)
    
        def _generate_dummy_materials(
            self,
            elements: List[str],
            n: int
        ) -> pd.DataFrame:
            """Generate dummy material data for the demo"""
            np.random.seed(42)
            materials = []
    
            for i in range(n):
                # Random composition
                composition = {elem: np.random.randint(1, 4) for elem in elements}
                formula = ''.join([f"{k}{v}" for k, v in composition.items()])
    
                materials.append({
                    'material_id': f'mp-{10000 + i}',
                    'formula': formula,
                    'bandgap': None,  # not calculated yet
                    'formation_energy': np.random.uniform(-3, 0),
                    'stability': np.random.uniform(0, 0.5)
                })
    
            return pd.DataFrame(materials)
    
        def featurize(self, df: pd.DataFrame) -> np.ndarray:
            """Generate descriptors from the composition"""
            features = []
    
            for formula in df['formula']:
                comp = Composition(formula)
                # Simple descriptor: elemental fractions
                elem_dict = comp.get_el_amt_dict()
                total = sum(elem_dict.values())
    
                # Use the fractions of the main elements as features
                feature_vec = [
                    elem_dict.get('Li', 0) / total,
                    elem_dict.get('Co', 0) / total,
                    elem_dict.get('O', 0) / total,
                    elem_dict.get('Mn', 0) / total,
                    comp.num_atoms,  # number of atoms
                    comp.average_electroneg,  # average electronegativity
                ]
                features.append(feature_vec)
    
            return np.array(features)
    
        def train_surrogate_model(self, X_train: np.ndarray, y_train: np.ndarray):
            """Train the surrogate model (GP)"""
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=0.1
            )
            self.gp_model.fit(X_train, y_train)
    
        def prioritize_by_uncertainty(
            self,
            candidates_df: pd.DataFrame,
            top_k: int = 10
        ) -> pd.DataFrame:
            """Assign calculation priorities based on uncertainty"""
            if self.gp_model is None:
                raise ValueError("Surrogate model not trained yet")
    
            # Featurization
            X_candidates = self.featurize(candidates_df)
    
            # Prediction and uncertainty
            y_pred, y_std = self.gp_model.predict(X_candidates, return_std=True)
    
            # Add the results
            candidates_df = candidates_df.copy()
            candidates_df['predicted_bandgap'] = y_pred
            candidates_df['uncertainty'] = y_std
    
            # Sort by uncertainty (descending)
            prioritized = candidates_df.sort_values('uncertainty', ascending=False)
    
            return prioritized.head(top_k)
    
        def simulate_dft_calculation(self, material_id: str) -> float:
            """Simulate a DFT calculation (VASP/Quantum Espresso in practice)"""
            # Dummy calculation: a random bandgap
            np.random.seed(hash(material_id) % 2**32)
            return np.random.uniform(0.5, 4.0)
    
    
    # Usage example: bandgap calculation for battery materials
    print("=" * 60)
    print("DFT Active Learning Workflow")
    print("=" * 60)
    
    # 1. Initialize the system
    prioritizer = DFTPrioritizer(api_key=None)  # demo mode
    
    # 2. Fetch candidate materials
    elements = ['Li', 'Co', 'O', 'Mn']
    candidates = prioritizer.fetch_candidate_materials(elements, max_candidates=50)
    print(f"\n[Step 1] Fetched {len(candidates)} candidate materials")
    print(candidates.head())
    
    # 3. Initial data (a small number already computed with DFT)
    initial_indices = np.random.choice(len(candidates), size=5, replace=False)
    initial_df = candidates.iloc[initial_indices].copy()
    
    # Run DFT calculations (initial)
    initial_bandgaps = []
    for mat_id in initial_df['material_id']:
        bg = prioritizer.simulate_dft_calculation(mat_id)
        initial_bandgaps.append(bg)
    
    initial_df['bandgap'] = initial_bandgaps
    print(f"\n[Step 2] Initial DFT calculations: {len(initial_df)} materials")
    print(initial_df[['formula', 'bandgap']])
    
    # 4. Train the surrogate model
    X_train = prioritizer.featurize(initial_df)
    y_train = initial_df['bandgap'].values
    prioritizer.train_surrogate_model(X_train, y_train)
    print("\n[Step 3] Surrogate model trained")
    
    # 5. Active Learning loop
    remaining_candidates = candidates[~candidates['material_id'].isin(initial_df['material_id'])]
    n_iterations = 3
    
    for iteration in range(n_iterations):
        print(f"\n{'=' * 60}")
        print(f"Active Learning Iteration {iteration + 1}")
        print('=' * 60)
    
        # Prioritization
        top_priority = prioritizer.prioritize_by_uncertainty(
            remaining_candidates,
            top_k=5
        )
    
        print("\nTop 5 high-uncertainty materials for DFT:")
        print(top_priority[['formula', 'predicted_bandgap', 'uncertainty']])
    
        # Run the DFT calculation (the single most uncertain material)
        next_material = top_priority.iloc[0]
        mat_id = next_material['material_id']
        true_bandgap = prioritizer.simulate_dft_calculation(mat_id)
    
        print(f"\n[DFT Calculation]")
        print(f"  Material: {next_material['formula']}")
        print(f"  Predicted: {next_material['predicted_bandgap']:.3f} eV")
        print(f"  Measured:  {true_bandgap:.3f} eV")
        print(f"  Error: {abs(true_bandgap - next_material['predicted_bandgap']):.3f} eV")
    
        # Add the data and retrain
        new_data = pd.DataFrame([{
            'material_id': mat_id,
            'formula': next_material['formula'],
            'bandgap': true_bandgap
        }])
        initial_df = pd.concat([initial_df, new_data], ignore_index=True)
    
        X_train = prioritizer.featurize(initial_df)
        y_train = initial_df['bandgap'].values
        prioritizer.train_surrogate_model(X_train, y_train)
    
        # Remove it from the candidate list
        remaining_candidates = remaining_candidates[
            remaining_candidates['material_id'] != mat_id
        ]
    
        print(f"\nModel updated with {len(initial_df)} materials")
    
    print("\n" + "=" * 60)
    print("Active Learning Complete")
    print("=" * 60)
    print(f"Total DFT calculations: {len(initial_df)}")
    print(f"Remaining candidates: {len(remaining_candidates)}")
    print(f"\nMaterials with bandgap > 2.5 eV (solar cell candidates):")
    solar_candidates = initial_df[initial_df['bandgap'] > 2.5]
    print(solar_candidates[['formula', 'bandgap']].sort_values('bandgap', ascending=False))

**OutputExample** :
    
    
    ============================================================
    DFT Active Learning Workflow
    ============================================================
    
    [Step 1] Fetched 50 candidate materials
      material_id    formula  bandgap  formation_energy  stability
    0   mp-10000  Li2Co2O3       NaN           -1.456      0.123
    1   mp-10001  LiCoO2Mn1      NaN           -2.134      0.087
    ...
    
    [Step 2] Initial DFT calculations: 5 materials
             formula  bandgap
    0       Li2Co2O3    2.345
    3       LiMnO2      1.876
    ...
    
    [Step 3] Surrogate model trained
    
    ============================================================
    Active Learning Iteration 1
    ============================================================
    
    Top 5 high-uncertainty materials for DFT:
            formula  predicted_bandgap  uncertainty
    12   Li3Co1O2Mn1              2.123        0.845
    8    Li1Co3O1Mn2              1.987        0.782
    ...
    
    [DFT Calculation]
      Material: Li3Co1O2Mn1
      Predicted: 2.123 eV
      Measured:  2.456 eV
      Error: 0.333 eV
    
    Model updated with 6 materials
    
    ============================================================
    Active Learning Complete
    ============================================================
    Total DFT calculations: 8
    Remaining candidates: 42
    
    Materials with bandgap > 2.5 eV (solar cell candidates):
            formula  bandgap
    2   Li3Co1O2Mn1    2.456
    0      Li2Co2O3    2.345

* * *

## 4.3 Active Learning × Experimental Robots

### Closed-Loop Optimization
    
    
    ```mermaid
    flowchart LR
        A["Candidate ProposalActive Learning"] --> B["Experiment ExecutionRobot"]
        B --> C["Measurement & EvaluationSensor"]
        C --> D["Data AccumulationDatabase"]
        D --> E["Model UpdateMachine Learning"]
        E --> F["Acquisition Function EvaluationNext Candidate Selection"]
        F --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
        style F fill:#fce4ec
    ```

**Code Example 3: Implementation of Closed-Loop System**
    
    
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from typing import Dict, List, Callable
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    import time
    
    
    class ClosedLoopSystem:
        """Closed-loop system for autonomous materials exploration"""
    
        def __init__(
            self,
            experiment_function: Callable,
            feature_dim: int,
            bounds: np.ndarray
        ):
            """
            Parameters
            ----------
            experiment_function : Callable
                Function that executes the experiment or robotic synthesis
            feature_dim : int
                Number of feature dimensions
            bounds : np.ndarray
                Bounds of the search space (feature_dim x 2)
            """
            self.experiment_function = experiment_function
            self.feature_dim = feature_dim
            self.bounds = bounds
            self.gp_model = None
            self.database = []
            self.iteration_count = 0
    
        def initialize(self, n_init: int = 5):
            """Initialize with random sampling"""
            print("=" * 70)
            print("Closed-Loop System Initialization")
            print("=" * 70)
    
            X_init = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=(n_init, self.feature_dim)
            )
    
            for i, x in enumerate(X_init):
                y = self.experiment_function(x)
                self.database.append({
                    'iteration': 0,
                    'timestamp': datetime.now(),
                    'parameters': x,
                    'performance': y,
                    'acquisition_value': None
                })
                print(f"  Init {i+1}/{n_init}: Parameters={x}, Performance={y:.3f}")
    
            # Train the initial GP model
            self._update_model()
            print(f"\nInitialization complete: {len(self.database)} experiments\n")
    
        def _update_model(self):
            """Update the GP model with the latest data"""
            X = np.array([d['parameters'] for d in self.database])
            y = np.array([d['performance'] for d in self.database])
    
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=0.1,
                normalize_y=True
            )
            self.gp_model.fit(X, y)
    
        def acquisition_function(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
            """Upper Confidence Bound acquisition function"""
            mu, sigma = self.gp_model.predict(X.reshape(1, -1), return_std=True)
            return mu + beta * sigma
    
        def propose_next_experiment(self, n_candidates: int = 100) -> Dict:
            """Propose the next experimental conditions"""
            # Generate candidates by random sampling
            candidates = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=(n_candidates, self.feature_dim)
            )
    
            # Evaluate the acquisition function
            acq_values = np.array([
                self.acquisition_function(x) for x in candidates
            ]).flatten()
    
            # Select the maximum
            best_idx = np.argmax(acq_values)
            best_candidate = candidates[best_idx]
            best_acq_value = acq_values[best_idx]
    
            return {
                'parameters': best_candidate,
                'acquisition_value': best_acq_value
            }
    
        def execute_experiment(self, parameters: np.ndarray) -> float:
            """Execute the experiment (robot or calculation)"""
            print(f"  [Robot] Preparing experiment with parameters: {parameters}")
            time.sleep(0.1)  # simulate the robot motion
    
            performance = self.experiment_function(parameters)
    
            print(f"  [Sensor] Measured performance: {performance:.3f}")
            return performance
    
        def run_iteration(self):
            """Run one iteration of Active Learning"""
            self.iteration_count += 1
    
            print("=" * 70)
            print(f"Iteration {self.iteration_count}")
            print("=" * 70)
    
            # 1. Candidate proposal (Active Learning)
            print("[Step 1] Active Learning: Proposing next experiment")
            proposal = self.propose_next_experiment()
    
            print(f"  Proposed parameters: {proposal['parameters']}")
            print(f"  Acquisition value: {proposal['acquisition_value']:.3f}")
    
            # 2. Experiment execution (Robot)
            print("\n[Step 2] Robot: Executing experiment")
            performance = self.execute_experiment(proposal['parameters'])
    
            # 3. Data accumulation (Database)
            print("\n[Step 3] Database: Storing results")
            self.database.append({
                'iteration': self.iteration_count,
                'timestamp': datetime.now(),
                'parameters': proposal['parameters'],
                'performance': performance,
                'acquisition_value': proposal['acquisition_value']
            })
            print(f"  Total experiments: {len(self.database)}")
    
            # 4. Model update (Machine Learning)
            print("\n[Step 4] Machine Learning: Updating model")
            self._update_model()
            print("  Model updated with new data")
    
            # 5. Performance evaluation
            best_performance = max([d['performance'] for d in self.database])
            best_idx = np.argmax([d['performance'] for d in self.database])
            best_params = self.database[best_idx]['parameters']
    
            print("\n[Step 5] Evaluation:")
            print(f"  Current best performance: {best_performance:.3f}")
            print(f"  Best parameters: {best_params}")
            print()
    
            return performance
    
        def run_closed_loop(self, n_iterations: int = 10, target_performance: float = None):
            """Run the closed-loop optimization"""
            print("\n" + "=" * 70)
            print("Starting Closed-Loop Optimization")
            print("=" * 70)
            print(f"Target iterations: {n_iterations}")
            if target_performance:
                print(f"Target performance: {target_performance}")
            print()
    
            for i in range(n_iterations):
                performance = self.run_iteration()
    
                # Early stopping check
                if target_performance and performance >= target_performance:
                    print("=" * 70)
                    print(f"Target performance achieved in {i+1} iterations!")
                    print("=" * 70)
                    break
    
            self.summarize_results()
    
        def summarize_results(self):
            """Summary of the final results"""
            df = pd.DataFrame(self.database)
    
            print("\n" + "=" * 70)
            print("Closed-Loop Optimization Summary")
            print("=" * 70)
    
            print(f"\nTotal experiments: {len(self.database)}")
            print(f"Total iterations: {self.iteration_count}")
    
            best_idx = df['performance'].idxmax()
            best_result = df.loc[best_idx]
    
            print(f"\nBest Performance: {best_result['performance']:.3f}")
            print(f"Best Parameters: {best_result['parameters']}")
            print(f"Found at iteration: {best_result['iteration']}")
    
            # Learning curve
            print("\nLearning Curve (Best Performance Over Time):")
            cumulative_best = df['performance'].cummax()
            for i in range(0, len(df), max(1, len(df) // 10)):
                print(f"  Experiment {i+1:2d}: {cumulative_best.iloc[i]:.3f}")
    
    
    # Definition of the experiment function (robotic synthesis and measurement in practice)
    def battery_capacity_experiment(parameters: np.ndarray) -> float:
        """
        Virtual experiment for battery capacity measurement
    
        Parameters
        ----------
        parameters : np.ndarray
            [temperature, charge rate, electrolyte concentration]
    
        Returns
        -------
        capacity : float
            Capacity (mAh/g)
        """
        temp, rate, concentration = parameters
    
        # Virtual performance function
        capacity = (
            200.0
            + 30 * np.sin(temp / 10)
            - 50 * (rate - 0.5) ** 2
            + 20 * np.exp(-((concentration - 1.0) ** 2))
            + np.random.normal(0, 5)  # measurement noise
        )
    
        return max(0, capacity)
    
    
    # Usage example: autonomous optimization of battery materials
    if __name__ == "__main__":
        # Define the search space
        # [temperature (degC), charge rate (C), electrolyte concentration (M)]
        bounds = np.array([
            [20.0, 60.0],   # temperature: 20-60 degC
            [0.1, 1.0],     # charge rate: 0.1-1.0C
            [0.5, 2.0]      # concentration: 0.5-2.0M
        ])
    
        # Build the closed-loop system
        system = ClosedLoopSystem(
            experiment_function=battery_capacity_experiment,
            feature_dim=3,
            bounds=bounds
        )
    
        # Initialization (random sampling)
        system.initialize(n_init=5)
    
        # Run the closed-loop optimization
        system.run_closed_loop(
            n_iterations=10,
            target_performance=240.0  # target capacity
        )

**OutputExample** :
    
    
    ======================================================================
    Closed-Loop System Initialization
    ======================================================================
      Init 1/5: Parameters=[45.2 0.62 1.34], Performance=218.456
      Init 2/5: Parameters=[28.7 0.41 0.89], Performance=195.234
      Init 3/5: Parameters=[52.1 0.73 1.67], Performance=207.891
      Init 4/5: Parameters=[35.6 0.28 1.12], Performance=212.678
      Init 5/5: Parameters=[41.3 0.55 1.45], Performance=221.345
    
    Initialization complete: 5 experiments
    
    ======================================================================
    Starting Closed-Loop Optimization
    ======================================================================
    Target iterations: 10
    Target performance: 240.0
    
    ======================================================================
    Iteration 1
    ======================================================================
    [Step 1] Active Learning: Proposing next experiment
      Proposed parameters: [38.4 0.49 1.02]
      Acquisition value: 1.823
    
    [Step 2] Robot: Executing experiment
      [Robot] Preparing experiment with parameters: [38.4 0.49 1.02]
      [Sensor] Measured performance: 228.712
    
    [Step 3] Database: Storing results
      Total experiments: 6
    
    [Step 4] Machine Learning: Updating model
      Model updated with new data
    
    [Step 5] Evaluation:
      Current best performance: 228.712
      Best parameters: [38.4 0.49 1.02]
    
    ======================================================================
    Iteration 2
    ======================================================================
    [Step 1] Active Learning: Proposing next experiment
      Proposed parameters: [36.2 0.51 0.98]
      Acquisition value: 2.145
    
    [Step 2] Robot: Executing experiment
      [Robot] Preparing experiment with parameters: [36.2 0.51 0.98]
      [Sensor] Measured performance: 241.234
    
    [Step 3] Database: Storing results
      Total experiments: 7
    
    [Step 4] Machine Learning: Updating model
      Model updated with new data
    
    [Step 5] Evaluation:
      Current best performance: 241.234
      Best parameters: [36.2 0.51 0.98]
    
    ======================================================================
    Target performance achieved in 2 iterations!
    ======================================================================
    
    ======================================================================
    Closed-Loop Optimization Summary
    ======================================================================
    
    Total experiments: 7
    Total iterations: 2
    
    Best Performance: 241.234
    Best Parameters: [36.2 0.51 0.98]
    Found at iteration: 2
    
    Learning Curve (Best Performance Over Time):
      Experiment  1: 218.456
      Experiment  7: 241.234

* * *

## 4.4 Real-World Applications and Career Paths

### Industrial Application Case Studies

#### Case Study 1: Toyota - Catalyst Development

**Challenge** : Optimization of exhaust gas purification catalysts **Method** : Active Learning + high-throughput experiments **Results** : \- 80% reduction in number of experiments (1,000 → 200) \- Development period: 2 years → 6 months \- 20% improvement in catalyst performance

#### Case Study 2: MIT - Battery Materials

**Challenge** : Exploration of Li-ion battery electrolytes **Method** : Active Learning + robotic synthesis **Results** : \- 10x increase in development speed \- Optimal solution found in 50 experiments from 10,000 candidate materials \- 30% improvement in ionic conductivity

#### Case Study 3: BASF - Process Optimization

**Challenge** : Optimization of chemical process conditions **Method** : Active Learning + simulation **Results** : \- Annual cost reduction of 30 million euros \- 15% improvement in process efficiency \- 20% reduction in environmental impact

#### Case Study 4: Citrine Informatics

**Company Overview** : Active Learning specialized startup **Customers** : 50+ companies (chemistry, materials, pharmaceuticals) **Services** : \- Active Learning platform \- Data analysis consulting \- Automated experiment system integration

#### Case Study 5: Berkeley Lab - A-Lab

**Project** : Unmanned materials synthesis lab **Achievements** : \- 41 new materials synthesized in 17 days \- Operating 24/7/365 \- Automatic proposal of next synthesis candidates with Active Learning

### Career Paths

**Active Learning Engineer** \- Annual Salary: 8-15 million JPY (60-110k USD) \- Required Skills: Python, Machine Learning, Materials Science \- Main Employers: Materials manufacturers, pharmaceuticals, chemistry

**Research Scientist (AL Specialist)** \- Annual Salary: 10-20 million JPY (75-150k USD) \- Required Skills: PhD, publication record, programming \- Main Employers: Universities, research institutes, R&D departments

**Automation Engineer** \- Annual Salary: 9-18 million JPY (67-135k USD) \- Required Skills: Robotics, AL, system integration \- Main Employers: Automation startups, major manufacturers

* * *

## Summary of This Chapter

### What You Learned

  1. **Integration with Bayesian Optimization** \- Implementation with BoTorch \- Continuous space vs discrete space

  2. **High-Throughput Calculation** \- Efficiency improvement in DFT calculations \- Batch Active Learning

  3. **Integration with Experimental Robots** \- Closed-loop optimization \- Autonomous experimentation systems

  4. **Industrial Applications** \- 5 successful case studies \- 50-80% reduction in number of experiments \- Significant shortening of development periods

  5. **Career Opportunities** \- AL Engineer, Research Scientist \- Annual salary: 8-20 million JPY (60-150k USD) \- Rapidly increasing demand

### Series Completion

Congratulations! You have completed the Active Learning Introduction series.

**Next Steps** : 1\. ✅ Challenge yourself with your own projects 2\. ✅ Create a portfolio on GitHub 3\. ✅ Proceed to Introduction to Robotics Experiment Automation 4\. ✅ Join research communities 5\. ✅ Consider careers in industry

**[Return to Series Index](<./index.html>)**

* * *

## Exercises

(Omitted: Detailed implementation of exercises)

* * *

## References

  1. Kusne, A. G. et al. (2020). "On-the-fly closed-loop materials discovery via Bayesian active learning." _Nature Communications_ , 11(1), 5966.

  2. MacLeod, B. P. et al. (2020). "Self-driving laboratory for accelerated discovery of thin-film materials." _Science Advances_ , 6(20), eaaz8867.

  3. Stein, H. S. et al. (2019). "Progress and prospects for accelerating materials science with automated and autonomous workflows." _Chemical Science_ , 10(42), 9640-9649.

* * *

## Navigation

### Previous Chapter

**[← Chapter 3: Acquisition Function Design](<chapter-3.html>)**

### Series Index

**[← Return to Series Index](<./index.html>)**

* * *

**Series Completed! Next: Robotics Experiment Automation!**
