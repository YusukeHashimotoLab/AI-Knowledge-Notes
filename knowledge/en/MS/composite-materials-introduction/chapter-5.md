---
title: Chapter 5 Python Practice
chapter_title: Chapter 5 Python Practice
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/composite-materials-introduction/chapter-5.html>) | Last sync: 2025-11-16

### Composite Materials Introduction

  * [Table of Contents](<index.html>)
  * [Chapter 1 Fundamentals of Composite Materials](<chapter-1.html>)
  * [Chapter 2 Fiber-Reinforced Composites](<chapter-2.html>)
  * [Chapter 3 Particle & Laminated Composites](<chapter-3.html>)
  * [Chapter 4 Evaluation of Composite Materials](<chapter-4.html>)
  * [Chapter 5 Python Practice](<chapter-5.html>)

#### Materials Science Series

  * [Polymer Materials Introduction](<../polymer-materials-introduction/index.html>)
  * [Thin Film & Nano Materials Introduction](<../thin-film-nano-introduction/index.html>)
  * [Composite Materials Introduction](<index.html>)

# Chapter 5 Python Practice

This chapter covers Chapter 5 Python Practice. You will learn essential concepts and techniques.

### Learning Objectives

  * **Basic Level:** Implement Classical Lamination Theory (CLT) in Python and calculate A-B-D matrices
  * **Application Level:** Design stacking sequences using optimization algorithms and perform performance prediction
  * **Advanced Level:** Implement finite element method preprocessing and expand to large-scale analysis

## 5.1 Complete Implementation of Classical Lamination Theory

### 5.1.1 Object-Oriented Design

We design a composite material analysis tool based on classes to ensure reusability and extensibility. 

#### Example 5.1: Implementation of CLT Analysis Library
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import List, Tuple, Optional
    
    @dataclass
    class Material:
        """Class to hold single-layer material properties"""
        name: str
        E1: float  # Longitudinal elastic modulus [GPa]
        E2: float  # Transverse elastic modulus [GPa]
        nu12: float  # Major Poisson's ratio
        G12: float  # Shear modulus [GPa]
        Xt: float  # Longitudinal tensile strength [MPa]
        Xc: float  # Longitudinal compressive strength [MPa]
        Yt: float  # Transverse tensile strength [MPa]
        Yc: float  # Transverse compressive strength [MPa]
        S: float   # Shear strength [MPa]
    
        def __post_init__(self):
            """Verify reciprocal theorem"""
            self.nu21 = self.nu12 * self.E2 / self.E1
    
        def Q_matrix(self) -> np.ndarray:
            """Calculate reduced stiffness matrix [Q]"""
            denom = 1 - self.nu12 * self.nu21
            Q11 = self.E1 / denom
            Q22 = self.E2 / denom
            Q12 = self.nu12 * self.E2 / denom
            Q66 = self.G12
    
            return np.array([
                [Q11, Q12, 0],
                [Q12, Q22, 0],
                [0, 0, Q66]
            ]) * 1000  # GPa → MPa
    
    class Laminate:
        """Laminate analysis class"""
    
        def __init__(self, material: Material, layup: List[float], t_ply: float):
            """
            Parameters:
            -----------
            material : Material
                Single-layer material
            layup : List[float]
                Stacking sequence [θ1, θ2, ..., θn] (degrees)
            t_ply : float
                Single-layer thickness [mm]
            """
            self.material = material
            self.layup = np.array(layup)
            self.t_ply = t_ply
            self.n_plies = len(layup)
            self.total_thickness = self.n_plies * t_ply
    
            # Calculate z-coordinates (reference to mid-plane)
            self.z = np.linspace(
                -self.total_thickness / 2,
                self.total_thickness / 2,
                self.n_plies + 1
            )
    
            # Calculate A, B, D matrices
            self.A, self.B, self.D = self._compute_ABD()
    
        @staticmethod
        def transformation_matrix(theta: float) -> np.ndarray:
            """Coordinate transformation matrix [T]"""
            theta_rad = np.radians(theta)
            c = np.cos(theta_rad)
            s = np.sin(theta_rad)
    
            return np.array([
                [c**2, s**2, 2*s*c],
                [s**2, c**2, -2*s*c],
                [-s*c, s*c, c**2 - s**2]
            ])
    
        def Q_bar(self, theta: float) -> np.ndarray:
            """Off-axis stiffness matrix [Q̄]"""
            Q = self.material.Q_matrix()
            T = self.transformation_matrix(theta)
            T_inv = np.linalg.inv(T)
    
            return T_inv @ Q @ T_inv.T
    
        def _compute_ABD(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Calculate A-B-D matrices"""
            A = np.zeros((3, 3))
            B = np.zeros((3, 3))
            D = np.zeros((3, 3))
    
            for k in range(self.n_plies):
                Q_bar = self.Q_bar(self.layup[k])
                z_k = self.z[k]
                z_k1 = self.z[k + 1]
    
                A += Q_bar * (z_k1 - z_k)
                B += 0.5 * Q_bar * (z_k1**2 - z_k**2)
                D += (1/3) * Q_bar * (z_k1**3 - z_k**3)
    
            return A, B, D
    
        def ABD_matrix(self) -> np.ndarray:
            """Complete 6×6 ABD matrix"""
            return np.block([
                [self.A, self.B],
                [self.B, self.D]
            ])
    
        def is_symmetric(self) -> bool:
            """Determine whether stacking sequence is symmetric"""
            n = len(self.layup)
            for i in range(n // 2):
                if self.layup[i] != self.layup[n - 1 - i]:
                    return False
            return True
    
        def effective_properties(self) -> dict:
            """Calculate equivalent in-plane properties"""
            # Compliance matrix
            a = np.linalg.inv(self.A)
    
            # Equivalent Young's moduli
            Ex = 1 / (a[0, 0] * self.total_thickness)
            Ey = 1 / (a[1, 1] * self.total_thickness)
    
            # Equivalent Poisson's ratios
            nu_xy = -a[0, 1] / a[0, 0]
            nu_yx = -a[1, 0] / a[1, 1]
    
            # Equivalent shear modulus
            Gxy = 1 / (a[2, 2] * self.total_thickness)
    
            return {
                'Ex': Ex / 1000,  # MPa → GPa
                'Ey': Ey / 1000,
                'nu_xy': nu_xy,
                'nu_yx': nu_yx,
                'Gxy': Gxy / 1000
            }
    
        def print_summary(self):
            """Display laminate information"""
            print("="*70)
            print(f"Laminate Summary: {self.material.name}")
            print("="*70)
            print(f"Stacking Sequence: {self.layup}")
            print(f"Number of Plies: {self.n_plies}")
            print(f"Single Ply Thickness: {self.t_ply} mm")
            print(f"Total Thickness: {self.total_thickness} mm")
            print(f"Symmetric Laminate: {self.is_symmetric()}")
    
            print("\n[A] Matrix (N/mm):")
            print(self.A)
    
            print("\n[B] Matrix (N):")
            print(self.B)
            print(f"B-matrix Norm: {np.linalg.norm(self.B):.2e}")
    
            print("\n[D] Matrix (N·mm):")
            print(self.D)
    
            props = self.effective_properties()
            print("\nEquivalent In-Plane Properties:")
            print(f"  Ex = {props['Ex']:.1f} GPa")
            print(f"  Ey = {props['Ey']:.1f} GPa")
            print(f"  νxy = {props['nu_xy']:.3f}")
            print(f"  Gxy = {props['Gxy']:.1f} GPa")
            print("="*70)
    
    # Usage example
    # CFRP material definition
    cfrp = Material(
        name="T300/Epoxy",
        E1=140.0, E2=10.0, nu12=0.30, G12=5.0,
        Xt=1500, Xc=1200, Yt=50, Yc=200, S=70
    )
    
    # Stacking sequences
    layup_symmetric = [0, 45, -45, 90, 90, -45, 45, 0]
    layup_quasi_iso = [0, 45, -45, 90]
    
    # Create laminates
    lam_sym = Laminate(cfrp, layup_symmetric, t_ply=0.125)
    lam_qi = Laminate(cfrp, layup_quasi_iso, t_ply=0.125)
    
    # Display summaries
    lam_sym.print_summary()
    print("\n")
    lam_qi.print_summary()

### 5.1.2 Stress and Strain Analysis

Calculate the stress in each layer from applied loads and compare with failure criteria. 

#### Example 5.2: Stress Analysis of Laminates and First Ply Failure
    
    
    class FailureCriterion:
        """Base class for failure criteria"""
    
        def __init__(self, material: Material):
            self.material = material
    
        def failure_index(self, sigma1: float, sigma2: float, tau12: float) -> float:
            """Calculate failure index (implemented in derived classes)"""
            raise NotImplementedError
    
    class TsaiWuCriterion(FailureCriterion):
        """Tsai-Wu failure criterion"""
    
        def __init__(self, material: Material):
            super().__init__(material)
    
            # Tsai-Wu coefficients
            self.F1 = 1/material.Xt - 1/material.Xc
            self.F2 = 1/material.Yt - 1/material.Yc
            self.F11 = 1/(material.Xt * material.Xc)
            self.F22 = 1/(material.Yt * material.Yc)
            self.F66 = 1/material.S**2
            self.F12 = -0.5 * np.sqrt(self.F11 * self.F22)
    
        def failure_index(self, sigma1: float, sigma2: float, tau12: float) -> float:
            """Tsai-Wu failure index"""
            FI = (self.F1 * sigma1 + self.F2 * sigma2 +
                  self.F11 * sigma1**2 + self.F22 * sigma2**2 +
                  self.F66 * tau12**2 + 2 * self.F12 * sigma1 * sigma2)
            return FI
    
    class LaminateAnalysis:
        """Laminate load analysis class"""
    
        def __init__(self, laminate: Laminate, criterion: FailureCriterion):
            self.laminate = laminate
            self.criterion = criterion
    
        def analyze_loading(self, Nx: float, Ny: float, Nxy: float,
                            Mx: float = 0, My: float = 0, Mxy: float = 0) -> List[dict]:
            """
            Stress analysis of each layer under loading conditions
    
            Parameters:
            -----------
            Nx, Ny, Nxy : float
                Resultant forces [N/mm]
            Mx, My, Mxy : float
                Resultant moments [N·mm/mm]
    
            Returns:
            --------
            results : List[dict]
                Stress and failure index for each layer
            """
            # Inverse of ABD matrix
            ABD_inv = np.linalg.inv(self.laminate.ABD_matrix())
    
            # Load vector
            load = np.array([Nx, Ny, Nxy, Mx, My, Mxy])
    
            # Mid-plane strains and curvatures
            strain_curvature = ABD_inv @ load
            epsilon0 = strain_curvature[:3]
            kappa = strain_curvature[3:]
    
            results = []
    
            for k in range(self.laminate.n_plies):
                # Layer mid-plane position
                z_mid = (self.laminate.z[k] + self.laminate.z[k + 1]) / 2
    
                # Strain in global coordinate system
                epsilon_global = epsilon0 + z_mid * kappa
    
                # Stress in global coordinate system
                Q_bar = self.laminate.Q_bar(self.laminate.layup[k])
                stress_global = Q_bar @ epsilon_global
    
                # Transform to principal axis coordinate system
                T = self.laminate.transformation_matrix(self.laminate.layup[k])
                stress_local = T @ stress_global
    
                sigma1, sigma2, tau12 = stress_local
    
                # Failure index
                FI = self.criterion.failure_index(sigma1, sigma2, tau12)
                SF = 1 / np.sqrt(FI) if FI > 0 else np.inf
    
                results.append({
                    'ply': k + 1,
                    'angle': self.laminate.layup[k],
                    'z': z_mid,
                    'strain_global': epsilon_global,
                    'stress_global': stress_global,
                    'stress_local': stress_local,
                    'FI': FI,
                    'SF': SF
                })
    
            return results
    
        def first_ply_failure(self, Nx: float, Ny: float, Nxy: float) -> Tuple[int, float]:
            """
            Determine First Ply Failure load
    
            Returns:
            --------
            fpf_ply : int
                Number of first layer to fail
            fpf_load : float
                FPF load multiplier
            """
            # Analysis under unit load
            results = self.analyze_loading(Nx, Ny, Nxy)
    
            # Find minimum safety factor
            min_sf = min(r['SF'] for r in results)
            fpf_ply = min((r for r in results), key=lambda r: r['SF'])['ply']
    
            return fpf_ply, min_sf
    
    # Usage example
    cfrp = Material(
        name="T300/Epoxy",
        E1=140.0, E2=10.0, nu12=0.30, G12=5.0,
        Xt=1500, Xc=1200, Yt=50, Yc=200, S=70
    )
    
    layup = [0, 45, -45, 90]
    lam = Laminate(cfrp, layup, t_ply=0.125)
    
    # Tsai-Wu criterion
    criterion = TsaiWuCriterion(cfrp)
    
    # Analysis object
    analysis = LaminateAnalysis(lam, criterion)
    
    # Loading conditions (uniaxial tension)
    Nx = 100  # N/mm
    Ny = 0
    Nxy = 0
    
    # Stress analysis
    results = analysis.analyze_loading(Nx, Ny, Nxy)
    
    # Display results
    print("Laminate Stress Analysis Results:")
    print("="*80)
    print(f"Loading: Nx = {Nx} N/mm, Ny = {Ny} N/mm, Nxy = {Nxy} N/mm")
    print("-"*80)
    print(f"{'Ply':>3} {'Angle':>6} {'σ1':>10} {'σ2':>10} {'τ12':>10} {'FI':>8} {'SF':>8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['ply']:3d} {r['angle']:6.0f}° {r['stress_local'][0]:10.1f} "
              f"{r['stress_local'][1]:10.1f} {r['stress_local'][2]:10.1f} "
              f"{r['FI']:8.3f} {r['SF']:8.2f}")
    
    # FPF
    fpf_ply, fpf_sf = analysis.first_ply_failure(Nx, Ny, Nxy)
    print("-"*80)
    print(f"First Ply Failure: Ply {fpf_ply} (Angle {layup[fpf_ply-1]}°)")
    print(f"Safety Factor: {fpf_sf:.2f}")
    print(f"Failure Load: Nx = {Nx * fpf_sf:.1f} N/mm")

## 5.2 Optimal Stacking Design

### 5.2.1 Genetic Algorithm (GA)

Genetic algorithms are effective for optimization of discrete variables (fiber orientation angles). 

#### Example 5.3: Stacking Sequence Optimization by GA
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import random
    from typing import List, Callable
    import numpy as np
    
    class GeneticAlgorithm:
        """Genetic algorithm for stacking design optimization"""
    
        def __init__(self, n_plies: int, angle_options: List[float],
                     objective_func: Callable, symmetric: bool = True,
                     pop_size: int = 50, generations: int = 100):
            """
            Parameters:
            -----------
            n_plies : int
                Number of plies
            angle_options : List[float]
                Available angles [degrees]
            objective_func : Callable
                Objective function (layup → score, smaller is better)
            symmetric : bool
                Whether to enforce symmetric laminate
            """
            self.n_plies = n_plies
            self.angle_options = angle_options
            self.objective_func = objective_func
            self.symmetric = symmetric
            self.pop_size = pop_size
            self.generations = generations
    
            # For symmetric laminates, treat only half as genes
            self.gene_length = n_plies // 2 if symmetric else n_plies
    
        def create_individual(self) -> List[float]:
            """Randomly generate individual (stacking sequence)"""
            genes = [random.choice(self.angle_options) for _ in range(self.gene_length)]
    
            if self.symmetric:
                # Make symmetric
                return genes + genes[::-1]
            else:
                return genes
    
        def fitness(self, individual: List[float]) -> float:
            """Fitness (inverse of objective function)"""
            score = self.objective_func(individual)
            return 1 / (1 + score)  # Higher fitness for smaller score
    
        def selection(self, population: List[List[float]]) -> List[List[float]]:
            """Tournament selection"""
            tournament_size = 3
            selected = []
    
            for _ in range(len(population)):
                tournament = random.sample(population, tournament_size)
                winner = max(tournament, key=self.fitness)
                selected.append(winner)
    
            return selected
    
        def crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
            """Single-point crossover"""
            point = random.randint(1, self.gene_length - 1)
    
            if self.symmetric:
                # Crossover on half of genes
                genes1 = parent1[:self.gene_length]
                genes2 = parent2[:self.gene_length]
                child_genes = genes1[:point] + genes2[point:]
                return child_genes + child_genes[::-1]
            else:
                return parent1[:point] + parent2[point:]
    
        def mutate(self, individual: List[float], mutation_rate: float = 0.1) -> List[float]:
            """Mutation"""
            if self.symmetric:
                genes = individual[:self.gene_length]
                mutated_genes = []
    
                for gene in genes:
                    if random.random() < mutation_rate:
                        mutated_genes.append(random.choice(self.angle_options))
                    else:
                        mutated_genes.append(gene)
    
                return mutated_genes + mutated_genes[::-1]
            else:
                return [
                    random.choice(self.angle_options) if random.random() < mutation_rate else gene
                    for gene in individual
                ]
    
        def optimize(self) -> Tuple[List[float], float]:
            """Execute optimization"""
            # Initial population
            population = [self.create_individual() for _ in range(self.pop_size)]
    
            best_history = []
    
            for gen in range(self.generations):
                # Fitness evaluation
                fitnesses = [self.fitness(ind) for ind in population]
                best_idx = np.argmax(fitnesses)
                best_individual = population[best_idx]
                best_score = self.objective_func(best_individual)
    
                best_history.append(best_score)
    
                if gen % 10 == 0:
                    print(f"Generation {gen}: Best Score = {best_score:.4f}, "
                          f"Layup = {best_individual}")
    
                # Selection
                selected = self.selection(population)
    
                # Next generation
                next_population = [best_individual]  # Elitism
    
                while len(next_population) < self.pop_size:
                    parent1, parent2 = random.sample(selected, 2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    next_population.append(child)
    
                population = next_population
    
            # Best individual from final generation
            fitnesses = [self.fitness(ind) for ind in population]
            best_idx = np.argmax(fitnesses)
            best_individual = population[best_idx]
            best_score = self.objective_func(best_individual)
    
            return best_individual, best_score
    
    # Define optimization problem
    cfrp = Material(
        name="T300/Epoxy",
        E1=140.0, E2=10.0, nu12=0.30, G12=5.0,
        Xt=1500, Xc=1200, Yt=50, Yc=200, S=70
    )
    
    t_ply = 0.125
    
    # Objective function: Minimize difference between Ex and Ey (approach quasi-isotropic)
    def objective_quasi_isotropic(layup):
        lam = Laminate(cfrp, layup, t_ply)
        props = lam.effective_properties()
        # Relative difference between Ex and Ey
        diff = abs(props['Ex'] - props['Ey']) / props['Ex']
        return diff
    
    # Execute GA
    angle_options = [0, 45, -45, 90]
    n_plies = 8
    
    ga = GeneticAlgorithm(
        n_plies=n_plies,
        angle_options=angle_options,
        objective_func=objective_quasi_isotropic,
        symmetric=True,
        pop_size=50,
        generations=100
    )
    
    best_layup, best_score = ga.optimize()
    
    print("\n" + "="*70)
    print("Optimal Stacking Design Results:")
    print("="*70)
    print(f"Optimal Layup: {best_layup}")
    print(f"Objective Function Value (Ex-Ey difference): {best_score:.4f}")
    
    # Detailed analysis of optimal layup
    lam_opt = Laminate(cfrp, best_layup, t_ply)
    lam_opt.print_summary()

### 5.2.2 Multi-Objective Optimization

Simultaneously optimize multiple objectives such as strength, stiffness, and weight. 

#### Example 5.4: Multi-Objective Optimization by NSGA-II
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    from scipy.optimize import differential_evolution
    import matplotlib.pyplot as plt
    
    def multi_objective_function(layup_continuous, material, t_ply,
                                  target_Nx, target_Ny):
        """
        Multi-objective function: Minimize weight & Ensure strength
    
        Parameters:
        -----------
        layup_continuous : array
            Stacking sequence as continuous variables [0-3] → [0, 45, -45, 90]
    
        Returns:
        --------
        objectives : tuple
            (weight, inverse safety factor)
        """
        # Convert continuous variables to discrete angles
        angle_map = {0: 0, 1: 45, 2: -45, 3: 90}
        layup = [angle_map[int(round(x))] for x in layup_continuous]
    
        # Create laminate
        lam = Laminate(material, layup, t_ply)
    
        # Weight (proportional to thickness)
        mass = lam.total_thickness
    
        # Safety factor (Tsai-Wu)
        criterion = TsaiWuCriterion(material)
        analysis = LaminateAnalysis(lam, criterion)
    
        try:
            results = analysis.analyze_loading(target_Nx, target_Ny, 0)
            min_sf = min(r['SF'] for r in results)
        except:
            min_sf = 0.1  # Penalty on error
    
        # Objectives: Minimize weight, maximize safety factor (minimize inverse)
        return mass, 1 / min_sf
    
    # Pareto frontier exploration (simplified: scalarization method)
    cfrp = Material(
        name="T300/Epoxy",
        E1=140.0, E2=10.0, nu12=0.30, G12=5.0,
        Xt=1500, Xc=1200, Yt=50, Yc=200, S=70
    )
    
    t_ply = 0.125
    target_Nx = 150  # N/mm
    target_Ny = 50   # N/mm
    n_plies = 12
    
    # Weighted scalarization method
    weights = np.linspace(0, 1, 11)
    pareto_solutions = []
    
    for w in weights:
        def scalarized_objective(x):
            mass, inv_sf = multi_objective_function(x, cfrp, t_ply, target_Nx, target_Ny)
            # Normalize and weighted sum
            return w * mass / 2.0 + (1 - w) * inv_sf * 10
    
        # Optimization (differential_evolution)
        bounds = [(0, 3)] * n_plies  # Continuous values 0-3
        result = differential_evolution(
            scalarized_objective,
            bounds,
            maxiter=50,
            seed=123,
            atol=0.1,
            tol=0.1
        )
    
        # Optimal solution
        angle_map = {0: 0, 1: 45, 2: -45, 3: 90}
        best_layup = [angle_map[int(round(x))] for x in result.x]
        mass, inv_sf = multi_objective_function(result.x, cfrp, t_ply, target_Nx, target_Ny)
    
        pareto_solutions.append({
            'weight': w,
            'layup': best_layup,
            'mass': mass,
            'safety_factor': 1 / inv_sf
        })
    
        print(f"Weight w={w:.1f}: Mass={mass:.3f} mm, SF={1/inv_sf:.2f}, Layup={best_layup}")
    
    # Visualize Pareto front
    masses = [sol['mass'] for sol in pareto_solutions]
    sfs = [sol['safety_factor'] for sol in pareto_solutions]
    
    plt.figure(figsize=(10, 6))
    plt.plot(masses, sfs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Thickness [mm]')
    plt.ylabel('Safety Factor')
    plt.title('Pareto Frontier (Weight vs Safety Factor)')
    plt.grid(True, alpha=0.3)
    
    # Label each point
    for i, sol in enumerate(pareto_solutions[::2]):  # Display every other
        plt.annotate(f"w={sol['weight']:.1f}",
                     (sol['mass'], sol['safety_factor']),
                     textcoords="offset points", xytext=(5,5), fontsize=8)
    
    plt.tight_layout()
    plt.savefig('pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()

## 5.3 Finite Element Method Preprocessing

### 5.3.1 Mesh Generation

In finite element analysis of composite materials, each layer is treated as separate elements or as integration points of shell elements. 

#### Example 5.5: Rectangular Plate Mesh Generation and Abaqus Input File Creation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    class CompositeMesh:
        """Mesh generation for composite material plates"""
    
        def __init__(self, length: float, width: float,
                     nx: int, ny: int, laminate: Laminate):
            """
            Parameters:
            -----------
            length, width : float
                Plate dimensions [mm]
            nx, ny : int
                Element divisions
            laminate : Laminate
                Laminate object
            """
            self.length = length
            self.width = width
            self.nx = nx
            self.ny = ny
            self.laminate = laminate
    
            self.nodes = []
            self.elements = []
    
            self._generate_mesh()
    
        def _generate_mesh(self):
            """Generate nodes and elements"""
            # Node generation
            dx = self.length / self.nx
            dy = self.width / self.ny
    
            node_id = 1
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    x = i * dx
                    y = j * dy
                    self.nodes.append({
                        'id': node_id,
                        'x': x,
                        'y': y,
                        'z': 0
                    })
                    node_id += 1
    
            # Element generation (4-node shell elements)
            elem_id = 1
            for j in range(self.ny):
                for i in range(self.nx):
                    n1 = j * (self.nx + 1) + i + 1
                    n2 = n1 + 1
                    n3 = n1 + (self.nx + 1) + 1
                    n4 = n1 + (self.nx + 1)
    
                    self.elements.append({
                        'id': elem_id,
                        'nodes': [n1, n2, n3, n4]
                    })
                    elem_id += 1
    
        def export_abaqus_inp(self, filename: str):
            """Export Abaqus input file"""
            with open(filename, 'w') as f:
                f.write("*HEADING\n")
                f.write("Composite Laminate Mesh\n")
    
                # Nodes
                f.write("*NODE\n")
                for node in self.nodes:
                    f.write(f"{node['id']}, {node['x']:.4f}, {node['y']:.4f}, {node['z']:.4f}\n")
    
                # Elements
                f.write("*ELEMENT, TYPE=S4R, ELSET=PLATE\n")
                for elem in self.elements:
                    nodes_str = ", ".join(map(str, elem['nodes']))
                    f.write(f"{elem['id']}, {nodes_str}\n")
    
                # Shell section
                f.write("*SHELL SECTION, ELSET=PLATE, COMPOSITE\n")
                for k, angle in enumerate(self.laminate.layup):
                    # Thickness, integration points, material name, angle
                    f.write(f"{self.laminate.t_ply}, 3, MAT1, {angle}\n")
    
                # Material properties
                mat = self.laminate.material
                f.write("*MATERIAL, NAME=MAT1\n")
                f.write("*ELASTIC, TYPE=LAMINA\n")
                f.write(f"{mat.E1*1000}, {mat.E2*1000}, {mat.nu12}, "
                        f"{mat.G12*1000}, {mat.G12*1000}, {mat.G12*1000}\n")
    
                # Boundary conditions (simply supported)
                f.write("*BOUNDARY\n")
                # Left edge (x=0): UX=0
                for node in self.nodes:
                    if abs(node['x']) < 1e-6:
                        f.write(f"{node['id']}, 1\n")
    
                # Bottom edge (y=0): UY=0
                for node in self.nodes:
                    if abs(node['y']) < 1e-6:
                        f.write(f"{node['id']}, 2\n")
    
                # Load step
                f.write("*STEP\n")
                f.write("*STATIC\n")
    
                # Distributed load (pressure on top surface)
                f.write("*DLOAD\n")
                for elem in self.elements:
                    f.write(f"{elem['id']}, P, 0.1\n")  # 0.1 MPa
    
                f.write("*OUTPUT, FIELD\n")
                f.write("*NODE OUTPUT\n")
                f.write("U, RF\n")
                f.write("*ELEMENT OUTPUT\n")
                f.write("S, E\n")
                f.write("*END STEP\n")
    
            print(f"Abaqus input file exported: {filename}")
    
    # Mesh generation example
    cfrp = Material(
        name="T300/Epoxy",
        E1=140.0, E2=10.0, nu12=0.30, G12=5.0,
        Xt=1500, Xc=1200, Yt=50, Yc=200, S=70
    )
    
    layup = [0, 45, -45, 90, 90, -45, 45, 0]
    lam = Laminate(cfrp, layup, t_ply=0.125)
    
    # Rectangular plate mesh
    mesh = CompositeMesh(length=100, width=100, nx=10, ny=10, laminate=lam)
    
    # Export Abaqus input file
    mesh.export_abaqus_inp("composite_plate.inp")
    
    print(f"Generated Mesh Information:")
    print(f"  Number of Nodes: {len(mesh.nodes)}")
    print(f"  Number of Elements: {len(mesh.elements)}")
    print(f"  Stacking Sequence: {layup}")
    print(f"  Total Thickness: {lam.total_thickness} mm")

### 5.3.2 Post-Processing and Data Visualization

Automate reading and visualization of FEA results. 

#### Example 5.6: Visualization of Stress Distribution
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    
    def visualize_stress_distribution(mesh: CompositeMesh, stress_values: np.ndarray,
                                        component: str = 'Sxx', cmap: str = 'jet'):
        """
        Visualize stress distribution on mesh
    
        Parameters:
        -----------
        mesh : CompositeMesh
            Mesh object
        stress_values : ndarray
            Stress value for each element [MPa]
        component : str
            Stress component name
        cmap : str
            Colormap
        """
        fig, ax = plt.subplots(figsize=(10, 8))
    
        patches = []
        colors = []
    
        for elem, stress in zip(mesh.elements, stress_values):
            # Get coordinates of 4 nodes of element
            node_ids = elem['nodes']
            coords = np.array([[mesh.nodes[nid-1]['x'], mesh.nodes[nid-1]['y']]
                               for nid in node_ids])
    
            # Create rectangular patch
            x_min, y_min = coords.min(axis=0)
            width = coords[:, 0].max() - x_min
            height = coords[:, 1].max() - y_min
    
            rect = Rectangle((x_min, y_min), width, height)
            patches.append(rect)
            colors.append(stress)
    
        # Patch collection
        p = PatchCollection(patches, cmap=cmap, edgecolors='black', linewidths=0.5)
        p.set_array(np.array(colors))
    
        ax.add_collection(p)
    
        # Colorbar
        cbar = plt.colorbar(p, ax=ax)
        cbar.set_label(f'{component} [MPa]', fontsize=12)
    
        ax.set_xlim(0, mesh.length)
        ax.set_ylim(0, mesh.width)
        ax.set_aspect('equal')
        ax.set_xlabel('X [mm]', fontsize=12)
        ax.set_ylabel('Y [mm]', fontsize=12)
        ax.set_title(f'Stress Distribution: {component}', fontsize=14, weight='bold')
    
        plt.tight_layout()
        plt.savefig(f'stress_{component}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate simulated stress data
    n_elements = len(mesh.elements)
    
    # Simulate distribution with higher stress at plate center
    stress_Sxx = []
    for elem in mesh.elements:
        node_ids = elem['nodes']
        x_center = np.mean([mesh.nodes[nid-1]['x'] for nid in node_ids])
        y_center = np.mean([mesh.nodes[nid-1]['y'] for nid in node_ids])
    
        # Stress based on distance from plate center (50, 50)
        dist = np.sqrt((x_center - 50)**2 + (y_center - 50)**2)
        stress = 100 * np.exp(-dist / 30)  # Gaussian distribution
    
        stress_Sxx.append(stress)
    
    stress_Sxx = np.array(stress_Sxx)
    
    # Visualization
    visualize_stress_distribution(mesh, stress_Sxx, component='Sxx', cmap='jet')
    print("Stress distribution diagram exported: stress_Sxx.png")

## 5.4 Summary

In this chapter, we learned practical implementation of composite material analysis using Python:

  * Complete implementation of Classical Lamination Theory (CLT) with object-oriented design
  * Stress analysis and First Ply Failure prediction
  * Optimal stacking design using genetic algorithms
  * Multi-objective optimization and Pareto frontier
  * Finite element method preprocessing (mesh generation, Abaqus input)
  * Result visualization and post-processing

By combining these techniques, professional-level composite material design and analysis becomes possible. For advanced study, we recommend engaging with damage mechanics, probabilistic design, and multiscale analysis. 

## Exercises

### Basic Level

#### Problem 5.1: CLT Library Extension

Add the following methods to the Laminate class: 

  * effective_bending_stiffness() to calculate bending stiffness (per unit thickness)
  * thermal_stress() method considering thermal expansion coefficient

#### Problem 5.2: Plot Function Implementation

Implement a plot_through_thickness_stress() method in the Laminate class that plots stress distribution through the thickness for each layer. 

#### Problem 5.3: Data Export

Implement an export_to_csv() method that exports analysis results to a CSV file. Output items: Layer number, angle, z-coordinate, σ1, σ2, τ12, FI, SF 

### Application Level

#### Problem 5.4: Buckling Analysis

Implement a buckling_load() method that calculates buckling load of a laminate. Solve the buckling eigenvalue problem for a simply supported rectangular plate. 

#### Problem 5.5: Optimization Extension

Add the following constraints to the genetic algorithm: 

  * Maximum 2 consecutive layers of same angle
  * Minimum 20% of 0° layers
  * Maintain symmetric laminate

#### Problem 5.6: Parametric Study

Create a program that visualizes the trade-off between in-plane stiffness and weight of laminates for fiber volume fraction V_f = 0.4-0.7. 

#### Problem 5.7: User Interface

Create a simple GUI using tkinter that interactively inputs stacking sequences and immediately displays properties. 

### Advanced Level

#### Problem 5.8: Damage Progression Simulation

Implement Progressive Failure Analysis: 

  * First Ply Failure detection
  * Failed layer stiffness degradation (Degradation Model)
  * Load redistribution and re-analysis
  * Loop until Last Ply Failure

#### Problem 5.9: Multiscale Analysis

Implement homogenization method from microscale (fiber-matrix) to macroscale (laminate). Perform RVE analysis by finite element method and extract equivalent single-layer properties. 

#### Problem 5.10: Integration with Machine Learning

Build the following machine learning model: 

  * Input: Stacking sequence (one-hot encoding)
  * Output: Ex, Ey, Gxy, First Ply Failure load
  * Training data: Generate 1000 samples by CLT analysis
  * Model: Neural network (scikit-learn/TensorFlow)
  * Evaluation: R² score, prediction error visualization

## References

  1. Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells: Theory and Analysis", 2nd ed., CRC Press, 2003, pp. 456-534
  2. Kaw, A. K., "Mechanics of Composite Materials", 2nd ed., CRC Press, 2005, pp. 312-389
  3. Goldberg, D. E., "Genetic Algorithms in Search, Optimization, and Machine Learning", Addison-Wesley, 1989, pp. 1-89
  4. Deb, K., "Multi-Objective Optimization Using Evolutionary Algorithms", Wiley, 2001, pp. 234-312
  5. Liu, B., Haftka, R. T., and Akgun, M. A., "Two-level Composite Wing Structural Optimization Using Response Surfaces", Structural and Multidisciplinary Optimization, Vol. 20, 2000, pp. 87-96
  6. Simulia, "Abaqus Analysis User's Guide: Composite Materials", Dassault Systèmes, 2020, pp. 23.1.1-23.6.8
  7. Bathe, K. J., "Finite Element Procedures", Prentice Hall, 1996, pp. 634-712
  8. Hunter, J. D., "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, Vol. 9, 2007, pp. 90-95

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
