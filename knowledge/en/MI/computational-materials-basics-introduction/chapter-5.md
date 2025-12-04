---
title: "Chapter 5: Integration of First-Principles Calculations and Machine Learning"
chapter_title: "Chapter 5: Integration of First-Principles Calculations and Machine Learning"
subtitle: Machine Learning Potential and Active Learning
reading_time: 20-25 minutes
difficulty: Advanced
code_examples: 6
exercises: 0
version: 1.0
created_at: "by:"
---

# Chapter 5: Integration of First-Principles Calculations and Machine Learning

This chapter demonstrates the minimal execution path using VASP/Quantum ESPRESSO/LAMMPS. Standard pre- and post-processing tools are also overviewed in a comprehensive list.

**💡 Note:** The most efficient approach is to first run the pipeline on a small system to familiarize yourself with input/output formats and unit systems.

## Learning Objectives

By reading this chapter, you will be able to: \- Understand the basic concepts of Machine Learning Potentials (MLP) \- Explain the differences and usage of Classical MD, AIMD, and MLP \- Train neural network potentials from DFT calculation data \- Understand efficient data generation strategies using Active Learning \- Grasp the latest trends in Universal MLP and Foundation Models

* * *

## 5.1 Why Machine Learning Potentials Are Needed

### Comparison of Three Computational Methods
    
    
    ```mermaid
    flowchart LR
        A[Classical MD] -->|Accuracy vs Speed| B[AIMD]
        B -->|Accuracy vs Speed| C[Machine Learning Potential]
        C - Data-driven .-> B
    
        style A fill:#ffcccc
        style B fill:#ccffcc
        style C fill:#ccccff
    ```

Item | Classical MD | AIMD (DFT-MD) | MLP-MD  
---|---|---|---  
**Force Calculation** | Empirical force field | DFT (First-principles) | Machine learning model  
**Accuracy** | Medium (force field dependent) | High (quantum mechanical) | High (DFT-equivalent)  
**Computational Speed** | Very fast (ns/day) | Very slow (ps/day) | Fast (ns/day)  
**System Size** | Millions of atoms | Hundreds of atoms | Thousands to tens of thousands of atoms  
**Applicability** | Trained systems only | General-purpose | Within training data range  
**Development Cost** | Low (using existing force fields) | None | High (training data generation)  
  
### Advantages of MLP

**"DFT-level accuracy at Classical MD speed"**

  * ✅ Accurately describes chemical reactions (bond breaking/formation)
  * ✅ Long-time simulations (ns-μs scale)
  * ✅ Large-scale systems (thousands to tens of thousands of atoms)
  * ✅ Applicable to novel materials without existing force fields

**Challenges** : \- ❌ Cost of generating training data (DFT calculations) \- ❌ Accuracy degrades outside training data range \- ❌ Model training requires computational resources and expertise

* * *

## 5.2 Types of Machine Learning Potentials

### 1\. Gaussian Approximation Potential (GAP)

**Principle** : Kernel method (Gaussian Process)

$$ E_{\text{GAP}}(\mathbf{R}) = \sum_{i=1}^N \alpha_i K(\mathbf{R}, \mathbf{R}_i) $$

  * $K$: Kernel function (measures similarity)
  * $\mathbf{R}_i$: Atomic configuration in training data
  * $\alpha_i$: Training parameters

**Features** : \- ✅ Uncertainty estimation possible (advantageous for Active Learning) \- ✅ Can learn from limited data \- ❌ Computational cost increases with number of training data points

### 2\. Neural Network Potential (NNP)

**Behler-Parrinello type** : Describes local environment of each atom

$$ E_{\text{NNP}} = \sum_{i=1}^{N_{\text{atoms}}} E_i^{\text{NN}}({\mathbf{G}_i}) $$

  * $E_i^{\text{NN}}$: Neural network energy of atom $i$
  * $\mathbf{G}_i$: Symmetry Functions, describing the environment around atom $i$

**Example of symmetry functions** (radial component):

$$ G_i^{\text{rad}} = \sum_{j \neq i} e^{-\eta(r_{ij} - R_s)^2} f_c(r_{ij}) $$

  * $r_{ij}$: Interatomic distance
  * $f_c(r)$: Cutoff function (ignores beyond a certain distance)

**Features** : \- ✅ Fast even for large-scale systems \- ✅ Computational cost constant regardless of training data size \- ❌ Uncertainty estimation difficult

### 3\. Message Passing Neural Network (MPNN)

A type of Graph Neural Network (GNN):

$$ \mathbf{h}_i^{(k+1)} = \text{Update}\left(\mathbf{h}_i^{(k)}, \sum_{j \in \mathcal{N}(i)} \text{Message}(\mathbf{h}_i^{(k)}, \mathbf{h}_j^{(k)}, \mathbf{e}_{ij})\right) $$

  * $\mathbf{h}_i^{(k)}$: Hidden state of atom $i$ at layer $k$
  * $\mathcal{N}(i)$: Neighboring atoms of atom $i$
  * $\mathbf{e}_{ij}$: Bond information (distance, angle)

**Representative models** : SchNet, DimeNet, GemNet, MACE

**Features** : \- ✅ Naturally realizes rotation and translation invariance \- ✅ Efficiently learns long-range interactions \- ✅ State-of-the-art high-accuracy models

### 4\. Moment Tensor Potential (MTP)

**Principle** : Describes atomic environment with many-body expansion

$$ E_{\text{MTP}} = \sum_i \sum_{\alpha} c_{\alpha} B_{\alpha}(\mathbf{R}_i) $$

$B_{\alpha}$ are moment tensor basis functions.

**Features** : \- ✅ Fast (linear model) \- ✅ Easy to train \- ❌ Lower expressiveness than NNP

* * *

## 5.3 Neural Network Potential Training (Practice)

### Example 1: NNP Training Using AMP (Water Molecule)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from ase.build import molecule
    from ase.calculators.emt import EMT
    from gpaw import GPAW, PW
    from amp import Amp
    from amp.descriptor.gaussian import Gaussian
    from amp.model.neuralnetwork import NeuralNetwork
    import matplotlib.pyplot as plt
    
    # Step 1: Generate training data (MD simulation + DFT)
    def generate_training_data(n_samples=50):
        """
        DFT calculations for various water molecule configurations
        """
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase import units
    
        h2o = molecule('H2O')
        h2o.center(vacuum=5.0)
    
        # DFT calculator
        calc = GPAW(mode=PW(300), xc='PBE', txt=None)
        h2o.calc = calc
    
        # Initial velocity
        MaxwellBoltzmannDistribution(h2o, temperature_K=500)
    
        # MD simulation
        dyn = VelocityVerlet(h2o, timestep=1.0*units.fs)
    
        images = []
        for i in range(n_samples):
            dyn.run(10)  # Sample every 10 steps
            atoms_copy = h2o.copy()
            atoms_copy.calc = calc
            atoms_copy.get_potential_energy()  # Execute DFT calculation
            atoms_copy.get_forces()
            images.append(atoms_copy)
            print(f"Sample {i+1}/{n_samples} collected")
    
        return images
    
    print("Generating training data...")
    train_images = generate_training_data(n_samples=50)
    
    # Step 2: NNP training
    print("Training Neural Network Potential...")
    
    # Descriptor: Gaussian symmetry functions
    descriptor = Gaussian()
    
    # Model: Neural network
    model = NeuralNetwork(hiddenlayers=(10, 10, 10))  # 3 layers, 10 nodes each
    
    # AMP potential
    calc_nnp = Amp(descriptor=descriptor,
                   model=model,
                   label='h2o_nnp',
                   dblabel='h2o_nnp')
    
    # Training
    calc_nnp.train(images=train_images,
                   energy_coefficient=1.0,
                   force_coefficient=0.04)
    
    print("Training complete!")
    
    # Step 3: Accuracy evaluation on test data
    print("\nGenerating test data...")
    test_images = generate_training_data(n_samples=10)
    
    E_dft = []
    E_nnp = []
    F_dft = []
    F_nnp = []
    
    for atoms in test_images:
        # DFT
        atoms.calc = GPAW(mode=PW(300), xc='PBE', txt=None)
        e_dft = atoms.get_potential_energy()
        f_dft = atoms.get_forces().flatten()
    
        # NNP
        atoms.calc = calc_nnp
        e_nnp = atoms.get_potential_energy()
        f_nnp = atoms.get_forces().flatten()
    
        E_dft.append(e_dft)
        E_nnp.append(e_nnp)
        F_dft.extend(f_dft)
        F_nnp.extend(f_nnp)
    
    E_dft = np.array(E_dft)
    E_nnp = np.array(E_nnp)
    F_dft = np.array(F_dft)
    F_nnp = np.array(F_nnp)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy
    axes[0].scatter(E_dft, E_nnp, alpha=0.6)
    axes[0].plot([E_dft.min(), E_dft.max()],
                 [E_dft.min(), E_dft.max()], 'r--', label='Perfect')
    axes[0].set_xlabel('DFT Energy (eV)', fontsize=12)
    axes[0].set_ylabel('NNP Energy (eV)', fontsize=12)
    axes[0].set_title('Energy Prediction', fontsize=14)
    mae_e = np.mean(np.abs(E_dft - E_nnp))
    axes[0].text(0.05, 0.95, f'MAE = {mae_e:.3f} eV',
                transform=axes[0].transAxes, va='top')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Force
    axes[1].scatter(F_dft, F_nnp, alpha=0.3, s=10)
    axes[1].plot([F_dft.min(), F_dft.max()],
                 [F_dft.min(), F_dft.max()], 'r--', label='Perfect')
    axes[1].set_xlabel('DFT Force (eV/Å)', fontsize=12)
    axes[1].set_ylabel('NNP Force (eV/Å)', fontsize=12)
    axes[1].set_title('Force Prediction', fontsize=14)
    mae_f = np.mean(np.abs(F_dft - F_nnp))
    axes[1].text(0.05, 0.95, f'MAE = {mae_f:.3f} eV/Å',
                transform=axes[1].transAxes, va='top')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nnp_accuracy.png', dpi=150)
    plt.show()
    
    print(f"\nNNP Accuracy:")
    print(f"Energy MAE: {mae_e:.4f} eV")
    print(f"Force MAE: {mae_f:.4f} eV/Å")
    

**Target accuracy** : \- Energy: MAE < 1 meV/atom \- Force: MAE < 0.1 eV/Å

* * *

## 5.4 Active Learning

### Basic Concept

**Problem** : Performing DFT calculations for all configurations is impractical (high computational cost)

**Solution** : **Prioritize sampling configurations with the most information**
    
    
    ```mermaid
    flowchart TD
        A[Small initial dataset] --> B[NNP training]
        B --> C[MD simulation with NNP]
        C --> D[Detect high-uncertainty configurations]
        D --> E{Additional data needed?}
        E -->|Yes| F[Additional DFT calculations]
        F --> G[Add to dataset]
        G --> B
        E -->|No| H[Training complete]
    
        style A fill:#e3f2fd
        style H fill:#c8e6c9
    ```

### Methods for Uncertainty Estimation

**1\. Ensemble Method** : \- Train multiple NNPs (different initial values, data splits) \- Use prediction variance as uncertainty

$$ \sigma_E^2 = \frac{1}{M}\sum_{m=1}^M (E_m - \bar{E})^2 $$

**2\. Dropout Method** : \- Randomly disable nodes during training \- Apply dropout during inference as well, predict multiple times \- Use prediction variance as uncertainty

**3\. Query-by-Committee** : \- Use multiple models with different algorithms \- Sample configurations with low agreement in predictions

### Active Learning Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from ase.md.langevin import Langevin
    from ase import units
    
    def active_learning_loop(initial_images, n_iterations=5, n_md_steps=1000):
        """
        Efficient training data generation using Active Learning
        """
        dataset = initial_images.copy()
    
        for iteration in range(n_iterations):
            print(f"\n--- Iteration {iteration+1}/{n_iterations} ---")
    
            # Step 1: Train NNP
            print("Training NNP...")
            nnp = train_nnp(dataset)  # AMP training as described above
    
            # Step 2: MD simulation with NNP
            print("Running MD with NNP...")
            h2o = dataset[0].copy()
            h2o.calc = nnp
    
            # Langevin MD (with thermostat)
            dyn = Langevin(h2o, timestep=1.0*units.fs,
                           temperature_K=500, friction=0.01)
    
            # Collect high-uncertainty configurations
            uncertain_images = []
            uncertainties = []
    
            for step in range(n_md_steps):
                dyn.run(1)
    
                # Uncertainty estimation with Ensemble (simplified)
                # In practice, calculate variance with multiple NNPs
                uncertainty = estimate_uncertainty(h2o, nnp)  # Virtual function
    
                if uncertainty > threshold:  # Add if above threshold
                    atoms_copy = h2o.copy()
                    uncertain_images.append(atoms_copy)
                    uncertainties.append(uncertainty)
    
            print(f"Found {len(uncertain_images)} uncertain configurations")
    
            # Step 3: DFT calculations for high-uncertainty configurations
            print("Running DFT for uncertain configurations...")
            for atoms in uncertain_images[:10]:  # Top 10
                atoms.calc = GPAW(mode=PW(300), xc='PBE', txt=None)
                atoms.get_potential_energy()
                atoms.get_forces()
                dataset.append(atoms)
    
            print(f"Dataset size: {len(dataset)}")
    
        return dataset, nnp
    
    # Execution
    initial_data = generate_training_data(n_samples=20)
    final_dataset, final_nnp = active_learning_loop(initial_data, n_iterations=5)
    
    print(f"\nFinal dataset size: {len(final_dataset)}")
    print(f"vs. random sampling: 50-100 samples would be needed")
    print(f"Efficiency gain: {100/len(final_dataset):.1f}x")
    

**Advantages of Active Learning** : \- Can reduce training data by 50-90% \- Prioritizes important configurations (phase transitions, reaction pathways) \- Efficient use of computational resources

* * *

## 5.5 Latest Trends

### 1\. Universal Machine Learning Potential

**Goal** : One model covering diverse material systems

**Representative examples** : \- **CHGNet** (2023): Trained on 1.4 million materials from Materials Project \- Covers 89 elements \- Includes magnetism \- Open source

  * **M3GNet** (2022): Many-body graph network
  * Applicable to crystals, surfaces, molecules
  * Predicts forces, stresses, magnetic moments

  * **MACE** (2023): Equivariant message passing

  * High accuracy (approximately twice DFT error)
  * Can train on small-scale data

**Usage** :
    
    
    from chgnet.model import CHGNet
    from pymatgen.core import Structure
    
    # Load pre-trained model
    model = CHGNet.load()
    
    # Predict for arbitrary crystal structure
    structure = Structure.from_file('POSCAR')
    energy = model.predict_structure(structure)
    
    print(f"Predicted energy: {energy} eV")
    

### 2\. Foundation Models for Materials

**Materials science version of Large Language Models (LLM)** :

  * **MatGPT** : Pre-trained on materials databases
  * **LLaMat** : Crystal structure → property prediction

**Transfer learning** : \- Pre-training on large-scale data \- Fine-tuning with limited data \- Practical accuracy with 10-100 samples

### 3\. Application to Autonomous Experiments

**Closed-loop optimization** :
    
    
    ML prediction → Optimal candidate proposal → Robot experiment → Measurement → Data accumulation → ML retraining
    

**Real examples** : \- **A-Lab** (Berkeley, 2023): Synthesized and evaluated 41 materials in 17 days \- **Autonomous materials discovery** : Catalysts, battery materials, quantum dots

* * *

## 5.6 Practical Guidelines for MLP

### When to Use MLP

**Suitable cases** : \- ✅ Long-time MD (ns-μs) required \- ✅ Large-scale systems (thousands of atoms or more) \- ✅ Includes chemical reactions \- ✅ Novel materials without existing force fields \- ✅ Computational resources available for training data generation

**Unsuitable cases** : \- ❌ One-time short MD (direct AIMD is simpler) \- ❌ Cannot ensure representativeness of training data \- ❌ Extrapolation beyond training data range required \- ❌ Existing high-accuracy force fields available (ReaxFF, COMB, etc.)

### Implementation Workflow
    
    
    ```mermaid
    flowchart TD
        A[Problem definition] --> B[Initial data generation 20-100 samples]
        B --> C[NNP training]
        C --> D[Accuracy evaluation on validation set]
        D --> E{Accuracy OK?}
        E -->|No| F[Active Learning]
        F --> G[Additional DFT calculations]
        G --> C
        E -->|Yes| H[Production MD simulation]
        H --> I[Property calculations]
    
        style A fill:#e3f2fd
        style I fill:#c8e6c9
    ```

### Recommended Tools

Tool | Method | Features  
---|---|---  
**AMP** | NNP | Python native, ASE integration  
**DeePMD** | NNP | Fast, parallelized, TensorFlow  
**SchNetPack** | GNN | SchNet, research-oriented  
**MACE** | Equivariant GNN | Latest, high accuracy  
**GAP** | Gaussian Process | Uncertainty estimation  
**MTP** | Moment Tensor | Fast training  
**CHGNet** | Universal | Pre-trained  
  
* * *

## 5.7 Chapter Summary

### What We Learned

  1. **Need for MLP** \- DFT-level accuracy + Classical MD speed \- Long-time and large-scale system simulations

  2. **Types of MLP** \- GAP (Gaussian Process) \- NNP (Neural Network) \- MPNN (Graph Neural Network) \- MTP (Moment Tensor)

  3. **NNP Training** \- DFT data generation \- Implementation with AMP \- Accuracy evaluation

  4. **Active Learning** \- Uncertainty estimation \- Efficient data generation \- 50-90% computational reduction

  5. **Latest Trends** \- Universal MLP (CHGNet, M3GNet) \- Foundation Models \- Autonomous experiments

### Key Points

  * MLP is a new paradigm in computational materials science
  * Active Learning is the key to training efficiency
  * Pre-trained models available through Universal MLP
  * Practical applications are advancing (autonomous experiments, materials discovery)

### Next Steps

  * Try MLP on your own research topics
  * Follow latest papers (_npj Computational Materials_ , _Nature Materials_)
  * Contribute to open-source tools
  * Collaborate with experimental researchers

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Summarize the differences between Classical MD, AIMD, and MLP-MD in a table.

Sample Answer | Item | Classical MD | AIMD (DFT-MD) | MLP-MD | |-----|-------------|-------------|--------| | **Force Calculation Method** | Empirical force field (analytical) | DFT (First-principles) | Machine learning model | | **Accuracy** | Medium (depends on force field quality) | High (quantum mechanically accurate) | High (DFT-equivalent) | | **Computational Speed** | Very fast (1 ns/day) | Very slow (10 ps/day) | Fast (1 ns/day) | | **System Size** | Millions of atoms | Hundreds of atoms | Thousands to tens of thousands of atoms | | **Chemical Reactions** | Not described (ReaxFF can) | Accurately described | Accurately described | | **Applicability** | Only systems with trained force fields | General-purpose | Within training data range | | **Development Cost** | Low (existing force fields) | None | High (training data generation) | | **Applications** | Diffusion, phase transitions, large-scale | Chemical reactions, electronic states | Reactions + long-time MD | **Guidelines for selection**: \- Existing force field available → Classical MD \- Chemical reactions with short time → AIMD \- Chemical reactions + long time → MLP-MD \- Novel materials discovery → AIMD → MLP → large-scale MD 

### Problem 2 (Difficulty: medium)

Explain why Active Learning is efficient, with specific examples.

Sample Answer **Basic Principle of Active Learning**: Traditional machine learning (Random Sampling): \- Randomly sample data \- Many data points are duplicates of "known regions" \- Inefficient Active Learning (Uncertainty Sampling): \- Prioritize sampling configurations where model is "uncertain" \- Efficiently acquire new information \- High accuracy with less data **Specific Example: NNP Training for Water Molecules** **Random Sampling (Traditional method)**: \- Randomly sample 100 configurations from 300K equilibrium state \- 80% are near equilibrium structure (similar configurations) \- Remaining 20% are reaction pathways or high-energy configurations \- Result: 100 DFT calculations, accuracy MAE = 5 meV/atom **Active Learning**: \- Train from initial 20 configurations \- Detect high-uncertainty configurations during MD simulation \- Configurations with extended O-H bonds (dissociation process) \- Configurations with highly distorted H-O-H angles \- High-energy excited states \- DFT calculations for these configurations (20 additional configurations) \- Total 40 DFT calculations, accuracy MAE = 3 meV/atom **Reasons for Efficiency**: 1\. **Maximize information content**: \- Avoid duplicates of similar configurations \- Prioritize regions model "doesn't know" 2\. **Balance exploration and exploitation**: \- Stable predictions in known configurations (exploitation) \- Acquire new information in unknown configurations (exploration) 3\. **Adaptive sampling**: \- Automatically detect important regions (reaction pathways, phase transitions) \- Does not rely on human intuition **Actual efficiency gains**: \- 50-90% DFT calculation reduction (literature values) \- Particularly effective for complex systems (multi-component, reactive systems) \- 10-50x efficiency gain in total training time **Example: Li-ion battery electrolyte**: \- Random: 10,000 DFT calculations, 2 months \- Active Learning: 2,000 DFT calculations, 2 weeks \- Efficiency: 5x, equivalent accuracy 

### Problem 3 (Difficulty: hard)

Discuss the advantages and limitations of Universal Machine Learning Potentials (CHGNet, M3GNet, etc.).

Sample Answer **Overview of Universal MLP (Example: CHGNet)**: \- **Training data**: Materials Project (1.4 million materials, 89 elements) \- **Model**: Graph Neural Network \- **Predictions**: Energy, forces, stresses, magnetic moments **Advantages**: 1\. **Immediately usable**: \- Pre-trained → No additional training needed \- Predictable for arbitrary crystal structures \- Screen thousands of materials in seconds 2\. **Wide applicability**: \- 89 elements (H to Am) \- Oxides, alloys, semiconductors, insulators \- Magnetic materials also supported 3\. **Foundation for transfer learning**: \- Fine-tuning with limited data (10-100 samples) \- Efficiently create system-specific high-accuracy models 4\. **Accelerate materials discovery**: \- Large-scale candidate screening (1 million materials/day) \- Narrow down experimental candidates \- Combination with high-throughput calculations **Limitations**: 1\. **Accuracy limits**: \- Approximately 2-5 times DFT error (CHGNet: MAE ~30 meV/atom) \- Insufficient for precision calculations \- Inferior to dedicated MLP for specific systems 2\. **Extrapolation problems**: \- Accuracy degrades for configurations not in training data (extreme temperature/pressure) \- Uncertain for novel material systems (ultra-high pressure, new element combinations) 3\. **Data bias**: \- Depends on Materials Project calculation conditions (PBE functional) \- Systematic deviations from experiments (band gap underestimation, etc.) \- Over/under-representation of specific material classes 4\. **Lack of physical constraints**: \- No strict guarantee of energy conservation \- Drift in long-time MD \- Symmetry breaking (rare) **Practical Strategies**: **Scenario 1: Materials Screening** \- Narrow down from 1 million candidates to top 1000 with Universal MLP \- Precision calculations with DFT \- Efficiency: 1000x **Scenario 2: Precision MD for Specific Systems** \- Transfer learning from Universal MLP \- Additional training with system-specific data (100 samples) \- Accuracy improvement: MAE 5 meV/atom (practical level) **Scenario 3: Novel Material Classes** \- Universal MLP as reference only \- Build dedicated MLP from scratch (Active Learning) \- Training data: 500-1000 samples **Future Outlook**: 1\. **Dataset expansion**: \- Integration of experimental data \- Data from diverse computational methods (GW, DMFT) 2\. **Evolution to Foundation Models**: \- Equivalent to GPT in natural language processing \- Few-shot learning (adapt with few samples) \- Zero-shot transfer (new systems without training) 3\. **Integration with experiments**: \- Autonomous experiment loops \- Real-time feedback **Conclusion**: Universal MLP is becoming "foundational infrastructure" for materials science, but is not omnipotent. Important to use dedicated MLP appropriately depending on application. 

* * *

## Data Licenses and Citations

### Datasets Used

  1. **Materials Project Database** (CC BY 4.0) \- DFT data for 1.4 million materials (CHGNet training) \- URL: https://materialsproject.org \- Citation: Jain, A., et al. (2013). _APL Materials_ , 1, 011002.

  2. **Open Catalyst Project** (CC BY 4.0) \- DFT dataset for catalyst surfaces \- URL: https://opencatalystproject.org/

  3. **QM9 Dataset** (CC0) \- DFT data for 134,000 small molecules \- URL: http://quantum-machine.org/datasets/

### Software Used

  1. **AMP - Atomistic Machine-learning Package** (GPL v3) \- URL: https://amp.readthedocs.io/

  2. **CHGNet** (MIT License) \- Universal ML Potential \- URL: https://github.com/CederGroupHub/chgnet

  3. **M3GNet** (BSD 3-Clause) \- Graph Neural Network Potential \- URL: https://github.com/materialsvirtuallab/m3gnet

  4. **MACE** (MIT License) \- Equivariant Message Passing \- URL: https://github.com/ACEsuit/mace

* * *

## Code Reproducibility Checklist

### Environment Setup
    
    
    # Basic MLP environment
    conda create -n mlp python=3.11
    conda activate mlp
    conda install pytorch torchvision -c pytorch
    conda install -c conda-forge ase gpaw
    
    # Individual MLP tools
    pip install amp-atomistics  # AMP
    pip install chgnet  # CHGNet
    pip install m3gnet  # M3GNet
    pip install mace-torch  # MACE
    

### GPU Requirements (Recommended)

Training Data Size | GPU Memory | Training Time | Recommended GPU  
---|---|---|---  
100 samples | ~2 GB | ~30 minutes | GTX 1060  
1,000 samples | ~8 GB | ~3 hours | RTX 3070  
10,000 samples | ~16 GB | ~1 day | RTX 3090/A100  
  
### Troubleshooting

**Problem** : CUDA out of memory **Solution** :
    
    
    # Reduce batch size
    model.train(batch_size=8)  # 32 → 8
    

**Problem** : Training does not converge **Solution** :
    
    
    # Adjust learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 1e-3 → 1e-4
    

* * *

## Practical Pitfalls and Countermeasures

### 1\. Training Data Bias
    
    
    # ❌ Wrong: Only equilibrium structures
    train_data = [equilibrium_structures]  # Range too narrow
    
    # ✅ Correct: Sample diverse configurations
    # - Equilibrium structures
    # - MD trajectories (high temperature)
    # - Mid-structure optimization
    # - High-energy configurations
    

### 2\. Inappropriate Force Weights
    
    
    # ❌ Imbalanced: Energy-only emphasis
    loss = energy_loss  # Ignores forces
    
    # ✅ Balanced: Forces also emphasized
    loss = energy_loss + 0.1 * force_loss  # Force weight 0.1
    

### 3\. Use in Extrapolation Region
    
    
    # ❌ Dangerous: Prediction outside training range
    # Training: 0-1000 K
    # Usage: 2000 K → Inaccurate
    
    # ✅ Safe: Use within training range
    # Or warn with uncertainty estimation
    

### 4\. Active Learning Threshold Setting
    
    
    # ❌ Threshold too high → Data shortage
    uncertainty_threshold = 10.0  # Too loose
    
    # ✅ Appropriate threshold
    uncertainty_threshold = 0.1  # Energy [eV/atom]
    

* * *

## Quality Assurance Checklist

### MLP Training Validity

  * [ ] Training error: Energy MAE < 10 meV/atom
  * [ ] Training error: Force MAE < 0.1 eV/Å
  * [ ] Test error within twice training error (no overfitting)
  * [ ] Stable performance on validation set

### Physical Validity

  * [ ] Energy conservation (verify with NVE MD)
  * [ ] Translation and rotation invariance
  * [ ] Symmetry conservation
  * [ ] No abnormal forces (> 10 eV/Å)

### Active Learning Efficiency

  * [ ] Training data reduction rate > 50%
  * [ ] Iterations to convergence < 10
  * [ ] Final accuracy equal to or better than random sampling

* * *

## References

  1. Behler, J., & Parrinello, M. (2007). "Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces." _Physical Review Letters_ , 98, 146401. DOI: [10.1103/PhysRevLett.98.146401](<https://doi.org/10.1103/PhysRevLett.98.146401>)

  2. Bartók, A. P., et al. (2010). "Gaussian Approximation Potentials: The Accuracy of Quantum Mechanics, without the Electrons." _Physical Review Letters_ , 104, 136403. DOI: [10.1103/PhysRevLett.104.136403](<https://doi.org/10.1103/PhysRevLett.104.136403>)

  3. Schütt, K. T., et al. (2017). "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions." _NeurIPS_.

  4. Chen, C., & Ong, S. P. (2022). "A universal graph deep learning interatomic potential for the periodic table." _Nature Computational Science_ , 2, 718-728. DOI: [10.1038/s43588-022-00349-3](<https://doi.org/10.1038/s43588-022-00349-3>)

  5. Batatia, I., et al. (2022). "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields." _NeurIPS_.

  6. CHGNet: https://github.com/CederGroupHub/chgnet

  7. M3GNet: https://github.com/materialsvirtuallab/m3gnet
  8. MACE: https://github.com/ACEsuit/mace

* * *

## Author Information

**Created by** : MI Knowledge Hub Content Team **Date Created** : 2025-10-17 **Version** : 1.0 **Series** : Computational Materials Basics Introduction v1.0

**License** : Creative Commons BY-NC-SA 4.0

* * *

**Congratulations! You have completed the Computational Materials Basics Introduction series!**

Next steps: \- Execute actual calculations on your own research topics \- Proceed to High-Throughput Computing Introduction series \- Deepen knowledge by reading latest papers \- Join the community (GitHub, conferences)

**Continuous learning opens the future of materials science!**
