---
title: "Chapter 3: Hands-On MLP with Python - SchNetPack Tutorial"
chapter_title: "Chapter 3: Hands-On MLP with Python - SchNetPack Tutorial"
---

# Chapter 3: Hands-On MLP with Python - SchNetPack Tutorial

Complete the entire training and evaluation workflow with a small dataset. Clarify checkpoints for reproducibility and overfitting prevention.

**💡 Supplement:** The basic three-piece set includes seed fixing, data splitting, and learning curve recording. Stabilize with early stopping and weight decay.

## Learning Objectives

By reading this chapter, you will master the following:  
\- Install SchNetPack in Python environment and set up the environment  
\- Train an MLP model using a small-scale dataset (aspirin molecule from MD17)  
\- Evaluate trained model accuracy and verify energy/force prediction errors  
\- Execute MLP-MD simulations and analyze trajectories  
\- Understand common errors and their solutions

* * *

## 3.1 Environment Setup: Installing Required Tools

To practice MLP, you need to set up a Python environment and SchNetPack.

### Required Software

Tool | Version | Purpose  
---|---|---  
**Python** | 3.9-3.11 | Base language  
**PyTorch** | 2.0+ | Deep learning framework  
**SchNetPack** | 2.0+ | MLP training and inference  
**ASE** | 3.22+ | Atomic structure manipulation, MD execution  
**NumPy/Matplotlib** | Latest | Data analysis and visualization  
  
### Installation Steps

**Step 1: Create Conda Environment**
    
    
    # Create new Conda environment (Python 3.10)
    conda create -n mlp-tutorial python=3.10 -y
    conda activate mlp-tutorial
    

**Step 2: Install PyTorch**
    
    
    # CPU version (local machine, lightweight)
    conda install pytorch cpuonly -c pytorch
    
    # GPU version (if CUDA available)
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    

**Step 3: Install SchNetPack and ASE**
    
    
    # SchNetPack (pip recommended)
    pip install schnetpack
    
    # ASE (Atomic Simulation Environment)
    pip install ase
    
    # Visualization tools
    pip install matplotlib seaborn
    

**Step 4: Verify Installation**
    
    
    # Example 1: Environment verification script (5 lines)
    import torch
    import schnetpack as spk
    print(f"PyTorch: {torch.__version__}")
    print(f"SchNetPack: {spk.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    

**Expected Output** :
    
    
    PyTorch: 2.1.0
    SchNetPack: 2.0.3
    GPU available: False  # For CPU
    

* * *

## 3.2 Data Preparation: Obtaining MD17 Dataset

SchNetPack includes the **MD17** benchmark dataset for small molecules.

### What is MD17 Dataset

  * **Content** : Molecular dynamics trajectories from DFT calculations
  * **Target molecules** : 10 types including aspirin, benzene, ethanol
  * **Data count** : Approximately 100,000 configurations per molecule
  * **Accuracy** : PBE/def2-SVP level (DFT)
  * **Purpose** : Benchmarking MLP methods

### Downloading and Loading Data

**Example 2: Loading MD17 Dataset (10 lines)**
    
    
    from schnetpack.datasets import MD17
    from schnetpack.data import AtomsDataModule
    
    # Download aspirin molecule dataset (approximately 100,000 configurations)
    dataset = MD17(
        datapath='./data',
        molecule='aspirin',
        download=True
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Properties: {dataset.available_properties}")
    print(f"First sample: {dataset[0]}")
    

**Output** :
    
    
    Total samples: 211762
    Properties: ['energy', 'forces']
    First sample: {'_atomic_numbers': tensor([...]), 'energy': tensor(-1234.5), 'forces': tensor([...])}
    

### Data Splitting

**Example 3: Splitting into Train/Validation/Test Sets (10 lines)**
    
    
    # Split data into train:val:test = 70%:15%:15%
    data_module = AtomsDataModule(
        datapath='./data',
        dataset=dataset,
        batch_size=32,
        num_train=100000,      # Number of training data
        num_val=10000,          # Number of validation data
        num_test=10000,         # Number of test data
        split_file='split.npz', # Save split information
    )
    data_module.prepare_data()
    data_module.setup()
    

**Explanation** :  
\- `batch_size=32`: Process 32 configurations at once (memory efficiency)  
\- `num_train=100000`: Large data improves generalization  
\- `split_file`: Save split to file (ensure reproducibility)

* * *

## 3.3 Model Training with SchNetPack

Train a SchNet model to learn energy and forces.

### SchNet Architecture Configuration

**Example 4: Defining SchNet Model (15 lines)**
    
    
    import schnetpack.transform as trn
    from schnetpack.representation import SchNet
    from schnetpack.model import AtomisticModel
    from schnetpack.task import ModelOutput
    
    # 1. SchNet representation layer (atomic configuration → feature vector)
    representation = SchNet(
        n_atom_basis=128,      # Dimension of atomic feature vectors
        n_interactions=6,      # Number of message passing layers
        cutoff=5.0,            # Cutoff radius (Å)
        n_filters=128          # Number of filters
    )
    
    # 2. Output layer (energy prediction)
    output = ModelOutput(
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        metrics={'MAE': spk.metrics.MeanAbsoluteError()}
    )
    

**Parameter Explanation** :  
\- `n_atom_basis=128`: Each atom's feature vector is 128-dimensional (typical value)  
\- `n_interactions=6`: 6 layers of message passing (deeper captures longer-range interactions)  
\- `cutoff=5.0Å`: Ignore atomic interactions beyond this distance (computational efficiency)

### Running Training

**Example 5: Setting Up Training Loop (15 lines)**
    
    
    import pytorch_lightning as pl
    from schnetpack.task import AtomisticTask
    
    # Define training task
    task = AtomisticTask(
        model=AtomisticModel(representation, [output]),
        outputs=[output],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={'lr': 1e-4}  # Learning rate
    )
    
    # Configure Trainer
    trainer = pl.Trainer(
        max_epochs=50,               # Maximum 50 epochs
        accelerator='cpu',           # Use CPU (GPU: 'gpu')
        devices=1,
        default_root_dir='./training'
    )
    
    # Start training
    trainer.fit(task, datamodule=data_module)
    

**Training Time Estimate** :  
\- CPU (4 cores): Approximately 2-3 hours (100,000 configurations)  
\- GPU (RTX 3090): Approximately 15-20 minutes

### Monitoring Training Progress

**Example 6: Visualization with TensorBoard (10 lines)**
    
    
    # Launch TensorBoard (separate terminal)
    # tensorboard --logdir=./training/lightning_logs
    
    # Check logs from Python
    import pandas as pd
    
    metrics = pd.read_csv('./training/lightning_logs/version_0/metrics.csv')
    print(metrics[['epoch', 'train_loss', 'val_loss']].tail(10))
    

**Expected Output** :
    
    
       epoch  train_loss  val_loss
    40    40      0.0023    0.0031
    41    41      0.0022    0.0030
    42    42      0.0021    0.0029
    ...
    

**Observation Points** :  
\- Both `train_loss` and `val_loss` decreasing → Normal learning progress  
\- `val_loss` starts increasing → Sign of **overfitting** → Consider Early Stopping

* * *

## 3.4 Accuracy Verification: Energy and Force Prediction Accuracy

Evaluate whether the trained model achieves DFT accuracy.

### Test Set Evaluation

**Example 7: Test Set Evaluation (12 lines)**
    
    
    # Evaluate on test set
    test_results = trainer.test(task, datamodule=data_module)
    
    # Display results
    print(f"Energy MAE: {test_results[0]['test_energy_MAE']:.4f} eV")
    print(f"Energy RMSE: {test_results[0]['test_energy_RMSE']:.4f} eV")
    
    # Force evaluation (requires separate calculation)
    from schnetpack.metrics import MeanAbsoluteError
    force_mae = MeanAbsoluteError(target='forces')
    # ... Force evaluation code
    

**Good Accuracy Benchmarks** (aspirin molecule, 21 atoms):  
\- **Energy MAE** : < 1 kcal/mol (< 0.043 eV)  
\- **Force MAE** : < 1 kcal/mol/Å (< 0.043 eV/Å)

### Correlation Plot of Predictions vs True Values

**Example 8: Visualizing Prediction Accuracy (15 lines)**
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Predict on test data
    model = task.model
    predictions, targets = [], []
    
    for batch in data_module.test_dataloader():
        pred = model(batch)['energy'].detach().numpy()
        true = batch['energy'].numpy()
        predictions.extend(pred)
        targets.extend(true)
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=1)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('DFT Energy (eV)')
    plt.ylabel('MLP Predicted Energy (eV)')
    plt.title('Energy Prediction Accuracy')
    plt.show()
    

**Ideal Result** :  
\- Points concentrate along red diagonal line (y=x)  
\- R² > 0.99 (coefficient of determination)

* * *

## 3.5 MLP-MD Simulation: Running Molecular Dynamics

Execute MD simulations 10⁴ times faster than DFT using trained MLP.

### MLP-MD Setup with ASE

**Example 9: Preparing MLP-MD Calculation (10 lines)**
    
    
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    import schnetpack.interfaces.ase_interface as spk_ase
    
    # Wrap MLP as ASE Calculator
    calculator = spk_ase.SpkCalculator(
        model_file='./training/best_model.ckpt',
        device='cpu'
    )
    
    # Prepare initial structure (first configuration from MD17)
    atoms = dataset.get_atoms(0)
    atoms.calc = calculator
    

### Initial Velocity Setup and Equilibration

**Example 10: Temperature Initialization (10 lines)**
    
    
    # Set velocity distribution at 300K
    temperature = 300  # K
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # Zero momentum (remove overall translation)
    from ase.md.velocitydistribution import Stationary
    Stationary(atoms)
    
    print(f"Initial kinetic energy: {atoms.get_kinetic_energy():.3f} eV")
    print(f"Initial potential energy: {atoms.get_potential_energy():.3f} eV")
    

### Running MD Simulation

**Example 11: MD Execution and Trajectory Saving (12 lines)**
    
    
    from ase.io.trajectory import Trajectory
    
    # Configure MD simulator
    timestep = 0.5 * units.fs  # 0.5 femtoseconds
    dyn = VelocityVerlet(atoms, timestep=timestep)
    
    # Output trajectory file
    traj = Trajectory('aspirin_md.traj', 'w', atoms)
    dyn.attach(traj.write, interval=10)  # Save every 10 steps
    
    # Run MD for 10,000 steps (5 picoseconds)
    dyn.run(10000)
    print("MD simulation completed!")
    

**Computational Time Estimate** :  
\- CPU (4 cores): Approximately 5 minutes (10,000 steps)  
\- With DFT: Approximately 1 week (10,000 steps)  
\- **Achieved 10⁴× speedup!**

### Trajectory Analysis

**Example 12: Energy Conservation and RDF Calculation (15 lines)**
    
    
    from ase.io import read
    import numpy as np
    
    # Read trajectory
    traj_data = read('aspirin_md.traj', index=':')
    
    # Check energy conservation
    energies = [a.get_total_energy() for a in traj_data]
    plt.plot(energies)
    plt.xlabel('Time step')
    plt.ylabel('Total Energy (eV)')
    plt.title('Energy Conservation Check')
    plt.show()
    
    # Calculate energy drift (monotonic increase/decrease)
    drift = (energies[-1] - energies[0]) / len(energies)
    print(f"Energy drift: {drift:.6f} eV/step")
    

**Good Simulation Indicators** :  
\- Energy drift: < 0.001 eV/step  
\- Total energy oscillates with time (conservation law)

* * *

## 3.6 Physical Property Calculations: Vibrational Spectra and Diffusion Coefficients

Calculate physical properties from MLP-MD.

### Vibrational Spectrum (Power Spectrum)

**Example 13: Vibrational Spectrum Calculation (15 lines)**
    
    
    from scipy.fft import fft, fftfreq
    
    # Extract velocity time series for one atom
    atom_idx = 0  # First atom
    velocities = np.array([a.get_velocities()[atom_idx] for a in traj_data])
    
    # Fourier transform of x-direction velocity
    vx = velocities[:, 0]
    freq = fftfreq(len(vx), d=timestep)
    spectrum = np.abs(fft(vx))**2
    
    # Plot positive frequencies only
    mask = freq > 0
    plt.plot(freq[mask] * 1e15 / (2 * np.pi), spectrum[mask])  # Hz → THz conversion
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Power Spectrum')
    plt.title('Vibrational Spectrum')
    plt.xlim(0, 100)
    plt.show()
    

**Interpretation** :  
\- Peaks correspond to molecular vibrational modes  
\- Compare with DFT-calculated vibrational spectrum for accuracy verification

### Mean Square Displacement (MSD) and Diffusion Coefficient

**Example 14: MSD Calculation (15 lines)**
    
    
    def calculate_msd(traj, atom_idx=0):
        """Calculate mean square displacement"""
        positions = np.array([a.positions[atom_idx] for a in traj])
        msd = np.zeros(len(positions))
    
        for t in range(len(positions)):
            displacement = positions[t:] - positions[:-t or None]
            msd[t] = np.mean(np.sum(displacement**2, axis=1))
    
        return msd
    
    # Calculate and plot MSD
    msd = calculate_msd(traj_data)
    time_ps = np.arange(len(msd)) * timestep / units.fs * 1e-3  # picoseconds
    
    plt.plot(time_ps, msd)
    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (Ų)')
    plt.title('Mean Square Displacement')
    plt.show()
    

**Diffusion Coefficient Calculation** :
    
    
    # Calculate diffusion coefficient from linear region of MSD (Einstein relation)
    # D = lim_{t→∞} MSD(t) / (6t)
    linear_region = slice(100, 500)
    fit = np.polyfit(time_ps[linear_region], msd[linear_region], deg=1)
    D = fit[0] / 6  # Ų/ps → cm²/s conversion needed
    print(f"Diffusion coefficient: {D:.6f} Ų/ps")
    

* * *

## 3.7 Active Learning: Efficient Data Addition

Automatically detect configurations where the model is uncertain and add DFT calculations.

### Ensemble Uncertainty Evaluation

**Example 15: Prediction Uncertainty (15 lines)**
    
    
    # Train multiple independent models (omitted: run Example 5 three times)
    models = [model1, model2, model3]  # 3 independent models
    
    def predict_with_uncertainty(atoms, models):
        """Ensemble prediction with uncertainty"""
        predictions = []
        for model in models:
            atoms.calc = spk_ase.SpkCalculator(model_file=model, device='cpu')
            predictions.append(atoms.get_potential_energy())
    
        mean = np.mean(predictions)
        std = np.std(predictions)
        return mean, std
    
    # Evaluate uncertainty for each configuration in MD trajectory
    uncertainties = []
    for atoms in traj_data[::100]:  # Every 100 frames
        _, std = predict_with_uncertainty(atoms, models)
        uncertainties.append(std)
    
    # Identify configurations with high uncertainty
    threshold = np.percentile(uncertainties, 95)
    high_uncertainty_frames = np.where(np.array(uncertainties) > threshold)[0]
    print(f"High uncertainty frames: {high_uncertainty_frames}")
    

**Next Steps** :  
\- Add configurations with high uncertainty to DFT calculations  
\- Update dataset and retrain model  
\- Verify accuracy improvement

* * *

## 3.8 Troubleshooting: Common Errors and Solutions

Introduce problems and solutions frequently encountered in practice.

Error | Cause | Solution  
---|---|---  
**Out of Memory (OOM)** | Batch size too large | Reduce `batch_size` from 32→16→8  
**Loss becomes NaN** | Learning rate too high | Lower `lr=1e-4`→`1e-5`  
**Energy drift in MD** | Timestep too large | Reduce `timestep=0.5fs`→`0.25fs`  
**Poor generalization** | Training data biased | Diversify data with Active Learning  
**CUDA error** | GPU compatibility issue | Verify PyTorch and CUDA versions  
  
### Debugging Best Practices
    
    
    # 1. Test with small-scale data
    data_module.num_train = 1000  # Quick test with 1,000 configurations
    
    # 2. Check overfitting on 1 batch
    trainer = pl.Trainer(max_epochs=100, overfit_batches=1)
    # If training error approaches 0, model has learning capacity
    
    # 3. Gradient clipping
    task = AtomisticTask(..., gradient_clip_val=1.0)  # Prevent gradient explosion
    

* * *

## 3.9 Chapter Summary

### What You Learned

  1. **Environment Setup**  
\- Installing Conda environment, PyTorch, SchNetPack  
\- Choosing GPU/CPU environment

  2. **Data Preparation**  
\- Downloading and loading MD17 dataset  
\- Splitting into training/validation/test sets

  3. **Model Training**  
\- Configuring SchNet architecture (6 layers, 128 dimensions)  
\- Training for 50 epochs (CPU: 2-3 hours)  
\- Monitoring progress with TensorBoard

  4. **Accuracy Verification**  
\- Verifying Energy MAE < 1 kcal/mol achieved  
\- Correlation plot of predictions vs true values  
\- High accuracy with R² > 0.99

  5. **MLP-MD Execution**  
\- Integration as ASE Calculator  
\- Running MD for 10,000 steps (5 picoseconds)  
\- Experiencing 10⁴× speedup over DFT

  6. **Physical Property Calculations**  
\- Vibrational spectrum (Fourier transform)  
\- Diffusion coefficient (calculated from mean square displacement)

  7. **Active Learning**  
\- Configuration selection by ensemble uncertainty  
\- Automated data addition strategy

### Important Points

  * **SchNetPack is easy to implement** : MLP training possible with a few dozen lines of code
  * **Practical accuracy achieved with small data (100,000 configurations)** : MD17 is an excellent benchmark
  * **MLP-MD is practical** : 10⁴× faster than DFT, executable on personal PCs
  * **Efficiency with Active Learning** : Automatically discover important configurations, reduce data collection costs

### To the Next Chapter

In Chapter 4, you'll learn about the latest MLP methods (NequIP, MACE) and actual research applications:  
\- Theory of E(3) equivariant graph neural networks  
\- Dramatic improvement in data efficiency (100,000→3,000 configurations)  
\- Application cases to catalytic reactions and battery materials  
\- Realization of large-scale simulations (1 million atoms)

* * *

## Exercises

### Exercise 1 (Difficulty: easy)

In Example 4's SchNet configuration, change `n_interactions` (number of message passing layers) to 3, 6, and 9, train the models, and predict how test MAE changes.

Hint The deeper the layers, the better they capture long-range atomic interactions. However, too deep increases overfitting risk.  Sample Answer **Expected Results**: | `n_interactions` | Predicted Test MAE | Training Time | Characteristics | |-----------------|-------------|---------|------| | **3** | 0.8-1.2 kcal/mol | 1 hour | Shallow, cannot fully capture long-range interactions | | **6** | 0.5-0.8 kcal/mol | 2-3 hours | Well balanced (recommended) | | **9** | 0.6-1.0 kcal/mol | 4-5 hours | Overfitting risk, accuracy decreases with insufficient training data | **Experimental Method**: 
    
    
    for n in [3, 6, 9]:
        representation = SchNet(n_interactions=n, ...)
        task = AtomisticTask(...)
        trainer.fit(task, datamodule=data_module)
        results = trainer.test(task, datamodule=data_module)
        print(f"n={n}: MAE={results[0]['test_energy_MAE']:.4f} eV")
    

**Conclusion**: For small molecules (21 aspirin atoms), `n_interactions=6` is optimal. For large-scale systems (100+ atoms), 9-12 layers may be effective. 

### Exercise 2 (Difficulty: medium)

In Example 11's MLP-MD, if energy drift exceeds acceptable range (e.g., 0.01 eV/step), what countermeasures can you consider? List three.

Hint Consider from three perspectives: timestep, training accuracy, and MD algorithm.  Sample Answer **Countermeasure 1: Reduce Timestep** 
    
    
    timestep = 0.25 * units.fs  # Halve from 0.5fs to 0.25fs
    dyn = VelocityVerlet(atoms, timestep=timestep)
    

\- **Reason**: Smaller timestep reduces numerical integration error \- **Drawback**: 2× longer computation time **Countermeasure 2: Improve Model Training Accuracy** 
    
    
    # Train with more data
    data_module.num_train = 200000  # Increase from 100,000 to 200,000 configurations
    
    # Or increase weight of force loss function
    task = AtomisticTask(..., loss_weights={'energy': 1.0, 'forces': 1000})
    

\- **Reason**: Low force prediction accuracy causes unstable MD \- **Target**: Force MAE < 0.05 eV/Å **Countermeasure 3: Switch to Langevin Dynamics (Heat Bath Coupling)** 
    
    
    from ase.md.langevin import Langevin
    dyn = Langevin(atoms, timestep=0.5*units.fs,
                   temperature_K=300, friction=0.01)
    

\- **Reason**: Heat bath absorbs energy drift \- **Caution**: No longer strictly microcanonical ensemble (NVE) **Priority**: Countermeasure 2 (improve accuracy) → Countermeasure 1 (timestep) → Countermeasure 3 (Langevin) 

* * *

## 3.10 Data License and Reproducibility

Information on datasets and tool versions needed to reproduce the hands-on code in this chapter.

### 3.10.1 Datasets Used

Dataset | Description | License | Access Method  
---|---|---|---  
**MD17** | Small molecule MD trajectories (10 types including aspirin, benzene) | CC0 1.0 (Public Domain) | Built into SchNetPack (`MD17(molecule='aspirin')`)  
**Aspirin molecule** | 211,762 configurations, DFT (PBE/def2-SVP) | CC0 1.0 | [sgdml.org](<http://sgdml.org/#datasets>)  
  
**Notes** :  
\- **Commercial Use** : CC0 license allows all commercial use, modification, and redistribution freely  
\- **Paper Citation** : When using MD17, cite the following  
Chmiela, S., et al. (2017). "Machine learning of accurate energy-conserving molecular force fields." _Science Advances_ , 3(5), e1603015.  
\- **Data Integrity** : SchNetPack's download function verifies with SHA256 checksum

### 3.10.2 Environment Information for Code Reproducibility

To accurately reproduce the code examples in this chapter, use the following versions.

Tool | Recommended Version | Installation Command | Compatibility  
---|---|---|---  
**Python** | 3.10.x | `conda create -n mlp python=3.10` | Tested on 3.9-3.11  
**PyTorch** | 2.1.0 | `conda install pytorch=2.1.0` | 2.0 or higher required  
**SchNetPack** | 2.0.3 | `pip install schnetpack==2.0.3` | API differs between 2.0 and 1.x series  
**ASE** | 3.22.1 | `pip install ase==3.22.1` | 3.20 or higher recommended  
**PyTorch Lightning** | 2.1.0 | `pip install pytorch-lightning==2.1.0` | Compatible with SchNetPack 2.0.3  
**NumPy** | 1.24.3 | `pip install numpy==1.24.3` | 1.20 or higher  
**Matplotlib** | 3.7.1 | `pip install matplotlib==3.7.1` | 3.5 or higher  
  
**Saving Environment File** :
    
    
    # Save current environment in reproducible form
    conda env export > environment.yml
    
    # Reproduce in another environment
    conda env create -f environment.yml
    

**Ensuring Reproducibility with Docker** (recommended):
    
    
    # Dockerfile example
    FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    RUN pip install schnetpack==2.0.3 ase==3.22.1 pytorch-lightning==2.1.0
    

### 3.10.3 Complete Record of Training Hyperparameters

Complete record of hyperparameters used in Examples 4 and 5 (for paper reproduction):

Parameter | Value | Description  
---|---|---  
`n_atom_basis` | 128 | Dimension of atomic feature vectors  
`n_interactions` | 6 | Number of message passing layers  
`cutoff` | 5.0 Å | Cutoff radius for atomic interactions  
`n_filters` | 128 | Number of convolution filters  
`batch_size` | 32 | Mini-batch size  
`learning_rate` | 1e-4 | Initial learning rate (AdamW)  
`max_epochs` | 50 | Maximum training epochs  
`num_train` | 100,000 | Number of training data  
`num_val` | 10,000 | Number of validation data  
`num_test` | 10,000 | Number of test data  
`random_seed` | 42 | Random seed (data split reproducibility)  
  
**Complete Reproduction Code** :
    
    
    import torch
    torch.manual_seed(42)  # Ensure reproducibility
    
    representation = SchNet(
        n_atom_basis=128, n_interactions=6, cutoff=5.0, n_filters=128
    )
    task = AtomisticTask(
        model=AtomisticModel(representation, [output]),
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={'lr': 1e-4, 'weight_decay': 0.01}
    )
    trainer = pl.Trainer(max_epochs=50, deterministic=True)
    

### 3.10.4 Energy and Force Unit Conversion Table

Unit conversions for physical quantities used in this chapter (SchNetPack and ASE standard units):

Physical Quantity | SchNetPack/ASE | eV | kcal/mol | Hartree  
---|---|---|---|---  
**Energy** | eV | 1.0 | 23.06 | 0.03674  
**Force** | eV/Å | 1.0 | 23.06 | 0.01945  
**Distance** | Å | - | - | 1.889726 Bohr  
**Time** | fs (femtoseconds) | - | - | 0.02419 a.u.  
  
**Unit Conversion Example** :
    
    
    from ase import units
    
    # Energy conversion
    energy_ev = 1.0  # eV
    energy_kcal = energy_ev * 23.06052  # kcal/mol
    energy_hartree = energy_ev * 0.036749  # Hartree
    
    # Using ASE unit constants (recommended)
    print(f"{energy_ev} eV = {energy_ev * units.eV / units.kcal * units.mol} kcal/mol")
    

* * *

## 3.11 Practical Precautions: Common Failure Patterns in Hands-On

### 3.11.1 Pitfalls in Environment Setup and Installation

**Failure 1: PyTorch and CUDA Version Mismatch**

**Problem** :
    
    
    RuntimeError: CUDA error: no kernel image is available for execution on the device
    

**Cause** :  
PyTorch 2.1.0 is compiled with CUDA 11.8 or 12.1, but system CUDA is older version like 10.2

**Diagnostic Code** :
    
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    
    # Check system CUDA version (terminal)
    # nvcc --version
    

**Solution** :
    
    
    # For system CUDA 11.8
    conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    
    # Switch to CPU version if CUDA unavailable
    conda install pytorch==2.1.0 cpuonly -c pytorch
    

**Prevention** :  
Check GPU driver and CUDA version with `nvidia-smi` before environment setup

**Failure 2: Confusing SchNetPack 1.x and 2.x APIs**

**Problem** :
    
    
    AttributeError: module 'schnetpack' has no attribute 'AtomsData'
    

**Cause** :  
Running old tutorial code from SchNetPack 1.x series on 2.x series

**Version Check** :
    
    
    import schnetpack as spk
    print(spk.__version__)  # If 2.0.3, this chapter's code works
    

**Main API Changes** :

SchNetPack 1.x | SchNetPack 2.x  
---|---  
`spk.AtomsData` | `spk.data.AtomsDataModule`  
`spk.atomistic.Atomwise` | `spk.task.ModelOutput`  
`spk.train.Trainer` | `pytorch_lightning.Trainer`  
  
**Solution** :  
Use this chapter's code examples (2.x series) or refer to SchNetPack official documentation ([schnetpack.readthedocs.io](<https://schnetpack.readthedocs.io>)) 2.x series tutorials

**Failure 3: Misdiagnosis of Memory Shortage (OOM)**

**Problem** :
    
    
    RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
    

**Common Misconception** :  
"GPU memory shortage so need to buy new GPU" → **Wrong**

**Diagnostic Procedure** :
    
    
    # 1. Check batch size
    print(f"Current batch size: {data_module.batch_size}")
    
    # 2. Check GPU memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    

**Solutions (Priority Order)** :

  1. **Reduce batch size** : `batch_size=32` → `16` → `8` → `4`
  2. **Gradient accumulation** : Accumulate small batches multiple times to simulate large batch

    
    
    trainer = pl.Trainer(accumulate_grad_batches=4)  # Update every 4 batches
    

  3. **Mixed Precision training** : Halve memory usage

    
    
    trainer = pl.Trainer(precision=16)  # Use float16
    

**Guidelines** :  
\- GPU 4GB: batch_size=4-8  
\- GPU 8GB: batch_size=16-32  
\- GPU 24GB: batch_size=64-128

### 3.11.2 Training and Debugging Pitfalls

**Failure 4: Training Error Not Decreasing (NaN Loss)**

**Problem** :
    
    
    Epoch 5: train_loss=nan, val_loss=nan
    

**Top 3 Causes** :

  1. **Learning rate too high** : Gradient explosion → Parameters become NaN
  2. **Lack of data normalization** : Energy absolute values too large (e.g., -1000 eV)
  3. **Inappropriate force loss coefficient** : Force loss too dominant

**Diagnostic Code** :
    
    
    # Check model output immediately after training starts
    for batch in data_module.train_dataloader():
        output = task.model(batch)
        print(f"Energy prediction: {output['energy'][:5]}")  # First 5 samples
        print(f"Energy target: {batch['energy'][:5]}")
        break
    
    # NaN check
    print(f"Has NaN in prediction: {torch.isnan(output['energy']).any()}")
    

**Solutions** :

  1. **Lower learning rate** :

    
    
    optimizer_args={'lr': 1e-5}  # Decrease from 1e-4 to 1e-5
    

  2. **Gradient clipping** :

    
    
    trainer = pl.Trainer(gradient_clip_val=1.0)  # Clip gradient norm to ≤1.0
    

  3. **Data normalization** (automatic in SchNetPack 2.x, but verify manually):

    
    
    import schnetpack.transform as trn
    data_module.train_transforms = [
        trn.SubtractCenterOfMass(),
        trn.RemoveOffsets('energy', remove_mean=True)  # Remove energy offset
    ]
    

**Failure 5: Missing Overfitting Signs**

**Problem** :  
Training error decreases but validation error stagnates or increases
    
    
    Epoch 30: train_loss=0.001, val_loss=0.050  # val_loss worsening
    

**Cause** :  
Model memorizes training data, generalization performance to unseen data deteriorates

**Diagnostic Graph** :
    
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    metrics = pd.read_csv('./training/lightning_logs/version_0/metrics.csv')
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

**Solutions** :

  1. **Early Stopping** (automatic stopping):

    
    
    from pytorch_lightning.callbacks import EarlyStopping
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    trainer = pl.Trainer(callbacks=[early_stop])
    

  2. **Data augmentation** (increase training data):

    
    
    data_module.num_train = 200000  # Increase from 100,000 to 200,000
    

  3. **Dropout (not recommended, less effective for MLP)** : Instead reduce model parameter count

    
    
    representation = SchNet(n_atom_basis=64, n_interactions=4)  # Reduce from 128→64
    

### 3.11.3 MLP-MD Simulation Pitfalls

**Failure 6: Underestimating Energy Drift**

**Problem** :  
Energy monotonically increases/decreases during MD simulation (breaking conservation law)

**Misconception About Acceptable Range** :

  * "A little drift is unavoidable" → **Dangerous**
  * 0.01 eV/step drift results in 100 eV energy change over 10,000 steps (unrealistic)

**Quantitative Diagnosis** :
    
    
    from ase.io import read
    
    traj = read('aspirin_md.traj', index=':')
    energies = [a.get_total_energy() for a in traj]
    
    # Calculate drift (linear fit)
    import numpy as np
    time_steps = np.arange(len(energies))
    drift_rate, offset = np.polyfit(time_steps, energies, deg=1)
    print(f"Energy drift: {drift_rate:.6f} eV/step")
    
    # Check acceptable range
    if abs(drift_rate) > 0.001:
        print("⚠️ WARNING: Excessive energy drift detected!")
    

**Solutions (Detailed from Exercise 2)** :

  1. **Improve force training accuracy** (most important):

    
    
    # Significantly increase weight of force loss function
    task = AtomisticTask(
        loss_weights={'energy': 0.01, 'forces': 0.99}  # 99% weight on forces
    )
    

  2. **Optimize timestep** :

    
    
    # Stability test
    for dt in [0.1, 0.25, 0.5, 1.0]:  # fs
        # If stable at dt=0.5 but diverges at dt=1.0, adopt dt=0.5
    

  3. **Recognize MLP accuracy limits** :  
If force MAE > 0.1 eV/Å, long-time MD (>10 ps) has reduced reliability  
→ Add training data with Active Learning

**Failure 7: Not Verifying Physical Validity of MD Results**

**Problem** :  
Misconception that "MD completed so it's successful" → Actually unphysical structural changes

**Items to Verify** :

**1\. Temperature Control Check** :
    
    
    temperatures = [a.get_temperature() for a in traj]
    print(f"Average T: {np.mean(temperatures):.1f} K (target: 300 K)")
    print(f"Std T: {np.std(temperatures):.1f} K")
    # Standard deviation > 30K indicates abnormality
    

**2\. Structure Disruption Check** :
    
    
    from ase.geometry.analysis import Analysis
    
    # Compare initial and final structures
    ana_init = Analysis(traj[0])
    ana_final = Analysis(traj[-1])
    
    # Check if bonds are broken
    bonds_init = ana_init.all_bonds[0]
    bonds_final = ana_final.all_bonds[0]
    print(f"Initial bonds: {len(bonds_init)}, Final bonds: {len(bonds_final)}")
    
    # Change in bond count → Possible structure disruption
    

**3\. Validity of Radial Distribution Function (RDF)** :
    
    
    # Check if first peak position matches DFT calculations or X-ray diffraction data
    # (Implementation advanced, omitted)
    

**Solution** :  
If physically valid results not obtained, likely extrapolation beyond training data range  
→ Add those configurations to training data with Active Learning

* * *

## 3.12 Chapter Completion Checklist: Hands-On Quality Assurance

After completing this chapter, verify the following items. If you can check all items, you're ready to apply MLP in actual research projects.

### 3.12.1 Conceptual Understanding

**Understanding Tools and Environment** :

  * □ Can explain SchNetPack's role (MLP training library)
  * □ Understand relationship between PyTorch and SchNetPack (PyTorch-based MLP implementation)
  * □ Can explain ASE's role (atomic structure manipulation, MD execution)
  * □ Understand MD17 dataset characteristics (10 small molecules, DFT accuracy)

**Understanding Model Training** :

  * □ Can explain meaning of SchNet hyperparameters (`n_atom_basis`, `n_interactions`, `cutoff`)
  * □ Understand roles of training/validation/test sets
  * □ Can identify overfitting signs (validation error increase)
  * □ Understand that Energy MAE < 1 kcal/mol is high accuracy benchmark

**Understanding MLP-MD** :

  * □ Understand mechanism of MLP integration as ASE Calculator
  * □ Can explain difference between energy conservation law and energy drift
  * □ Understand why timestep (0.5 fs) affects stability
  * □ Can explain why MLP-MD is 10⁴× faster than DFT

### 3.12.2 Practical Skills

**Environment Setup** :

  * □ Can create Conda environment and install Python 3.10
  * □ Can correctly install PyTorch (CPU/GPU versions)
  * □ Can install SchNetPack 2.0.3 and ASE 3.22.1
  * □ Can run environment verification script and check versions

**Data Preparation and Training** :

  * □ Can download MD17 dataset and split into 100,000 configurations
  * □ Can define SchNet model and set hyperparameters
  * □ Can run 50-epoch training and monitor progress with TensorBoard
  * □ Can evaluate MAE on test set and achieve target accuracy (< 1 kcal/mol)

**MLP-MD Simulation** :

  * □ Can wrap trained model as ASE Calculator
  * □ Can set initial velocities with Maxwell-Boltzmann distribution
  * □ Can run MD for 10,000 steps (5 picoseconds)
  * □ Can save trajectory and verify energy conservation

**Analysis and Troubleshooting** :

  * □ Can calculate vibrational spectrum (power spectrum)
  * □ Can calculate diffusion coefficient from mean square displacement (MSD)
  * □ Can handle Out of Memory (OOM) errors (reduce batch size)
  * □ Can diagnose NaN loss causes and adjust learning rate

### 3.12.3 Application Skills

**Application Plan to Your Research** :

  * □ Can design MD17-equivalent dataset for your research target (molecules, materials)
  * □ Can estimate required DFT calculation count (from target accuracy and system size)
  * □ Can develop strategy to optimize SchNet hyperparameters for your system
  * □ Can clearly define physical properties to obtain from MLP-MD (diffusion coefficient, vibrational spectrum, reaction path)

**Problem Solving and Debugging** :

  * □ Can execute diagnostic procedure when training doesn't converge (learning rate, data normalization, gradient clipping)
  * □ Can identify causes of energy drift and select countermeasures
  * □ Can detect overfitting and apply Early Stopping or Data Augmentation
  * □ Can optimize batch size and training time according to GPU/CPU resources

**Preparation for Advanced Techniques** :

  * □ Understand Active Learning concept (Example 15) and can explain implementation flow
  * □ Understand importance of configuration selection by ensemble uncertainty
  * □ Have expectations for how data efficiency improves in next chapter (NequIP, MACE)
  * □ Can continue self-study using SchNetPack documentation ([schnetpack.readthedocs.io](<https://schnetpack.readthedocs.io>))

**Bridge to Next Chapter** :

  * □ Recognize SchNet limitations (data efficiency, rotational equivariance)
  * □ Interested in how E(3) equivariant architectures (NequIP, MACE) improve
  * □ Ready to learn actual research applications (catalysts, batteries, drug discovery) in Chapter 4

* * *

## References

  1. Schütt, K. T., et al. (2019). "SchNetPack: A Deep Learning Toolbox For Atomistic Systems." _Journal of Chemical Theory and Computation_ , 15(1), 448-455.  
DOI: [10.1021/acs.jctc.8b00908](<https://doi.org/10.1021/acs.jctc.8b00908>)

  2. Chmiela, S., et al. (2017). "Machine learning of accurate energy-conserving molecular force fields." _Science Advances_ , 3(5), e1603015.  
DOI: [10.1126/sciadv.1603015](<https://doi.org/10.1126/sciadv.1603015>)

  3. Larsen, A. H., et al. (2017). "The atomic simulation environment—a Python library for working with atoms." _Journal of Physics: Condensed Matter_ , 29(27), 273002.  
DOI: [10.1088/1361-648X/aa680e](<https://doi.org/10.1088/1361-648X/aa680e>)

  4. Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance deep learning library." _Advances in Neural Information Processing Systems_ , 32.  
arXiv: [1912.01703](<https://arxiv.org/abs/1912.01703>)

  5. Zhang, L., et al. (2020). "Active learning of uniformly accurate interatomic potentials for materials simulation." _Physical Review Materials_ , 3(2), 023804.  
DOI: [10.1103/PhysRevMaterials.3.023804](<https://doi.org/10.1103/PhysRevMaterials.3.023804>)

  6. Schütt, K. T., et al. (2017). "Quantum-chemical insights from deep tensor neural networks." _Nature Communications_ , 8(1), 13890.  
DOI: [10.1038/ncomms13890](<https://doi.org/10.1038/ncomms13890>)

* * *

## Author Information

**Created by** : MI Knowledge Hub Content Team  
**Created on** : 2025-10-17  
**Version** : 1.1 (Chapter 3 quality improvement)  
**Series** : MLP Introduction Series

**Update History** :  
\- 2025-10-19: v1.1 Quality improvement revision  
\- Added data license and reproducibility section (MD17 dataset, aspirin molecule information)  
\- Code reproducibility information (Python 3.10.x, PyTorch 2.1.0, SchNetPack 2.0.3, ASE 3.22.1)  
\- Complete record of training hyperparameters (11 items, for paper reproduction)  
\- Energy and force unit conversion table (eV, kcal/mol, Hartree mutual conversion)  
\- Added practical precautions section (7 failure patterns: CUDA mismatch, API confusion, OOM, NaN loss, overfitting, energy drift, physical validity)  
\- Added chapter completion checklist (12 conceptual understanding items, 16 practical skill items, 16 application skill items)  
\- 2025-10-17: v1.0 Chapter 3 initial release  
\- Python environment setup (Conda, PyTorch, SchNetPack)  
\- MD17 dataset preparation and splitting  
\- SchNet model training (15 code examples)  
\- MLP-MD execution and analysis (trajectory, vibrational spectrum, MSD)  
\- Active Learning uncertainty evaluation  
\- Troubleshooting table (5 items)  
\- 2 exercises (easy, medium)  
\- 6 references

**License** : Creative Commons BY-NC-SA 4.0

[← Back to Series Index](<index.html>)
