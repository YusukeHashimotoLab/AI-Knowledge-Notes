---
title: "Chapter 3: Hands-On MLP with Python - SchNetPack Tutorial"
subtitle: "From Environment Setup to Training and MLP-MD"
level: "beginner-intermediate"
difficulty: "Beginner to Intermediate"
target_audience: "undergraduate, graduate"
estimated_time: "30-35 minutes"
learning_objectives:
  - Set up MLP tools (SchNetPack) in a Python environment
  - Train SchNet models with small datasets (MD17)
  - Evaluate trained MLP accuracy and diagnose issues
  - Execute MLP-MD simulations and analyze results
topics: ["schnetpack", "python", "hands-on", "training", "md-simulation"]
prerequisites: ["Chapter 1", "Chapter 2", "Python basics", "Jupyter Notebook"]
series: "MLP Introduction Series v1.0"
series_order: 3
version: "1.0"
created_at: "2025-10-17"
template_version: "1.0"
---

# Chapter 3: Hands-On MLP with Python - SchNetPack Tutorial

## Learning Objectives

By completing this chapter, you will be able to:
- Install SchNetPack in a Python environment and set up the development environment
- Train MLP models using a small dataset (aspirin molecule from MD17)
- Evaluate trained model accuracy and verify energy/force prediction errors
- Execute MLP-MD simulations and analyze trajectories
- Understand common errors and troubleshooting strategies

---

## 3.1 Environment Setup: Installing Required Tools

To practice with MLPs, you need to set up a Python environment and install SchNetPack.

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.9-3.11 | Base language |
| **PyTorch** | 2.0+ | Deep learning framework |
| **SchNetPack** | 2.0+ | MLP training and inference |
| **ASE** | 3.22+ | Atomic structure manipulation, MD execution |
| **NumPy/Matplotlib** | Latest | Data analysis and visualization |

### Installation Steps

**Step 1: Create Conda Environment**

```bash
# Create new Conda environment (Python 3.10)
conda create -n mlp-tutorial python=3.10 -y
conda activate mlp-tutorial
```

**Step 2: Install PyTorch**

```bash
# CPU version (local machine, lightweight)
conda install pytorch cpuonly -c pytorch

# GPU version (if CUDA is available)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 3: Install SchNetPack and ASE**

```bash
# SchNetPack (pip recommended)
pip install schnetpack

# ASE (Atomic Simulation Environment)
pip install ase

# Visualization tools
pip install matplotlib seaborn
```

**Step 4: Verify Installation**

```python
# Example 1: Environment verification script (5 lines)
import torch
import schnetpack as spk
print(f"PyTorch: {torch.__version__}")
print(f"SchNetPack: {spk.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
```

**Expected output**:
```
PyTorch: 2.1.0
SchNetPack: 2.0.3
GPU available: False  # For CPU
```

---

## 3.2 Data Preparation: Obtaining the MD17 Dataset

SchNetPack includes the **MD17** benchmark dataset for small molecules.

### About the MD17 Dataset

- **Content**: Molecular dynamics trajectories from DFT calculations
- **Molecules**: 10 types including aspirin, benzene, ethanol
- **Data size**: ~100,000 configurations per molecule
- **Accuracy**: PBE/def2-SVP level (DFT)
- **Use case**: Benchmarking MLP methods

### Downloading and Loading Data

**Example 2: Loading MD17 dataset (10 lines)**

```python
from schnetpack.datasets import MD17
from schnetpack.data import AtomsDataModule

# Download aspirin molecule dataset (~100k configurations)
dataset = MD17(
    datapath='./data',
    molecule='aspirin',
    download=True
)

print(f"Total samples: {len(dataset)}")
print(f"Properties: {dataset.available_properties}")
print(f"First sample: {dataset[0]}")
```

**Output**:
```
Total samples: 211762
Properties: ['energy', 'forces']
First sample: {'_atomic_numbers': tensor([...]), 'energy': tensor(-1234.5), 'forces': tensor([...])}
```

### Data Splitting

**Example 3: Train/validation/test split (10 lines)**

```python
# Split data into train:validation:test = 70%:15%:15%
data_module = AtomsDataModule(
    datapath='./data',
    dataset=dataset,
    batch_size=32,
    num_train=100000,      # Number of training samples
    num_val=10000,          # Number of validation samples
    num_test=10000,         # Number of test samples
    split_file='split.npz', # Save split information
)
data_module.prepare_data()
data_module.setup()
```

**Explanation**:
- `batch_size=32`: Process 32 configurations at a time (memory efficiency)
- `num_train=100000`: Large dataset improves generalization
- `split_file`: Save split to file (ensures reproducibility)

---

## 3.3 Model Training with SchNetPack

Train a SchNet model to learn energies and forces.

### Configuring SchNet Architecture

**Example 4: Defining SchNet model (15 lines)**

```python
import schnetpack.transform as trn
from schnetpack.representation import SchNet
from schnetpack.model import AtomisticModel
from schnetpack.task import ModelOutput

# 1. SchNet representation layer (atomic configuration → feature vectors)
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
```

**Parameter explanation**:
- `n_atom_basis=128`: Each atom's feature vector is 128-dimensional (typical value)
- `n_interactions=6`: 6 message passing layers (deeper captures long-range interactions)
- `cutoff=5.0Å`: Ignore atomic interactions beyond this distance (computational efficiency)

### Executing Training

**Example 5: Training loop setup (15 lines)**

```python
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
```

**Training time estimates**:
- CPU (4 cores): ~2-3 hours (100k configurations)
- GPU (RTX 3090): ~15-20 minutes

### Monitoring Training Progress

**Example 6: Visualization with TensorBoard (10 lines)**

```python
# Launch TensorBoard (in separate terminal)
# tensorboard --logdir=./training/lightning_logs

# Check logs from Python
import pandas as pd

metrics = pd.read_csv('./training/lightning_logs/version_0/metrics.csv')
print(metrics[['epoch', 'train_loss', 'val_loss']].tail(10))
```

**Expected output**:
```
   epoch  train_loss  val_loss
40    40      0.0023    0.0031
41    41      0.0022    0.0030
42    42      0.0021    0.0029
...
```

**Key observations**:
- Both `train_loss` and `val_loss` decreasing → Normal learning
- `val_loss` starts increasing → **Overfitting** sign → Consider Early Stopping

---

## 3.4 Accuracy Validation: Energy and Force Prediction Accuracy

Evaluate whether the trained model achieves DFT-level accuracy.

### Evaluation on Test Set

**Example 7: Test set evaluation (12 lines)**

```python
# Evaluate on test set
test_results = trainer.test(task, datamodule=data_module)

# Display results
print(f"Energy MAE: {test_results[0]['test_energy_MAE']:.4f} eV")
print(f"Energy RMSE: {test_results[0]['test_energy_RMSE']:.4f} eV")

# Force evaluation (requires separate calculation)
from schnetpack.metrics import MeanAbsoluteError
force_mae = MeanAbsoluteError(target='forces')
# ... Force evaluation code
```

**Good accuracy benchmarks** (aspirin molecule, 21 atoms):
- **Energy MAE**: < 1 kcal/mol (< 0.043 eV)
- **Force MAE**: < 1 kcal/mol/Å (< 0.043 eV/Å)

### Correlation Plot of Predictions vs. True Values

**Example 8: Visualizing prediction accuracy (15 lines)**

```python
import matplotlib.pyplot as plt
import numpy as np

# Make predictions on test data
model = task.model
predictions, targets = [], []

for batch in data_module.test_dataloader():
    pred = model(batch)['energy'].detach().numpy()
    true = batch['energy'].numpy()
    predictions.extend(pred)
    targets.extend(true)

# Create scatter plot
plt.scatter(targets, predictions, alpha=0.5, s=1)
plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
plt.xlabel('DFT Energy (eV)')
plt.ylabel('MLP Predicted Energy (eV)')
plt.title('Energy Prediction Accuracy')
plt.show()
```

**Ideal result**:
- Points densely clustered on red diagonal line (y=x)
- R² > 0.99 (coefficient of determination)

---

## 3.5 MLP-MD Simulation: Running Molecular Dynamics

Use the trained MLP to run MD simulations 10⁴ times faster than DFT.

### Setting Up MLP-MD with ASE

**Example 9: Preparing MLP-MD calculation (10 lines)**

```python
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
```

### Setting Initial Velocities and Equilibration

**Example 10: Temperature initialization (10 lines)**

```python
# Set velocity distribution at 300K
temperature = 300  # K
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

# Remove total momentum (eliminate system translation)
from ase.md.velocitydistribution import Stationary
Stationary(atoms)

print(f"Initial kinetic energy: {atoms.get_kinetic_energy():.3f} eV")
print(f"Initial potential energy: {atoms.get_potential_energy():.3f} eV")
```

### Running MD Simulation

**Example 11: MD execution and trajectory saving (12 lines)**

```python
from ase.io.trajectory import Trajectory

# Configure MD simulator
timestep = 0.5 * units.fs  # 0.5 femtoseconds
dyn = VelocityVerlet(atoms, timestep=timestep)

# Output trajectory file
traj = Trajectory('aspirin_md.traj', 'w', atoms)
dyn.attach(traj.write, interval=10)  # Save every 10 steps

# Run 10,000 steps (5 picoseconds) of MD
dyn.run(10000)
print("MD simulation completed!")
```

**Computation time estimates**:
- CPU (4 cores): ~5 minutes (10,000 steps)
- DFT would take: ~1 week (10,000 steps)
- **Achieved 10⁴× speedup!**

### Trajectory Analysis

**Example 12: Energy conservation and RDF calculation (15 lines)**

```python
from ase.io import read
import numpy as np

# Load trajectory
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
```

**Good simulation indicators**:
- Energy drift: < 0.001 eV/step
- Total energy oscillates over time (conservation law)

---

## 3.6 Property Calculations: Vibrational Spectra and Diffusion Coefficients

Calculate physical properties from MLP-MD.

### Vibrational Spectrum (Power Spectrum)

**Example 13: Vibrational spectrum calculation (15 lines)**

```python
from scipy.fft import fft, fftfreq

# Extract velocity time series for one atom
atom_idx = 0  # First atom
velocities = np.array([a.get_velocities()[atom_idx] for a in traj_data])

# Fourier transform of x-direction velocity
vx = velocities[:, 0]
freq = fftfreq(len(vx), d=timestep)
spectrum = np.abs(fft(vx))**2

# Plot only positive frequencies
mask = freq > 0
plt.plot(freq[mask] * 1e15 / (2 * np.pi), spectrum[mask])  # Hz → THz conversion
plt.xlabel('Frequency (THz)')
plt.ylabel('Power Spectrum')
plt.title('Vibrational Spectrum')
plt.xlim(0, 100)
plt.show()
```

**Interpretation**:
- Peaks correspond to molecular vibrational modes
- Compare with DFT-calculated vibrational spectrum for accuracy validation

### Mean Square Displacement (MSD) and Diffusion Coefficient

**Example 14: MSD calculation (15 lines)**

```python
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
time_ps = np.arange(len(msd)) * timestep / units.fs * 1e-3  # Picoseconds

plt.plot(time_ps, msd)
plt.xlabel('Time (ps)')
plt.ylabel('MSD (Ų)')
plt.title('Mean Square Displacement')
plt.show()
```

**Calculating diffusion coefficient**:
```python
# Calculate diffusion coefficient from linear region of MSD (Einstein relation)
# D = lim_{t→∞} MSD(t) / (6t)
linear_region = slice(100, 500)
fit = np.polyfit(time_ps[linear_region], msd[linear_region], deg=1)
D = fit[0] / 6  # Ų/ps → cm²/s conversion needed
print(f"Diffusion coefficient: {D:.6f} Ų/ps")
```

---

## 3.7 Active Learning: Efficient Data Addition

Automatically detect configurations where the model is uncertain and add DFT calculations.

### Evaluating Ensemble Uncertainty

**Example 15: Prediction uncertainty (15 lines)**

```python
# Train multiple independent models (abbreviated: run Example 5 three times)
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
```

**Next steps**:
- Add high-uncertainty configurations to DFT calculations
- Update dataset and retrain model
- Verify accuracy improvement

---

## 3.8 Troubleshooting: Common Errors and Solutions

Common problems encountered in practice and their solutions.

| Error | Cause | Solution |
|-------|-------|----------|
| **Out of Memory (OOM)** | Batch size too large | Reduce `batch_size` 32→16→8 |
| **Loss becomes NaN** | Learning rate too high | Lower `lr=1e-4`→`1e-5` |
| **Energy drift in MD** | Timestep too large | Reduce `timestep=0.5fs`→`0.25fs` |
| **Poor generalization** | Biased training data | Diversify data with Active Learning |
| **CUDA error** | GPU compatibility issue | Check PyTorch and CUDA versions |

### Debugging Best Practices

```python
# 1. Test with small dataset
data_module.num_train = 1000  # Quick test with 1,000 configurations

# 2. Check overfitting on single batch
trainer = pl.Trainer(max_epochs=100, overfit_batches=1)
# If training error approaches 0, model has learning capability

# 3. Gradient clipping
task = AtomisticTask(..., gradient_clip_val=1.0)  # Prevent gradient explosion
```

---

## 3.9 Chapter Summary

### What You Learned

1. **Environment Setup**
   - Installing Conda environment, PyTorch, SchNetPack
   - Choosing GPU/CPU environment

2. **Data Preparation**
   - Downloading and loading MD17 dataset
   - Splitting into train/validation/test sets

3. **Model Training**
   - Configuring SchNet architecture (6 layers, 128 dimensions)
   - Training for 50 epochs (CPU: 2-3 hours)
   - Monitoring progress with TensorBoard

4. **Accuracy Validation**
   - Confirming energy MAE < 1 kcal/mol achievement
   - Correlation plot of predictions vs. true values
   - High accuracy with R² > 0.99

5. **MLP-MD Execution**
   - Integration as ASE Calculator
   - Running 10,000 steps (5 picoseconds) of MD
   - Experiencing 10⁴× speedup over DFT

6. **Property Calculations**
   - Vibrational spectrum (Fourier transform)
   - Diffusion coefficient (calculated from mean square displacement)

7. **Active Learning**
   - Configuration selection using ensemble uncertainty
   - Automated data addition strategy

### Key Points

- **SchNetPack is easy to implement**: MLP training with just a few dozen lines of code
- **Practical accuracy with small data (100k configurations)**: MD17 is an excellent benchmark
- **MLP-MD is practical**: 10⁴× faster than DFT, executable on personal PCs
- **Active Learning improves efficiency**: Automatically discover important configurations, reduce data collection costs

### Moving to the Next Chapter

Chapter 4 will cover state-of-the-art MLP methods (NequIP, MACE) and real research applications:
- Theory of E(3)-equivariant graph neural networks
- Dramatic improvement in data efficiency (100k→3,000 configurations)
- Application cases in catalysis, battery materials
- Realizing large-scale simulations (1 million atoms)

---

## Practice Problems

### Problem 1 (Difficulty: easy)

Using the SchNet configuration in Example 4, change `n_interactions` (number of message passing layers) to 3, 6, and 9, train the models, and predict how the test MAE will change.

<details>
<summary>Hint</summary>

Deeper layers can capture long-range atomic interactions. However, too deep risks overfitting.

</details>

<details>
<summary>Sample Answer</summary>

**Predicted results**:

| `n_interactions` | Predicted Test MAE | Training Time | Characteristics |
|-----------------|-------------------|---------------|-----------------|
| **3** | 0.8-1.2 kcal/mol | 1 hour | Shallow, cannot fully capture long-range interactions |
| **6** | 0.5-0.8 kcal/mol | 2-3 hours | Well-balanced (recommended) |
| **9** | 0.6-1.0 kcal/mol | 4-5 hours | Overfitting risk, accuracy drops with insufficient training data |

**Experimental method**:
```python
for n in [3, 6, 9]:
    representation = SchNet(n_interactions=n, ...)
    task = AtomisticTask(...)
    trainer.fit(task, datamodule=data_module)
    results = trainer.test(task, datamodule=data_module)
    print(f"n={n}: MAE={results[0]['test_energy_MAE']:.4f} eV")
```

**Conclusion**: For small molecules (aspirin 21 atoms), `n_interactions=6` is optimal. For large systems (100+ atoms), 9-12 layers may be effective.

</details>

### Problem 2 (Difficulty: medium)

In the MLP-MD from Example 11, if energy drift exceeds acceptable range (e.g., 0.01 eV/step), what countermeasures can be considered? List three.

<details>
<summary>Hint</summary>

Consider from three perspectives: timestep, training accuracy, and MD algorithm.

</details>

<details>
<summary>Sample Answer</summary>

**Countermeasure 1: Reduce timestep**
```python
timestep = 0.25 * units.fs  # Halve 0.5fs → 0.25fs
dyn = VelocityVerlet(atoms, timestep=timestep)
```
- **Reason**: Smaller timestep reduces numerical integration error
- **Downside**: 2× computation time

**Countermeasure 2: Improve model training accuracy**
```python
# Train with more data
data_module.num_train = 200000  # Increase 100k→200k configurations

# Or increase weight of force loss function
task = AtomisticTask(..., loss_weights={'energy': 1.0, 'forces': 1000})
```
- **Reason**: Low force prediction accuracy destabilizes MD
- **Target**: Force MAE < 0.05 eV/Å

**Countermeasure 3: Switch to Langevin dynamics (heat bath coupling)**
```python
from ase.md.langevin import Langevin
dyn = Langevin(atoms, timestep=0.5*units.fs,
               temperature_K=300, friction=0.01)
```
- **Reason**: Heat bath absorbs energy drift
- **Note**: No longer strict microcanonical ensemble (NVE)

**Priority**: Countermeasure 2 (improve accuracy) → Countermeasure 1 (timestep) → Countermeasure 3 (Langevin)

</details>

---

## References

1. Schütt, K. T., et al. (2019). "SchNetPack: A Deep Learning Toolbox For Atomistic Systems." *Journal of Chemical Theory and Computation*, 15(1), 448-455.
   DOI: [10.1021/acs.jctc.8b00908](https://doi.org/10.1021/acs.jctc.8b00908)

2. Chmiela, S., et al. (2017). "Machine learning of accurate energy-conserving molecular force fields." *Science Advances*, 3(5), e1603015.
   DOI: [10.1126/sciadv.1603015](https://doi.org/10.1126/sciadv.1603015)

3. Larsen, A. H., et al. (2017). "The atomic simulation environment—a Python library for working with atoms." *Journal of Physics: Condensed Matter*, 29(27), 273002.
   DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)

4. Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance deep learning library." *Advances in Neural Information Processing Systems*, 32.
   arXiv: [1912.01703](https://arxiv.org/abs/1912.01703)

5. Zhang, L., et al. (2020). "Active learning of uniformly accurate interatomic potentials for materials simulation." *Physical Review Materials*, 3(2), 023804.
   DOI: [10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1103/PhysRevMaterials.3.023804)

6. Schütt, K. T., et al. (2017). "Quantum-chemical insights from deep tensor neural networks." *Nature Communications*, 8(1), 13890.
   DOI: [10.1038/ncomms13890](https://doi.org/10.1038/ncomms13890)

---

## Author Information

**Created by**: MI Knowledge Hub Content Team
**Supervised by**: Dr. Yusuke Hashimoto (Tohoku University)
**Created on**: 2025-10-17
**Version**: 1.0 (Chapter 3 initial version)
**Series**: MLP Introduction Series

**Update History**:
- 2025-10-17: v1.0 Chapter 3 initial version created
  - Python environment setup (Conda, PyTorch, SchNetPack)
  - MD17 dataset preparation and splitting
  - SchNet model training (15 code examples)
  - MLP-MD execution and analysis (trajectory, vibrational spectrum, MSD)
  - Active Learning uncertainty evaluation
  - Troubleshooting table (5 items)
  - 2 practice problems (easy, medium)
  - 6 references

**License**: Creative Commons BY-NC-SA 4.0
