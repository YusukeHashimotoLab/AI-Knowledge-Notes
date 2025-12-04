---
title: "Chapter 2: Crystal Field Theory and Electronic Structure"
chapter_title: "Chapter 2: Crystal Field Theory and Electronic Structure"
subtitle: d-Orbital Splitting and Ligand Field Effects in Transition Metal Compounds
reading_time: 25-30 minutes
difficulty: Intermediate
code_examples: 8
version: 1.0
created_at: "by:"
---

Why do transition metal compounds display such diverse colors? Learn crystal field theory and understand the relationship between d-orbital splitting and electronic states. From the Jahn-Teller effect and ligand field theory to real material applications, explore energy levels through Python calculations. 

## Learning Objectives

By completing this chapter, you will master the following:

### Foundation Level (Beginners)

  * ✅ Explain the basic concepts and physical background of crystal field theory
  * ✅ Understand d-orbital splitting patterns in octahedral and tetrahedral coordination
  * ✅ Explain the meaning of crystal field splitting energy (Δ)

### Intermediate Level (Practitioners)

  * ✅ Understand the mechanism and application examples of the Jahn-Teller effect
  * ✅ Explain the difference between ligand field theory and crystal field theory
  * ✅ Calculate and visualize d-orbital energy levels using Python
  * ✅ Analyze the relationship between color and electronic states in transition metal compounds

### Advanced Level (Researchers)

  * ✅ Read Tanabe-Sugano diagrams and predict d-electron configurations
  * ✅ Determine crystal field parameters from experimental data
  * ✅ Predict magnetic and optical properties of transition metal compounds using first-principles calculations

* * *

## 2.1 Fundamentals of Crystal Field Theory

### What is Crystal Field Theory?

**Crystal Field Theory (CFT)** is a theory that explains how the energy levels of d-orbitals split when transition metal ions are surrounded by ligands. It was proposed by Hans Bethe and John Van Vleck in 1929.

#### 📖 Definition: Crystal Field Theory

An electrostatic model that treats ligands as **negative point charges** and describes how the electrostatic field (crystal field) lifts the degeneracy of d-orbitals, causing energy level splitting.

**Basic Concept** :

  1. In an isolated transition metal ion, the five d-orbitals are degenerate (same energy)
  2. When ligands approach, electrostatic repulsion increases the energy of d-orbitals
  3. The arrangement (symmetry) of ligands causes d-orbitals to split into different energy levels

### Spatial Distribution of d-Orbitals

There are five types of d-orbitals, each with different spatial distributions:

Orbital | Spatial Distribution | Characteristic  
---|---|---  
$d_{z^2}$ | Extends along z-axis | Strong repulsion with axial ligands  
$d_{x^2-y^2}$ | Extends along x-y plane | Strong repulsion with planar ligands  
$d_{xy}$ | Diagonal direction in xy plane | Weak repulsion with ligands  
$d_{xz}$ | Diagonal direction in xz plane | Weak repulsion with ligands  
$d_{yz}$ | Diagonal direction in yz plane | Weak repulsion with ligands  
  
**💡 Key Point**  
$d_{z^2}$ and $d_{x^2-y^2}$ point toward ligands and are called "eg orbitals," while $d_{xy}, d_{xz}, d_{yz}$ point between ligands and are called "t2g orbitals" (in octahedral coordination). 

## 2.2 Crystal Field Splitting in Octahedral Coordination

### Symmetry of Octahedral Coordination

In **octahedral coordination (O h symmetry)** where a transition metal ion is surrounded by six ligands, d-orbitals split as follows:
    
    
    ```mermaid
    graph TD
        A[Isolated Ion5 d-orbitals are degenerate] --> B[Spherical FieldOverall energy increase]
        B --> C[Octahedral FieldSplit into e_g and t_2g]
    
        D["e_g (d_z², d_x²-y²)Higher energy"] -.Δ_oct.-> E["t_2g (d_xy, d_xz, d_yz)Lower energy"]
    
        C --> D
        C --> E
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#f5a3fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f5b3fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#ff6b6b,stroke:#c92a2a,stroke-width:2px,color:#fff
        style E fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#fff
    ```

**Crystal Field Splitting Energy (Δ oct or 10Dq)**:

#### 📖 Definition: Δoct

The energy difference between eg and t2g orbitals in octahedral coordination.

$$\Delta_{\text{oct}} = E(e_g) - E(t_{2g})$$

**Energy Stabilization** :

  * eg orbitals: + 0.6 Δoct (destabilized)
  * t2g orbitals: - 0.4 Δoct (stabilized)
  * Energy barycenter is conserved

### Visualizing Octahedral Coordination Energy Diagram
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_octahedral_splitting(delta_oct=10):
        """
        Visualize d-orbital crystal field splitting in octahedral coordination
    
        Parameters:
        -----------
        delta_oct : float
            Crystal field splitting energy (in Dq units)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Energy levels
        E_barycenter = 0  # Energy barycenter
        E_eg = E_barycenter + 0.6 * delta_oct
        E_t2g = E_barycenter - 0.4 * delta_oct
    
        # Isolated ion (left)
        ax.hlines(0, 0, 1.5, colors='gray', linewidth=2, label='Isolated ion (5d degenerate)')
        ax.text(0.75, 0.5, '5d', ha='center', fontsize=12, fontweight='bold')
    
        # Spherical field (center)
        ax.hlines(E_barycenter, 2.5, 4, colors='blue', linewidth=2, label='Spherical field')
        ax.text(3.25, E_barycenter+0.5, '5d', ha='center', fontsize=12, fontweight='bold')
    
        # Octahedral field (right)
        ax.hlines(E_eg, 5, 7, colors='red', linewidth=3, label='e$_g$ (2 orbitals)')
        ax.text(6, E_eg+0.5, '$d_{z^2}$, $d_{x^2-y^2}$', ha='center', fontsize=11)
    
        ax.hlines(E_t2g, 5, 7, colors='green', linewidth=3, label='t$_{2g}$ (3 orbitals)')
        ax.text(6, E_t2g-0.8, '$d_{xy}$, $d_{xz}$, $d_{yz}$', ha='center', fontsize=11)
    
        # Arrow showing Δ_oct
        ax.annotate('', xy=(7.5, E_eg), xytext=(7.5, E_t2g),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(8, (E_eg + E_t2g)/2, f'$\\Delta_{{oct}}$ = {delta_oct} Dq',
                fontsize=13, fontweight='bold', color='purple')
    
        # Axis settings
        ax.set_xlim(-0.5, 9)
        ax.set_ylim(-5, 8)
        ax.set_ylabel('Energy (Dq)', fontsize=12)
        ax.set_xticks([0.75, 3.25, 6])
        ax.set_xticklabels(['Isolated Ion', 'Spherical Field', 'Octahedral Field'], fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title('Crystal Field Splitting in Octahedral Coordination', fontsize=14, fontweight='bold')
    
        plt.tight_layout()
        plt.show()
    
    # Execute
    plot_octahedral_splitting(delta_oct=10)
    

#### 🔬 Example: Color of [Ti(H2O)6]3+

**Electronic configuration** : Ti3+ has d1 electron configuration (1 d-electron)

  * Ground state: 1 electron in t2g orbital
  * Excited state: 1 electron in eg orbital
  * Visible light absorption: Absorbs green-yellow (approximately 20,000 cm-1)
  * Observed color: **Purple** (complementary color)

## 2.3 Crystal Field Splitting in Tetrahedral Coordination

### Characteristics of Tetrahedral Coordination

In **tetrahedral coordination (T d symmetry)**, with four ligands, the splitting pattern is reversed from octahedral:

  * e orbitals ($d_{z^2}, d_{x^2-y^2}$): **Lower** energy
  * t2 orbitals ($d_{xy}, d_{xz}, d_{yz}$): **Higher** energy
  * Splitting magnitude: $\Delta_{\text{tet}} \approx \frac{4}{9} \Delta_{\text{oct}}$ (approximately 44%)

**💡 Why is it reversed?**  
In tetrahedral coordination, ligands are positioned on body diagonals, so t2 orbitals (electron density in diagonal directions) experience stronger repulsion with ligands and have higher energy. 

### Comparison of Octahedral and Tetrahedral
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def compare_octahedral_tetrahedral():
        """Compare crystal field splitting in octahedral and tetrahedral coordination"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # Octahedral coordination
        delta_oct = 10
        E_eg_oct = 0.6 * delta_oct
        E_t2g_oct = -0.4 * delta_oct
    
        ax1.hlines(E_eg_oct, 0, 1, colors='red', linewidth=4, label='e$_g$')
        ax1.text(0.5, E_eg_oct + 0.5, 'e$_g$ (2)', ha='center', fontsize=12, fontweight='bold')
    
        ax1.hlines(E_t2g_oct, 0, 1, colors='green', linewidth=4, label='t$_{2g}$')
        ax1.text(0.5, E_t2g_oct - 0.7, 't$_{2g}$ (3)', ha='center', fontsize=12, fontweight='bold')
    
        ax1.annotate('', xy=(1.3, E_eg_oct), xytext=(1.3, E_t2g_oct),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax1.text(1.6, (E_eg_oct + E_t2g_oct)/2, '$\\Delta_{oct}$', fontsize=13, fontweight='bold')
    
        ax1.set_xlim(-0.2, 2)
        ax1.set_ylim(-6, 8)
        ax1.set_ylabel('Energy (Dq)', fontsize=12)
        ax1.set_title('Octahedral Coordination (O$_h$)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks([])
    
        # Tetrahedral coordination
        delta_tet = (4/9) * delta_oct  # Approximately 4.44 Dq
        E_t2_tet = 0.6 * delta_tet
        E_e_tet = -0.4 * delta_tet
    
        ax2.hlines(E_t2_tet, 0, 1, colors='green', linewidth=4, label='t$_2$')
        ax2.text(0.5, E_t2_tet + 0.5, 't$_2$ (3)', ha='center', fontsize=12, fontweight='bold')
    
        ax2.hlines(E_e_tet, 0, 1, colors='red', linewidth=4, label='e')
        ax2.text(0.5, E_e_tet - 0.7, 'e (2)', ha='center', fontsize=12, fontweight='bold')
    
        ax2.annotate('', xy=(1.3, E_t2_tet), xytext=(1.3, E_e_tet),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
        ax2.text(1.7, (E_t2_tet + E_e_tet)/2, f'$\\Delta_{{tet}}$\n≈ {delta_tet:.1f} Dq',
                fontsize=12, fontweight='bold')
    
        ax2.set_xlim(-0.2, 2.2)
        ax2.set_ylim(-6, 8)
        ax2.set_ylabel('Energy (Dq)', fontsize=12)
        ax2.set_title('Tetrahedral Coordination (T$_d$)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks([])
    
        plt.tight_layout()
        plt.show()
    
    compare_octahedral_tetrahedral()
    

## 2.4 Jahn-Teller Effect

### Jahn-Teller Theorem

#### 📖 Jahn-Teller Theorem (1937)

Nonlinear molecules with degenerate electronic states will **always undergo structural distortion to lift the degeneracy**. This lowers the total energy.

**Conditions for occurrence** :

  * eg or t2g orbitals are partially occupied
  * Typical examples: Cu2+ (d9), Mn3+ (d4 high spin), Cr2+ (d4 high spin)

### Jahn-Teller Distortion in Cu2+

Cu2+ has d9 electron configuration with 3 electrons in eg orbitals ($d_{x^2-y^2}$: 2, $d_{z^2}$: 1).

#### 🔬 Example: Structural Distortion in CuF2

**Ideal octahedron** : Six F- at equal distance (e.g., 2.0 Å)

**After Jahn-Teller distortion** :

  * z-axis direction: Cu-F bonds elongate (2.27 Å) → $d_{z^2}$ energy decreases
  * xy plane: Cu-F bonds contract (1.93 Å) → $d_{x^2-y^2}$ energy increases
  * Result: eg degeneracy is lifted → Energy stabilization

### Simulation of Jahn-Teller Effect
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def jahn_teller_distortion_energy():
        """Calculate energy change due to Jahn-Teller distortion"""
        # Distortion parameter (percentage change in bond length)
        distortion = np.linspace(-0.15, 0.15, 100)  # -15% to +15%
    
        # Cu2+ case: d9 configuration, 3 electrons in eg orbitals
        # eg degenerate energy in ideal octahedron
        E_ideal = 0
    
        # Energy after distortion (simplified model)
        # For axial elongation (positive distortion)
        E_dz2 = E_ideal - 1000 * distortion**2  # d_z2 orbital stabilized
        E_dx2y2 = E_ideal + 800 * distortion**2  # d_x2-y2 orbital destabilized
    
        # Electronic configuration: 2 electrons in d_z2, 1 electron in d_x2-y2
        E_total = 2 * E_dz2 + 1 * E_dx2y2
    
        # Elastic energy (cost of structural distortion)
        E_elastic = 500 * distortion**2
    
        # Total energy
        E_net = E_total + E_elastic
    
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Orbital energies
        ax1.plot(distortion * 100, E_dz2, 'b-', linewidth=2, label='$d_{z^2}$ (2 electrons)')
        ax1.plot(distortion * 100, E_dx2y2, 'r-', linewidth=2, label='$d_{x^2-y^2}$ (1 electron)')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Axial Distortion (%)', fontsize=12)
        ax1.set_ylabel('Orbital Energy (cm$^{-1}$)', fontsize=12)
        ax1.set_title('e$_g$ Orbital Splitting', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
    
        # Total energy
        ax2.plot(distortion * 100, E_total, 'g-', linewidth=2, label='Electronic Energy')
        ax2.plot(distortion * 100, E_elastic, 'orange', linewidth=2, label='Elastic Energy')
        ax2.plot(distortion * 100, E_net, 'purple', linewidth=3, label='Total Energy')
    
        # Most stable structure
        min_idx = np.argmin(E_net)
        min_distortion = distortion[min_idx]
        ax2.plot(min_distortion * 100, E_net[min_idx], 'ro', markersize=10, label='Most Stable Structure')
    
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Axial Distortion (%)', fontsize=12)
        ax2.set_ylabel('Energy (cm$^{-1}$)', fontsize=12)
        ax2.set_title('Jahn-Teller Stabilization Energy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        print(f"Most stable distortion: {min_distortion*100:.2f}%")
        print(f"Stabilization energy: {-E_net[min_idx]:.1f} cm^-1")
    
    jahn_teller_distortion_energy()
    

## 2.5 Ligand Field Theory

### Limitations of Crystal Field Theory and Ligand Field Theory

Crystal field theory treats ligands as simple **point charges** , but in reality there is **covalent bonding** between ligands and metal ions.

#### 📖 Ligand Field Theory (LFT)

A theory based on molecular orbital theory that considers **hybridization** between metal d-orbitals and ligand orbitals. Enables more precise description than crystal field theory.

Aspect | Crystal Field Theory (CFT) | Ligand Field Theory (LFT)  
---|---|---  
Treatment of ligands | Point charges | Treated as molecular orbitals  
Orbital hybridization | Not considered | Considers σ bonding and π bonding  
Prediction accuracy | Qualitative | Semi-quantitative to quantitative  
Applicability | Weak ligands | Strong ligands, π donation/backdonation  
  
### Spectrochemical Series

The arrangement of ligands by their **crystal field splitting strength** is called the **spectrochemical series** :

I- < Br- < Cl- < F- < OH- < H2O < NH3 < en < CN- < CO 

← Weak ligand field ────── Strong ligand field → 

  * **Weak ligands** (I-, Br-, Cl-): Small Δ → High spin state
  * **Strong ligands** (CN-, CO): Large Δ → Low spin state

### Calculating High Spin and Low Spin States
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def calculate_spin_states(d_electrons, delta_oct):
        """
        Calculate high spin/low spin from d-electron count and crystal field splitting
    
        Parameters:
        -----------
        d_electrons : int
            Number of d-electrons (1-10)
        delta_oct : float
            Crystal field splitting energy (unit: cm^-1)
    
        Returns:
        --------
        dict : Information about spin states
        """
        # Pairing energy (energy required for electron pairing)
        P = 15000  # Typical value: approximately 15,000 cm^-1
    
        # High spin configuration (Hund's rule priority)
        if d_electrons <= 3:
            high_spin = d_electrons  # Sequential placement in t2g orbitals
            hs_t2g = d_electrons
            hs_eg = 0
        elif d_electrons <= 5:
            high_spin = d_electrons  # Fill t2g then eg orbitals
            hs_t2g = 3
            hs_eg = d_electrons - 3
        elif d_electrons <= 8:
            high_spin = d_electrons - 5  # Fill t2g, eg then start pairing
            hs_t2g = min(6, d_electrons)
            hs_eg = max(0, d_electrons - 6)
        else:
            high_spin = 10 - d_electrons
            hs_t2g = 6
            hs_eg = d_electrons - 6
    
        # Low spin configuration (when Δ_oct > P)
        if d_electrons <= 6:
            low_spin = d_electrons % 2  # Fill t2g orbitals first
            ls_t2g = min(6, d_electrons)
            ls_eg = max(0, d_electrons - 6)
        else:
            low_spin = (d_electrons - 6) % 2
            ls_t2g = 6
            ls_eg = d_electrons - 6
    
        # Crystal Field Stabilization Energy (CFSE)
        cfse_high = (-0.4 * hs_t2g + 0.6 * hs_eg) * delta_oct
        cfse_low = (-0.4 * ls_t2g + 0.6 * ls_eg) * delta_oct
    
        # Total energy considering pairing energy
        n_pairs_high = (d_electrons - high_spin) / 2
        n_pairs_low = (d_electrons - low_spin) / 2
    
        E_high = cfse_high + n_pairs_high * P
        E_low = cfse_low + n_pairs_low * P
    
        return {
            'high_spin': high_spin,
            'low_spin': low_spin,
            'E_high': E_high,
            'E_low': E_low,
            'stable_state': 'Low Spin' if E_low < E_high else 'High Spin'
        }
    
    # Example: Fe2+ (d6)
    d_electrons = 6
    delta_values = np.linspace(5000, 25000, 100)
    
    high_spin_list = []
    low_spin_list = []
    
    for delta in delta_values:
        result = calculate_spin_states(d_electrons, delta)
        high_spin_list.append(result['E_high'])
        low_spin_list.append(result['E_low'])
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(delta_values, high_spin_list, 'r-', linewidth=2, label='High Spin (S=2)')
    plt.plot(delta_values, low_spin_list, 'b-', linewidth=2, label='Low Spin (S=0)')
    
    # Crossover point (spin transition point)
    idx_cross = np.argmin(np.abs(np.array(high_spin_list) - np.array(low_spin_list)))
    delta_cross = delta_values[idx_cross]
    plt.axvline(delta_cross, color='green', linestyle='--', linewidth=2, label=f'Spin Crossover: {delta_cross:.0f} cm$^{{-1}}$')
    
    plt.xlabel('Crystal Field Splitting Δ$_{oct}$ (cm$^{-1}$)', fontsize=12)
    plt.ylabel('Total Energy (cm$^{-1}$)', fontsize=12)
    plt.title(f'High Spin and Low Spin States of Fe$^{{2+}}$ (d$^{6}$)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Spin transition point: Δ_oct ≈ {delta_cross:.0f} cm^-1")
    

## 2.6 Tanabe-Sugano Diagrams

### What are Tanabe-Sugano Diagrams?

**Tanabe-Sugano diagrams** show electronic states of transition metal ions with dn electron configuration as a function of crystal field splitting strength (Δ/B).

**💡 How to Read Tanabe-Sugano Diagrams**  

  * Horizontal axis: Δ/B (crystal field splitting / Racah parameter)
  * Vertical axis: E/B (energy level / Racah parameter)
  * Each curve: Different electronic state (labeled with term symbols)
  * Ground state shown with bold line

### Simplified Tanabe-Sugano Diagram for d3 Electron Configuration
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_tanabe_sugano_d3():
        """Tanabe-Sugano diagram for d3 electron configuration (simplified version)"""
        # Range of Δ/B
        delta_over_B = np.linspace(0, 3.5, 100)
    
        # Simplified electronic state energies (actual is more complex)
        # 4F (ground state)
        E_4F = 0 * delta_over_B
    
        # 4P
        E_4P = 15 + 0 * delta_over_B
    
        # 2G
        E_2G = 17 + 2 * delta_over_B
    
        # 2H
        E_2H = 22 + 1.5 * delta_over_B
    
        # 2D
        E_2D = 28 + 0.8 * delta_over_B
    
        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
    
        ax.plot(delta_over_B, E_4F, 'b-', linewidth=3, label='$^4$F (ground state)')
        ax.plot(delta_over_B, E_4P, 'r-', linewidth=2, label='$^4$P')
        ax.plot(delta_over_B, E_2G, 'g-', linewidth=2, label='$^2$G')
        ax.plot(delta_over_B, E_2H, 'orange', linewidth=2, label='$^2$H')
        ax.plot(delta_over_B, E_2D, 'purple', linewidth=2, label='$^2$D')
    
        # Example absorption transition (Cr3+)
        delta_B_example = 2.3  # Typical value for Cr3+
        ax.axvline(delta_B_example, color='gray', linestyle='--', alpha=0.5)
        ax.text(delta_B_example + 0.1, 35, 'Cr$^{3+}$\n(ruby)', fontsize=10, color='red')
    
        ax.set_xlabel('Δ / B', fontsize=13, fontweight='bold')
        ax.set_ylabel('E / B', fontsize=13, fontweight='bold')
        ax.set_title('Tanabe-Sugano Diagram (d$^3$ Electron Configuration)', fontsize=15, fontweight='bold')
        ax.set_xlim(0, 3.5)
        ax.set_ylim(0, 40)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    plot_tanabe_sugano_d3()
    

#### 🔬 Example: Color of Ruby (Cr3+:Al2O3)

Cr3+ has d3 electron configuration and is octahedrally coordinated in Al2O3.

  * Ground state: 4A2g
  * Excited states: 4T2g (blue-green absorption), 4T1g (yellow absorption)
  * Result: Absorbs blue and yellow → Transmits **red**
  * Fluorescence: 2Eg → 4A2g transition (sharp red emission)

## 2.7 Applications of Transition Metal Compounds

### Applications in Catalysis

Transition metal compounds are widely used as catalysts for **redox reactions** and **ligand exchange reactions** due to their partially occupied d-orbitals.

Catalyst | Reaction | Role of Crystal Field  
---|---|---  
Fe2+/Fe3+ | Fenton reaction (H2O2 decomposition) | Electron transfer in eg orbitals  
Ti3+/Ti4+ | Ziegler-Natta polymerization | Ligand activation  
V2+/V3+ | Redox flow battery | Reversible redox  
Ru2+/Ru3+ | Water oxidation reaction | t2g-π* backdonation  
  
### Applications in Magnetic Materials

Unpaired electrons in transition metal compounds are the source of **magnetism**.

  * **Ferromagnets** : Fe, Co, Ni (metals), CrO2 (magnetic tape)
  * **Antiferromagnets** : MnO, NiO, FeO
  * **Ferrimagnets** : Fe3O4 (magnetite), γ-Fe2O3 (maghemite)

### Applications in Optical Materials

Materials utilizing light absorption by crystal field:

  * **Gemstones** : Ruby (Cr3+), Emerald (Cr3+), Sapphire (Fe2+/Ti4+)
  * **Pigments** : Prussian blue (Fe2+/Fe3+), chromium oxide green (Cr3+)
  * **Solar cells** : Dye-sensitized solar cells (Ru complexes)

### Visualizing the Relationship Between Crystal Field Splitting and Color
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    
    def crystal_field_color_chart():
        """Relationship between crystal field splitting and color for transition metal ions"""
    
        ions = ['Ti³⁺', 'V³⁺', 'Cr³⁺', 'Mn²⁺', 'Fe³⁺', 'Co²⁺', 'Ni²⁺', 'Cu²⁺']
        d_electrons = [1, 2, 3, 5, 5, 7, 8, 9]
        delta_oct = [20300, 18900, 17400, 21000, 14000, 9300, 8500, 12600]  # cm^-1
        colors = ['purple', 'green', 'red', 'pale pink', 'yellow', 'pink', 'green', 'blue']
        hex_colors = ['#9b59b6', '#27ae60', '#e74c3c', '#f8b3c7', '#f1c40f', '#ff69b4', '#2ecc71', '#3498db']
    
        fig, ax = plt.subplots(figsize=(12, 7))
    
        x_pos = np.arange(len(ions))
        bars = ax.bar(x_pos, delta_oct, color=hex_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
        # Data labels
        for i, (bar, ion, d, col) in enumerate(zip(bars, ions, d_electrons, colors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{int(height)} cm⁻¹', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2., -2000,
                    f'd{d}', ha='center', va='top', fontsize=10, color='gray')
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    col, ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white', rotation=90)
    
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ions, fontsize=12, fontweight='bold')
        ax.set_ylabel('Crystal Field Splitting Δ$_{oct}$ (cm$^{-1}$)', fontsize=13)
        ax.set_title('Crystal Field Splitting and Observed Colors for Transition Metal Ions in Octahedral Coordination', fontsize=14, fontweight='bold')
        ax.set_ylim(-3000, 24000)
        ax.grid(axis='y', alpha=0.3)
    
        # Reference line for spectrochemical series
        ax.axhline(15000, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Typical ligands (H₂O, NH₃)')
    
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    crystal_field_color_chart()
    

## 2.8 Crystal Field Calculations Using First-Principles Methods

### Calculating Crystal Field Parameters with DFT

In modern materials science, crystal field splitting energy can be directly calculated using **first-principles calculations (DFT)**.

**💡 DFT Calculation Workflow**  

  1. Crystal structure optimization
  2. Electronic structure calculation (band structure, DOS)
  3. Analyze d-orbital density of states (PDOS)
  4. Determine Δoct from energy difference between t2g and eg

### Generating Transition Metal Compound Structures with ASE
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from ase import Atoms
    from ase.visualize import view
    import numpy as np
    
    def create_octahedral_complex(metal='Ti', ligand='O', bond_length=2.0):
        """
        Generate octahedral coordination transition metal complex
    
        Parameters:
        -----------
        metal : str
            Central metal ion
        ligand : str
            Ligand atom
        bond_length : float
            Metal-ligand bond length (Å)
    
        Returns:
        --------
        atoms : ase.Atoms
            Generated structure
        """
        # Ligand positions in octahedral coordination (±x, ±y, ±z axes)
        ligand_positions = np.array([
            [bond_length, 0, 0],
            [-bond_length, 0, 0],
            [0, bond_length, 0],
            [0, -bond_length, 0],
            [0, 0, bond_length],
            [0, 0, -bond_length]
        ])
    
        # Create atomic object
        symbols = [metal] + [ligand] * 6
        positions = np.vstack([[0, 0, 0], ligand_positions])
    
        atoms = Atoms(symbols=symbols, positions=positions)
    
        # Set cell size (for visualization)
        cell_size = bond_length * 3
        atoms.set_cell([cell_size, cell_size, cell_size])
        atoms.center()
    
        return atoms
    
    # Generate TiO6 octahedron
    complex_TiO6 = create_octahedral_complex(metal='Ti', ligand='O', bond_length=2.0)
    
    print(f"Generated structure: {complex_TiO6.get_chemical_formula()}")
    print(f"Atomic positions:\n{complex_TiO6.get_positions()}")
    
    # Visualization (if ASE GUI available)
    # view(complex_TiO6)
    
    # Save structure to file
    complex_TiO6.write('TiO6_octahedral.xyz')
    print("Structure saved to TiO6_octahedral.xyz")
    

## Exercises

**Exercise 2.1: Foundation Level - Crystal Field Splitting Calculation**

**Problem** : For Ni2+ (d8) in octahedral coordination with crystal field splitting energy Δoct = 8,500 cm-1, determine the following:

  1. Energy (cm-1) of eg and t2g orbitals
  2. Crystal Field Stabilization Energy (CFSE)
  3. Which is more stable: high spin or low spin state?

**Hint** :

  * eg orbitals: + 0.6 Δoct
  * t2g orbitals: - 0.4 Δoct
  * Ni2+ (d8) is almost always in high spin state

**Exercise 2.2: Intermediate Level - Jahn-Teller Effect**

**Problem** : Among the following transition metal ions, select all that exhibit the Jahn-Teller effect and explain the reasons.

  1. Ti3+ (d1)
  2. Cr3+ (d3)
  3. Mn3+ (d4, high spin)
  4. Fe2+ (d6, high spin)
  5. Cu2+ (d9)

**Illustrate the electronic configuration in a diagram using Python**

**Exercise 2.3: Intermediate Level - Spectrochemical Series**

**Problem** : Explain why [Co(H2O)6]2+ and [Co(NH3)6]2+ have different colors using the spectrochemical series.

  * [Co(H2O)6]2+: Pink
  * [Co(NH3)6]2+: Yellow-brown

**Visualize the difference in absorption spectra using Python**

**Exercise 2.4: Advanced Level - Reading Tanabe-Sugano Diagrams**

**Problem** : From the Tanabe-Sugano diagram for Cr3+ (d3 electron configuration) in ruby, determine the following:

  1. Term symbol of the ground state
  2. Absorption bands appearing in the visible region (15,000-25,000 cm-1) and their origins
  3. Calculate Δ/B when Racah parameter B = 918 cm-1 and Δoct = 17,400 cm-1

**Draw a simplified Tanabe-Sugano diagram using Python and illustrate absorption transitions**

**Exercise 2.5: Advanced Level - Spin Crossover**

**Problem** : Calculate the conditions for an Fe2+ (d6) complex to transition from high spin state (S=2) to low spin state (S=0) using the following parameters:

  * Pairing energy: P = 15,000 cm-1
  * Crystal field splitting: Δoct = 10,000-25,000 cm-1

**Draw the spin crossover curve using Python and estimate the transition temperature**

**Exercise 2.6: Advanced Level - DFT Calculation Preparation**

**Problem** : To calculate crystal field splitting in MnO (rocksalt structure) using DFT, prepare the following:

  1. Generate MnO unit cell using ASE (lattice constant a = 4.445 Å)
  2. Create VASP input files (INCAR, POSCAR, KPOINTS)
  3. Set DFT+U method parameters (Ueff = 4.0 eV)

**Expected result** : Extract Δoct from Mn2+ d-orbital PDOS

**Exercise 2.7: Integrated Exercise - Catalyst Material Design**

**Problem** : To design transition metal oxides as water splitting catalysts, consider the following:

  1. Among Co3+, Ni3+, and Cu2+, which ion has the most appropriate eg orbital occupation?
  2. Which is more favorable for catalytic activity: octahedral or tetrahedral coordination?
  3. List three material property parameters that should be verified by DFT calculations

**Compare d-orbital energy diagrams for candidate materials using Python**

**Exercise 2.8: Research Project - New Magnetic Material Discovery**

**Project Assignment** : Propose a new ferromagnetic material utilizing the Jahn-Teller effect.

**Requirements** :

  1. Selection of appropriate transition metal ion (rationale for d-electron configuration)
  2. Crystal structure and coordination environment design
  3. Magnetic moment prediction using DFT calculations
  4. Estimation of Curie temperature
  5. Evaluation of synthesis feasibility

**Deliverables** :

  * Material design report (2 pages)
  * Python code (structure generation, energy calculation, visualization)
  * Complete set of VASP input files

## Summary

In this chapter, we learned **crystal field theory** and **ligand field theory** and understood the relationship between d-orbital splitting and electronic states in transition metal compounds.

### Review of Key Points

  * ✅ **Crystal Field Theory** : Treats ligands as point charges and explains d-orbital splitting electrostatically
  * ✅ **Octahedral Coordination** : Splits into eg and t2g, with Δoct as the splitting width
  * ✅ **Tetrahedral Coordination** : Splitting pattern is reversed, Δtet ≈ 4/9 Δoct
  * ✅ **Jahn-Teller Effect** : Degenerate electronic states are lifted by structural distortion
  * ✅ **Ligand Field Theory** : More precise theory considering covalency
  * ✅ **Spectrochemical Series** : Ordering of ligand crystal field splitting strength
  * ✅ **Tanabe-Sugano Diagrams** : Graphical representation of dn electronic states vs. Δ/B
  * ✅ **Applications** : Catalysts, magnetic materials, optical materials, pigments

### Connection to Next Chapter

In Chapter 3, we will learn how to actually calculate these concepts using **first-principles calculations (DFT)**. Master the Hohenberg-Kohn theorem, Kohn-Sham equations, and practical workflows using Python libraries (ASE, Pymatgen).
