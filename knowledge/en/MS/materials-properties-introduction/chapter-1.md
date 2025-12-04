---
title: "Chapter 1: Fundamentals of Solid-State Electronic Theory"
chapter_title: "Chapter 1: Fundamentals of Solid-State Electronic Theory"
subtitle: Introduction to Band Theory - Why Do Metals Conduct Electricity While Insulators Do Not?
reading_time: 30-35 minutes
difficulty: Intermediate
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Materials Properties](<../../MS/materials-properties-introduction/index.html>)›Chapter 1

🌐 EN | [🇯🇵 JP](<../../../jp/MS/materials-properties-introduction/chapter-1.html>) | Last sync: 2025-11-16

## Learning Objectives

Upon completing this chapter, you will be able to explain:

### Fundamental Understanding

  * The concepts of the free electron model and Fermi energy
  * The formation mechanism of band structures and the origin of band gaps
  * Differences in electronic structures of metals, semiconductors, and insulators

### Practical Skills

  * Calculate Fermi energy and density of states using Python
  * Plot and interpret simple band structure diagrams
  * Visualize and understand the concept of Fermi surfaces

### Application Capabilities

  * Estimate band gaps from experimental data
  * Predict electrical properties of materials using band theory
  * Possess foundational knowledge to design electronic structures of new materials

## 1.1 Why Do We Need Band Theory?

### 1.1.1 Starting with Everyday Questions

Why do copper wires conduct electricity while rubber does not? Both are composed of atoms, so where does this difference originate?

To answer this simple question, physicists established "band theory" in the early 20th century. This theory is one of the most important foundational theories in materials science and can explain:

  * **Origin of Conductivity** : Why metals conduct electricity while insulators do not
  * **Operating Principles of Semiconductors** : How transistors and solar cells function
  * **Optical Properties** : Why diamond is transparent while copper is opaque
  * **Magnetism** : What conditions make materials magnetic

**💡 Historical Background**

The foundations of band theory were laid in the 1920s by Felix Bloch and Rudolf Peierls. In the 1930s, Wilson, Brillouin, Wannier, and others developed it further, establishing a universal framework for describing the electronic structure of solids. This theory became the theoretical foundation supporting the semiconductor revolution (invention of the transistor, 1947).

### 1.1.2 Limitations of Classical Theory

In classical physics (19th century), the Drude model (1900) was proposed, treating electrons as freely moving particles. While this model could explain conductivity to some extent, it had the following critical problems:

Phenomenon | Drude Model Prediction | Experimental Result  
---|---|---  
Electronic contribution to specific heat | Large (3/2 Nk_B) | Very small (~0.01 Nk_B)  
Existence of insulators | Cannot explain | Clearly exists  
Temperature dependence of semiconductors | Same as metals | Completely different (exponential)  
  
It became clear that to resolve these contradictions, we need to consider **quantum mechanics** and the **periodic structure of crystals**.

## 1.2 Free Electron Model - The Simplest Quantum Description

### 1.2.1 Quantum Mechanical Basics: Particles Are Waves

According to the fundamental principle of quantum mechanics (de Broglie, 1924), all particles possess wave-like properties:

λ = h / p  
(wavelength = Planck's constant / momentum) 

For light particles like electrons, this wavelength becomes comparable to interatomic distances (~several Å), so **wave-like properties become prominent**.

### 1.2.2 Electron in a Box: Quantized Energy

Consider an electron confined in a one-dimensional "box" (length L). Solving the Schrödinger equation, energy can only take discrete values:

E_n = (n²π²ℏ²) / (2m L²) (n = 1, 2, 3, ...) 

Key points:

  * Energy is **discrete** , not continuous (quantized)
  * Even the lowest energy (n=1) is not zero (zero-point energy)
  * The larger the box (L↑), the narrower the energy spacing (E_n+1 - E_n ∝ 1/L²)

### 1.2.3 Extension to Three Dimensions: Actual Metals

Considering a real metal as a box with volume V = L³, electron states are specified by three quantum numbers (n_x, n_y, n_z):

E = (π²ℏ²/2mL²) × (n_x² + n_y² + n_z²) 

This equation means states with energy E are distributed on a **spherical surface in 3D n-space with radius R = √(n_x² + n_y² + n_z²)**.

### 1.2.4 Fermi Energy - The Water Level of the Electron Sea

Electrons in metals are **fermions** (obeying Pauli's exclusion principle), so at most 2 electrons (spin ↑↓) can occupy one quantum state.

When N electrons fill from the lowest energy state upward, the energy of the highest energy electron at absolute zero (T=0K) is called the **Fermi energy E_F** :

E_F = (ℏ²/2m) × (3π²n)^(2/3) 

Here, n = N/V is the electron density.

**✅ Important Numerical Example**

**Copper (Cu)** :

  * Electron density: n ≈ 8.45 × 10²² cm⁻³
  * Fermi energy: E_F ≈ 7.0 eV
  * Fermi temperature: T_F = E_F/k_B ≈ 81,000 K

Room temperature (300K) is only 0.4% of T_F. This is **why the electronic specific heat is small**. Only a small fraction (~T/T_F) of electrons are thermally excited.

### 1.2.5 Density of States (DOS)

The function indicating how many quantum states exist in the energy range from E to E+dE is called the **density of states D(E)**. In the free electron model:

D(E) = (V/2π²) × (2m/ℏ²)^(3/2) × √E 

Key characteristics:

  * D(E) ∝ √E : Proportional to the square root of energy
  * Few states at low energy
  * More states at higher energy

**💡 Pro Tip**

The density of states is extremely important for understanding material properties. Many physical quantities such as electrical conductivity, specific heat, and magnetic susceptibility are proportional to D(E_F) (density of states at the Fermi energy).

## 1.3 Let's Calculate with Python - Fermi Energy and Density of States

### Example 1: Basic Calculation - Fermi Energy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import constants
    
    # Physical constants
    hbar = constants.hbar  # Reduced Planck constant (J·s)
    m_e = constants.m_e    # Electron mass (kg)
    e = constants.e        # Elementary charge (C)
    
    def fermi_energy(n):
        """
        Calculate Fermi energy in the free electron model
    
        Args:
            n (float): Electron density [m^-3]
    
        Returns:
            float: Fermi energy [eV]
        """
        E_F = (hbar**2 / (2 * m_e)) * (3 * np.pi**2 * n)**(2/3)
        return E_F / e  # Convert J to eV
    
    # Typical metal electron densities
    metals = {
        'Li': 4.70e28,   # Lithium
        'Na': 2.65e28,   # Sodium
        'Cu': 8.45e28,   # Copper
        'Ag': 5.85e28,   # Silver
        'Au': 5.90e28    # Gold
    }
    
    print("Fermi Energy of Metals")
    print("-" * 40)
    for metal, n in metals.items():
        E_F = fermi_energy(n)
        T_F = E_F * e / constants.k  # Fermi temperature [K]
        print(f"{metal:3s}: E_F = {E_F:5.2f} eV, T_F = {T_F/1000:5.1f} × 10³ K")
    
    # Output example:
    # Fermi Energy of Metals
    # ----------------------------------------
    # Li : E_F =  4.74 eV, T_F =  55.0 × 10³ K
    # Na : E_F =  3.24 eV, T_F =  37.6 × 10³ K
    # Cu : E_F =  7.00 eV, T_F =  81.2 × 10³ K
    # Ag : E_F =  5.49 eV, T_F =  63.7 × 10³ K
    # Au : E_F =  5.53 eV, T_F =  64.2 × 10³ K
    

### Example 2: Visualizing Density of States
    
    
    def density_of_states(E, V):
        """
        3D free electron density of states
    
        Args:
            E (array): Energy [J]
            V (float): Volume [m^3]
    
        Returns:
            array: Density of states [J^-1 m^-3]
        """
        factor = V / (2 * np.pi**2) * (2 * m_e / hbar**2)**(3/2)
        DOS = factor * np.sqrt(E)
        return DOS
    
    # Copper parameters
    n_Cu = 8.45e28  # m^-3
    E_F_Cu = fermi_energy(n_Cu) * e  # J
    V = 1e-6  # 1 mm³
    
    # Energy range (0 to 2E_F)
    E = np.linspace(0.01 * E_F_Cu, 2 * E_F_Cu, 1000)
    DOS = density_of_states(E, V)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(E / e, DOS * e, 'b-', linewidth=2, label='D(E)')
    plt.axvline(E_F_Cu / e, color='r', linestyle='--', linewidth=2, label=f'E_F = {E_F_Cu/e:.2f} eV')
    plt.fill_between(E / e, 0, DOS * e, where=(E <= E_F_Cu), alpha=0.3, color='blue', label='Occupied states (T=0K)')
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Density of States (eV⁻¹ mm⁻³)', fontsize=12)
    plt.title('3D Free Electron Model: Density of States', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Verify total electron count at absolute zero
    occupied_DOS = DOS[E <= E_F_Cu]
    dE = E[1] - E[0]
    N_electrons = 2 * np.sum(occupied_DOS) * dE  # Factor of 2 for spin
    print(f"Calculated electron count: {N_electrons:.2e}")
    print(f"Expected value (n×V): {n_Cu * V:.2e}")
    # These should match
    

### Example 3: Fermi-Dirac Distribution and Temperature Effects
    
    
    def fermi_dirac(E, E_F, T):
        """
        Fermi-Dirac distribution function
    
        Args:
            E (array): Energy [J]
            E_F (float): Fermi energy [J]
            T (float): Temperature [K]
    
        Returns:
            array: Occupation probability [0, 1]
        """
        if T == 0:
            return (E <= E_F).astype(float)
        else:
            k_B = constants.k
            return 1 / (1 + np.exp((E - E_F) / (k_B * T)))
    
    # For copper
    E = np.linspace(0, 2 * E_F_Cu, 1000)
    temperatures = [0, 300, 1000, 3000, 10000]  # K
    
    plt.figure(figsize=(12, 5))
    
    # (a) Fermi-Dirac distribution
    plt.subplot(1, 2, 1)
    for T in temperatures:
        f = fermi_dirac(E, E_F_Cu, T)
        label = f'T = {T} K' if T > 0 else 'T = 0 K'
        plt.plot(E / e, f, linewidth=2, label=label)
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Occupation Probability f(E)', fontsize=12)
    plt.title('Fermi-Dirac Distribution', fontsize=14, fontweight='bold')
    plt.axvline(E_F_Cu / e, color='k', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # (b) Effectively excited electrons (changing portion at T > 0)
    plt.subplot(1, 2, 2)
    E_range = np.linspace(E_F_Cu - 0.5*e, E_F_Cu + 0.5*e, 500)
    for T in [300, 1000, 3000]:
        f = fermi_dirac(E_range, E_F_Cu, T)
        thermal_width = 2 * constants.k * T / e  # Energy width ~k_B T
        plt.plot(E_range / e, f, linewidth=2, label=f'T = {T} K (Δ ≈ {thermal_width*1000:.1f} meV)')
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Occupation Probability f(E)', fontsize=12)
    plt.title('Thermal Broadening near E_F', fontsize=14, fontweight='bold')
    plt.axvline(E_F_Cu / e, color='k', linestyle='--', alpha=0.5)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Fraction of excited electrons at room temperature
    T_room = 300  # K
    thermal_energy = constants.k * T_room
    fraction = T_room / (E_F_Cu / constants.k)
    print(f"\nAnalysis at room temperature ({T_room} K):")
    print(f"Thermal energy k_B T ≈ {thermal_energy/e*1000:.1f} meV")
    print(f"Fermi energy E_F = {E_F_Cu/e:.2f} eV")
    print(f"Fraction of excited electrons: ~{fraction*100:.2f}%")
    

**⚠️ Important Note**

While the free electron model succeeds in many qualitative predictions, it cannot explain:

  * **Existence of insulators** : Why some materials with electrons don't conduct electricity
  * **Band gaps in semiconductors** : Where do forbidden bands come from
  * **Effective mass variation** : Why electron mass varies by material

To understand these, we need to consider the **periodic potential of the crystal**.

## 1.4 Band Structure Formation - Effects of Periodic Potential

### 1.4.1 Why Do Bands Form?

Electrons in crystals move in a **periodic potential** created by regularly arranged atomic nuclei. This periodicity causes dramatic changes in the electronic energy structure.

**💡 Intuitive Understanding**

In isolated atoms, electrons have discrete energy levels (1s, 2s, 2p, ...). When N atoms come together to form a crystal:

  1. Each level splits into N levels (due to interaction)
  2. Since N ~ 10²³ is very large, levels form a practically continuous "band"
  3. However, **band gaps** remain between bands originating from different atomic orbitals

### 1.4.2 Bloch's Theorem

The solution to Schrödinger's equation in a periodic potential V(r + R) = V(r) takes a special form called a **Bloch function** :

ψ_nk(r) = e^(ik·r) u_nk(r) 

Where:

  * k: **Wave vector** (corresponding to crystal momentum of electron, ℏk)
  * u_nk(r): Function with crystal periodicity (u_nk(r + R) = u_nk(r))
  * n: **Band index** (from 1s, 2p, etc.)

Energy depends on k and n: **E = E_n(k)**

### 1.4.3 Brillouin Zone and Dispersion Relation

In a simple one-dimensional example (lattice constant a), the wave vector k is physically independent in the **first Brillouin zone** -π/a ≤ k ≤ π/a.

**✅ Important Concept**

**What happens at the Brillouin zone boundary (k = ±π/a)?**

For free electrons, E ∝ k² varies smoothly, but with periodic potential:

  * An **energy gap** opens at k = π/a
  * Due to electron waves undergoing **Bragg reflection** from the crystal lattice
  * This gap is the origin of the **band gap**

### 1.4.4 Simple Model: Band Structure of 1D Crystal

In the simplest "nearly free electron approximation," treating the periodic potential as a weak perturbation, the dispersion relation changes as follows:

E_±(k) = (ℏ²k²)/(2m) ± |V_G| × √(1 + (ℏ²k·G)/(m|V_G|)²) 

Where V_G is the Fourier component of the periodic potential and G is the reciprocal lattice vector.

### Example 4: Plotting 1D Band Structure
    
    
    def band_structure_1d(k, a, V_G):
        """
        Band structure of 1D nearly free electron model
    
        Args:
            k (array): Wave vector [m^-1]
            a (float): Lattice constant [m]
            V_G (float): Periodic potential [J]
    
        Returns:
            tuple: (E_lower, E_upper) Lower and upper bands [J]
        """
        E_free = (hbar * k)**2 / (2 * m_e)  # Free electron
    
        # Gap at Brillouin zone boundary
        G = 2 * np.pi / a
        gap = 2 * V_G
    
        # Band structure with perturbation (simplified)
        cos_term = np.cos(k * a)
        E_lower = E_free - V_G * np.abs(cos_term)
        E_upper = E_free + V_G * np.abs(cos_term)
    
        return E_lower, E_upper
    
    # Parameters
    a = 3e-10  # 3 Å lattice constant
    V_G = 2 * e  # 2 eV potential
    
    # First Brillouin zone
    k = np.linspace(-np.pi/a, np.pi/a, 500)
    E_lower, E_upper = band_structure_1d(k, a, V_G)
    
    # Free electron (for comparison)
    E_free = (hbar * k)**2 / (2 * m_e)
    
    # Plot
    plt.figure(figsize=(10, 7))
    plt.plot(k * a / np.pi, E_free / e, 'k--', linewidth=1.5, alpha=0.5, label='Free electron')
    plt.plot(k * a / np.pi, E_lower / e, 'b-', linewidth=2.5, label='Lower band')
    plt.plot(k * a / np.pi, E_upper / e, 'r-', linewidth=2.5, label='Upper band')
    
    # Highlight band gap region
    k_gap = np.pi / a
    E_gap_lower = band_structure_1d(np.array([k_gap]), a, V_G)[0][0] / e
    E_gap_upper = band_structure_1d(np.array([k_gap]), a, V_G)[1][0] / e
    plt.fill_between([-1, 1], E_gap_lower, E_gap_upper, alpha=0.2, color='gray', label='Band gap')
    
    plt.xlabel('Wave vector k (π/a)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title('1D Band Structure: Nearly Free Electron Model', fontsize=14, fontweight='bold')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(-1, color='k', linewidth=0.5, linestyle=':')
    plt.axvline(1, color='k', linewidth=0.5, linestyle=':')
    plt.text(0.7, E_gap_upper + 0.5, f'Band gap\n~{(E_gap_upper - E_gap_lower):.2f} eV', fontsize=10, ha='center')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(alpha=0.3)
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.show()
    

## 1.5 Revisiting Density of States (DOS) - Calculation from Band Structure

### 1.5.1 General Definition of Density of States

Given band structure E_n(k), the density of states at energy E is:

D(E) = Σ_n ∫ δ(E - E_n(k)) (dk/(2π)³) 

In actual calculations, we compute the area of the constant energy surface E_n(k) = E.

### 1.5.2 Features of DOS from Band Structure

Unlike free electron D(E) ∝ √E, systems with band structure exhibit:

  * **Band gaps** : Regions where D(E) = 0 (forbidden bands)
  * **Van Hove singularities** : D(E) increases divergently at points where ∇_k E = 0
  * **Band edges** : Characteristic changes in D(E) at band maxima and minima

### Example 5: Calculating Density of States from Band Structure
    
    
    def dos_from_band(E_array, E_band, V):
        """
        Calculate density of states from band structure (histogram method)
    
        Args:
            E_array (array): Energy axis [J]
            E_band (array): Band energy [J] (k-point array for 1D)
            V (float): Volume [m^3]
    
        Returns:
            array: Density of states [J^-1 m^-3]
        """
        hist, bin_edges = np.histogram(E_band, bins=E_array, density=False)
        dE = E_array[1] - E_array[0]
        # Normalization for 1D (different for 3D)
        DOS = hist / (dE * V) * 2  # Spin factor 2
        return DOS
    
    # 1D band structure example
    k = np.linspace(-np.pi/a, np.pi/a, 10000)
    E_lower, E_upper = band_structure_1d(k, a, V_G)
    
    # Energy axis
    E_axis = np.linspace(-2*e, 20*e, 500)
    
    # Calculate DOS for each band
    DOS_lower = dos_from_band(E_axis, E_lower, 1e-9)
    DOS_upper = dos_from_band(E_axis, E_upper, 1e-9)
    DOS_total = DOS_lower + DOS_upper
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) Band structure
    ax1.plot(k * a / np.pi, E_lower / e, 'b-', linewidth=2.5, label='Lower band')
    ax1.plot(k * a / np.pi, E_upper / e, 'r-', linewidth=2.5, label='Upper band')
    ax1.set_xlabel('Wave vector k (π/a)', fontsize=12)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_title('(a) Band Structure', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-1, 1)
    
    # (b) Density of states
    E_plot = E_axis / e
    ax2.plot(DOS_total * e, E_plot, 'k-', linewidth=2.5, label='Total DOS')
    ax2.fill_betweenx(E_plot, 0, DOS_total * e, alpha=0.3, color='blue')
    ax2.set_xlabel('Density of States (arb. units)', fontsize=12)
    ax2.set_ylabel('Energy (eV)', fontsize=12)
    ax2.set_title('(b) Density of States', fontsize=13, fontweight='bold')
    ax2.axhline(E_gap_lower, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(E_gap_upper, color='gray', linestyle='--', alpha=0.5)
    ax2.text(0.7 * np.max(DOS_total * e), (E_gap_lower + E_gap_upper) / 2,
             'Band gap\n(D=0)', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(E_plot[0], E_plot[-1])
    
    plt.tight_layout()
    plt.show()
    

## 1.6 Classification of Metals, Semiconductors, and Insulators

### 1.6.1 Electrical Properties Determined by Band Occupation

A material's conductivity is determined by **where the Fermi energy E_F is located in the band structure**.

Material Type | Band Occupation | Band Gap | Conductivity (room temp) | Typical Examples  
---|---|---|---|---  
**Metal** | Valence band partially occupied  
or multiple bands overlap | None (or negative) | 10⁴-10⁶ S/cm | Cu, Al, Na  
**Semiconductor** | Valence band fully occupied  
Conduction band empty | Small (0.1-3 eV) | 10⁻⁶-10² S/cm | Si, GaAs, GaN  
**Insulator** | Valence band fully occupied  
Conduction band empty | Large (>3 eV) | <10⁻¹⁰ S/cm | SiO₂, Al₂O₃  
  
### 1.6.2 Why Do Metals Conduct Electricity?

In metals:

  1. E_F is located **within** a band
  2. **Occupied and empty states coexist** near E_F
  3. Under electric field, electrons near E_F easily move to empty states (=current)
  4. Having D(E_F) > 0 is essential

**💡 Concrete Example: Sodium (Na)**

Na atom has electron configuration [Ne] 3s¹. In the crystal:

  * A 3s band forms, but since each atom has only one 3s electron, the **band is only half filled**
  * Therefore E_F is located in the middle of the band
  * → Excellent metallic conductivity (σ ≈ 2.1 × 10⁵ S/cm)

### 1.6.3 Why Don't Insulators Conduct Electricity?

In insulators (e.g., diamond):

  1. Valence electrons exactly **completely fill** the valence band
  2. E_F is **between the valence band top and conduction band bottom (within the band gap)**
  3. The band gap is large (diamond: 5.5 eV)
  4. Room temperature thermal energy (~0.026 eV) cannot excite electrons to the conduction band
  5. → No conductivity (σ < 10⁻¹⁴ S/cm)

### 1.6.4 Semiconductors - Intermediate Properties

Semiconductors have a structure similar to insulators but with a small band gap (Si: 1.1 eV, GaAs: 1.4 eV), so:

  * **Intrinsic conduction** : A few electrons are thermally excited to the conduction band (carrier density ∝ exp(-E_g/2k_BT))
  * **Impurity conduction** : Carrier density controllable by doping
  * **Temperature dependence** : Conductivity increases exponentially with temperature (opposite to metals)

**✅ Numerical Example: Si Intrinsic Carrier Density**

Silicon (T = 300 K):

  * Band gap: E_g = 1.12 eV
  * Intrinsic carrier density: n_i ≈ 1.5 × 10¹⁰ cm⁻³
  * Comparison: Si atom density 5 × 10²² cm⁻³ → Only 1 in a trillion excited

By doping (e.g., 10¹⁶ cm⁻³ phosphorus), carrier density can be increased by a million times.

### Example 6: Comparing Band Structures of Metals, Semiconductors, and Insulators
    
    
    def plot_band_occupation():
        """
        Visualize band occupation states of metals, semiconductors, and insulators
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
        E_range = np.linspace(-3, 3, 100)
    
        # (a) Metal
        ax = axes[0]
        # Overlapping bands
        ax.fill_between([-1, 1], -2, 0, alpha=0.6, color='blue', label='Valence band (filled)')
        ax.fill_between([-1, 1], 0, 2, alpha=0.3, color='blue', label='Conduction band (partial)')
        ax.axhline(0.5, color='red', linewidth=2, linestyle='--', label='Fermi level (E_F)')
        ax.text(0, 0.7, 'E_F', fontsize=12, color='red', ha='center', fontweight='bold')
        ax.set_ylim(-3, 3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title('(a) Metal', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xticks([])
    
        # (b) Semiconductor
        ax = axes[1]
        ax.fill_between([-1, 1], -2, -0.5, alpha=0.6, color='blue', label='Valence band (filled)')
        ax.fill_between([-1, 1], 0.5, 2, alpha=0.3, color='lightblue', label='Conduction band (empty)')
        ax.axhline(0, color='red', linewidth=2, linestyle='--', label='Fermi level (E_F)')
        ax.text(0, 0.15, 'E_F', fontsize=12, color='red', ha='center', fontweight='bold')
        # Band gap
        ax.annotate('', xy=(1.2, -0.5), xytext=(1.2, 0.5),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(1.5, 0, 'E_g\n~1 eV', fontsize=11, va='center', ha='left')
        ax.set_ylim(-3, 3)
        ax.set_xlim(-1.5, 1.8)
        ax.set_title('(b) Semiconductor', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
        # (c) Insulator
        ax = axes[2]
        ax.fill_between([-1, 1], -2, -1, alpha=0.6, color='blue', label='Valence band (filled)')
        ax.fill_between([-1, 1], 2, 3, alpha=0.3, color='lightblue', label='Conduction band (empty)')
        ax.axhline(0.5, color='red', linewidth=2, linestyle='--', label='Fermi level (E_F)')
        ax.text(0, 0.7, 'E_F', fontsize=12, color='red', ha='center', fontweight='bold')
        # Band gap
        ax.annotate('', xy=(1.2, -1), xytext=(1.2, 2),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(1.5, 0.5, 'E_g\n>3 eV', fontsize=11, va='center', ha='left')
        ax.set_ylim(-3, 3.5)
        ax.set_xlim(-1.5, 1.8)
        ax.set_title('(c) Insulator', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
        plt.tight_layout()
        plt.show()
    
    plot_band_occupation()
    

## 1.7 Fermi Surface - Understanding 3D Band Structure

### 1.7.1 What is the Fermi Surface?

In 3D **k-space** (wave vector space, also called reciprocal lattice space), the set of points satisfying E_n(k) = E_F is called the **Fermi surface**.

Why it's important:

  * Many properties like electrical conduction, magnetism, and superconductivity are carried by electrons near the Fermi surface
  * The **shape (topology)** of the Fermi surface determines material properties
  * Experimentally measurable (de Haas-van Alphen effect, ARPES, etc.)

### 1.7.2 Free Electron Fermi Surface - Perfect Sphere

In the free electron model with E = ℏ²k²/(2m), k satisfying E = E_F is:

|k| = k_F = √(2mE_F)/ℏ = (3π²n)^(1/3) 

That is, the Fermi surface is a **sphere of radius k_F** (Fermi sphere).

### 1.7.3 Actual Metal Fermi Surfaces - Complex Shapes

Due to the crystal's periodic potential, Fermi surfaces distort from spherical:

  * **Copper (Cu)** : Nearly spherical but with "necks" formed at Brillouin zone boundaries
  * **Gold (Au)** : Complex shape with protrusions at cube corners
  * **Iron (Fe)** : Complex multi-sheet structure related to magnetism

### Example 7: Visualizing 2D Fermi Surface
    
    
    def fermi_surface_2d():
        """
        Calculate Fermi surface from simple 2D band structure
        """
        # 2D grid
        kx = np.linspace(-np.pi/a, np.pi/a, 200)
        ky = np.linspace(-np.pi/a, np.pi/a, 200)
        KX, KY = np.meshgrid(kx, ky)
    
        # Simple band structure (tight-binding model)
        t = 2 * e  # hopping parameter
        E = -2 * t * (np.cos(KX * a) + np.cos(KY * a))
    
        # Fermi surfaces at several Fermi energies
        E_F_values = [-3*t, -2*t, 0, 2*t]
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    
        for idx, E_F in enumerate(E_F_values):
            ax = axes[idx]
    
            # Band structure contour
            contour = ax.contourf(KX*a/np.pi, KY*a/np.pi, E/t, levels=50, cmap='RdBu_r', alpha=0.7)
    
            # Fermi surface (E = E_F contour line)
            ax.contour(KX*a/np.pi, KY*a/np.pi, E/t, levels=[E_F/t], colors='black', linewidths=3)
    
            # Brillouin zone boundary
            ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', linewidth=2, alpha=0.5)
    
            ax.set_xlabel('k_x (π/a)', fontsize=11)
            ax.set_ylabel('k_y (π/a)', fontsize=11)
            ax.set_title(f'E_F = {E_F/t:.1f}t', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
    
            # Colorbar
            plt.colorbar(contour, ax=ax, label='E/t')
    
        plt.suptitle('Fermi Surface Evolution in 2D Tight-Binding Model',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
    
    fermi_surface_2d()
    

### Example 8: 3D Fermi Surface (Simple Example)
    
    
    from mpl_toolkits.mplot3d import Axes3D
    
    def fermi_surface_3d_sphere():
        """
        3D plot of free electron Fermi surface (sphere)
        """
        # Copper example
        n_Cu = 8.45e28  # m^-3
        k_F = (3 * np.pi**2 * n_Cu)**(1/3)
    
        # Sphere parameters
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        kx = k_F * np.outer(np.cos(u), np.sin(v))
        ky = k_F * np.outer(np.sin(u), np.sin(v))
        kz = k_F * np.outer(np.ones(np.size(u)), np.cos(v))
    
        # Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.plot_surface(kx * 1e-10, ky * 1e-10, kz * 1e-10,
                        color='cyan', alpha=0.6, edgecolor='navy', linewidth=0.1)
    
        # Brillouin zone boundary (cube)
        G = np.pi / a
        # Draw cube edges
        for i in [-1, 1]:
            for j in [-1, 1]:
                ax.plot([i*G*1e-10, i*G*1e-10], [j*G*1e-10, j*G*1e-10],
                       [-G*1e-10, G*1e-10], 'k--', alpha=0.3)
                ax.plot([i*G*1e-10, i*G*1e-10], [-G*1e-10, G*1e-10],
                       [j*G*1e-10, j*G*1e-10], 'k--', alpha=0.3)
                ax.plot([-G*1e-10, G*1e-10], [i*G*1e-10, i*G*1e-10],
                       [j*G*1e-10, j*G*1e-10], 'k--', alpha=0.3)
    
        ax.set_xlabel('k_x (Å⁻¹)', fontsize=12)
        ax.set_ylabel('k_y (Å⁻¹)', fontsize=12)
        ax.set_zlabel('k_z (Å⁻¹)', fontsize=12)
        ax.set_title('Fermi Surface of Copper (Free Electron Approximation)',
                     fontsize=14, fontweight='bold')
    
        # Isotropic aspect ratio
        max_range = G * 1e-10
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
        plt.tight_layout()
        plt.show()
    
        print(f"Copper Fermi wave vector: k_F = {k_F:.2e} m⁻¹ = {k_F*1e-10:.2f} Å⁻¹")
    
    fermi_surface_3d_sphere()
    

### 1.7.4 Fermi Surface and Transport Phenomena

Geometric properties of the Fermi surface directly affect material properties:

Fermi Surface Feature | Physical Effect | Typical Example  
---|---|---  
Nearly spherical | Isotropic conductivity | Cu, Al  
Ellipsoidal | Anisotropic conductivity | Bi (100x difference by crystal direction)  
Nesting  
(parallel sections) | Charge density waves, superconductivity | Layered compounds  
Topological changes | Lifshitz transition | Metals under pressure  
  
## 1.8 Experimental Verification - Is Band Theory Correct?

### 1.8.1 Angle-Resolved Photoemission Spectroscopy (ARPES)

The most direct method to measure band structure:

  * Irradiate sample with light to knock out electrons (photoelectric effect)
  * Measure energy and angle of emitted electrons
  * Directly determine E(k) from energy conservation

**✅ Measurement Precision (Latest ARPES)**

  * Energy resolution: ~1 meV
  * Angular resolution: ~0.1°
  * Time-resolved ARPES: femtosecond (10⁻¹⁵ s) scale

This enables observation of superconducting gaps (~1 meV) and charge ordering formation processes.

### 1.8.2 Optical Measurements - Determining Band Gaps

Band gaps in semiconductors can be precisely measured by optical absorption:

α(ω) ∝ √(ℏω - E_g) (for direct transitions) 

Where α is the absorption coefficient and ω is the angular frequency of light.

### Example 9: Estimating Band Gap from Optical Absorption
    
    
    def optical_absorption(photon_energy, E_g, A=1.0, indirect=False):
        """
        Calculate optical absorption coefficient of semiconductor
    
        Args:
            photon_energy (array): Photon energy [eV]
            E_g (float): Band gap [eV]
            A (float): Proportionality constant
            indirect (bool): True for indirect transition
    
        Returns:
            array: Absorption coefficient [arbitrary units]
        """
        alpha = np.zeros_like(photon_energy)
        above_gap = photon_energy > E_g
    
        if indirect:
            # Indirect transition: α ∝ (ℏω - E_g)²
            alpha[above_gap] = A * (photon_energy[above_gap] - E_g)**2
        else:
            # Direct transition: α ∝ √(ℏω - E_g)
            alpha[above_gap] = A * np.sqrt(photon_energy[above_gap] - E_g)
    
        return alpha
    
    # Typical semiconductor parameters
    semiconductors = {
        'GaN': {'E_g': 3.4, 'type': 'direct'},
        'Si': {'E_g': 1.12, 'type': 'indirect'},
        'GaAs': {'E_g': 1.42, 'type': 'direct'},
        'Ge': {'E_g': 0.66, 'type': 'indirect'}
    }
    
    # Plot
    photon_energy = np.linspace(0, 4, 500)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, params) in enumerate(semiconductors.items()):
        ax = axes[idx]
        E_g = params['E_g']
        indirect = (params['type'] == 'indirect')
    
        alpha = optical_absorption(photon_energy, E_g, A=1.0, indirect=indirect)
    
        ax.plot(photon_energy, alpha, 'b-', linewidth=2.5)
        ax.axvline(E_g, color='r', linestyle='--', linewidth=2, label=f'E_g = {E_g:.2f} eV')
        ax.fill_between(photon_energy, 0, alpha, where=(photon_energy > E_g), alpha=0.3, color='blue')
    
        ax.set_xlabel('Photon Energy (eV)', fontsize=11)
        ax.set_ylabel('Absorption Coefficient α (arb. units)', fontsize=11)
        ax.set_title(f'{name} ({params["type"]} gap)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, np.max(alpha) * 1.1)
    
    plt.suptitle('Optical Absorption Spectra of Semiconductors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Tauc plot for band gap determination (experimental data analysis method)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, params in semiconductors.items():
        E_g = params['E_g']
        indirect = (params['type'] == 'indirect')
        alpha = optical_absorption(photon_energy, E_g, A=1.0, indirect=indirect)
    
        # Tauc plot: (αhν)^(1/n) vs hν (n=1/2 for direct, n=2 for indirect)
        if indirect:
            y_axis = (alpha * photon_energy)**0.5  # (αhν)^1/2
        else:
            y_axis = (alpha * photon_energy)**2  # (αhν)^2
    
        ax.plot(photon_energy, y_axis, linewidth=2, label=name)
        ax.axvline(E_g, linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Photon Energy (eV)', fontsize=12)
    ax.set_ylabel('(αhν)^n (arb. units)', fontsize=12)
    ax.set_title('Tauc Plot for Band Gap Determination', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 4)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 How to use Tauc plot:")
    print("Extrapolate the rising edge linearly and find the intersection with the x-axis;")
    print("that corresponds to the band gap E_g.")
    

### 1.8.3 Quantum Oscillations - Precise Measurement of Fermi Surface

The **de Haas-van Alphen effect** is a phenomenon where magnetization oscillates in a strong magnetic field, enabling precise measurement of Fermi surface cross-sections:

Oscillation period ∝ 1/A  
(A: extremal cross-sectional area of Fermi surface perpendicular to magnetic field) 

This method can determine Fermi surface topology, effective mass, scattering rate, etc.

## Learning Objectives Verification

Having completed this chapter, you can now explain:

### Fundamental Understanding

  * ✅ The concepts of the free electron model and Fermi energy
  * ✅ The formation mechanism of band structures and the origin of band gaps
  * ✅ Differences in electronic structures of metals, semiconductors, and insulators

### Practical Skills

  * ✅ Calculate Fermi energy and density of states using Python
  * ✅ Plot and interpret simple band structure diagrams
  * ✅ Visualize and understand the concept of Fermi surfaces

### Application Capabilities

  * ✅ Estimate band gaps from experimental data
  * ✅ Predict electrical properties of materials using band theory
  * ✅ Possess foundational knowledge to design electronic structures of new materials

## Exercises

### Easy (Basic Verification)

Q1: Which is the most appropriate definition of Fermi energy?

a) Average energy of electrons in a metal  
b) Energy of the highest energy electron at absolute zero  
c) Minimum energy of the conduction band  
d) Energy at the center of the band gap

See Answer

**Correct: b) Energy of the highest energy electron at absolute zero**

**Explanation:**  
Fermi energy E_F is the energy of the highest energy electron when electrons fill from the lowest energy state upward at T=0K. Fermions (electrons) obey Pauli's exclusion principle, so at most 2 electrons (spin ↑↓) can occupy one state.

**Additional note:**  
Options c) and d) describe semiconductors or insulators, not metals. Option a) gives an average energy lower than E_F (approximately 3E_F/5 at T=0K).

Q2: Copper (Cu) has a Fermi energy of about 7.0 eV. What is the Fermi temperature T_F in K? (k_B = 8.617 × 10⁻⁵ eV/K)

See Answer

**Correct: Approximately 81,000 K**

**Calculation:**  
T_F = E_F / k_B = 7.0 eV / (8.617 × 10⁻⁵ eV/K) ≈ 81,200 K

**Key Point:**  
Room temperature (300 K) is only 0.4% of T_F. This is why the electronic specific heat is much smaller than classical predictions (3/2 Nk_B). Only electrons near the Fermi energy (within ~kT range) are actually thermally excited.

Q3: Explain the difference in band structure between metals and insulators.

See Answer

**Sample Answer:**

  * **Metal:** Fermi energy E_F is located within a band, with partially occupied bands. Or multiple bands overlap. Since D(E_F) > 0, carriers can easily move under electric field, providing conductivity.
  * **Insulator:** Valence band is completely occupied, conduction band is completely empty. E_F is located within the band gap (forbidden band). Since the band gap is large (>3 eV), electrons cannot be excited to the conduction band at room temperature, providing no conductivity.

**Key Point:**  
Whether a material is a metal or insulator is determined not by the shape of the bands but by "how filled the bands are with electrons." The same band structure with different electron counts will have different properties.

### Medium (Application)

Q4: Silicon (Si) has a band gap of 1.12 eV. Can it absorb visible light (wavelength 400-700 nm)? Calculate and explain. (h = 4.136 × 10⁻¹⁵ eV·s, c = 3 × 10⁸ m/s)

See Answer

**Conclusion: Can absorb some visible light (blue to violet) but not red**

**Calculation:**

Photon energy E = hc/λ :

  * λ = 400 nm (violet): E = (4.136 × 10⁻¹⁵ eV·s × 3 × 10⁸ m/s) / (400 × 10⁻⁹ m) = 3.10 eV
  * λ = 550 nm (green): E = 2.25 eV
  * λ = 700 nm (red): E = 1.77 eV

Since E_g = 1.12 eV:

  * Violet to green (E > 2 eV): Absorbed (E > E_g)
  * Red (E = 1.77 eV): Partially absorbed
  * Near-infrared (E < 1.12 eV): Transmitted (not absorbed)

**Practical Meaning:**  
This is one reason Si is used as a solar cell material. It covers the main part of sunlight (visible to near-infrared). However, since infrared is transmitted, tandem solar cells combining Si with materials having smaller band gaps like Ge (0.66 eV) or GaAs (1.42 eV) are also being researched.

Q5: In the 3D free electron model, the density of states is D(E) ∝ √E. Qualitatively explain how this leads to the electronic specific heat C_v ∝ T at the Fermi energy.

See Answer

**Explanation:**

  1. **Number of excited electrons** : At temperature T, only electrons within an energy range ~k_BT around E_F are thermally excited. This number is proportional to D(E_F) × k_BT.
  2. **Energy gained per electron** : On average ~k_BT.
  3. **Total energy** : U ~ [D(E_F) × k_BT] × k_BT = D(E_F) × (k_BT)²
  4. **Specific heat** : C_v = dU/dT ~ D(E_F) × k_B² × T ∝ T

**Important points:**

  * In classical theory, all electrons (N total) are excited, so C_v ~ Nk_B (temperature independent)
  * In quantum theory, only N × (T/T_F) electrons are excited, so C_v is proportional to T with a coefficient (T/T_F) times smaller
  * At room temperature T/T_F ~ 0.004, so electronic specific heat is much smaller than lattice vibration specific heat (~3Nk_B) and can be ignored

Q6: Modify the code in Example 1 to calculate and plot Fermi energy and Fermi temperature for alkali metals (Li, Na, K, Rb, Cs) from their electron densities. Discuss how E_F changes with increasing atomic number.

See Answer

**Code Example:**
    
    
    alkali_metals = {
        'Li': 4.70e28,
        'Na': 2.65e28,
        'K': 1.40e28,
        'Rb': 1.15e28,
        'Cs': 0.91e28
    }
    
    names = list(alkali_metals.keys())
    E_F_values = [fermi_energy(n) for n in alkali_metals.values()]
    T_F_values = [E_F * e / constants.k / 1000 for E_F in E_F_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(names, E_F_values, color='skyblue', edgecolor='navy', linewidth=2)
    ax1.set_ylabel('Fermi Energy (eV)', fontsize=12)
    ax1.set_title('Fermi Energy of Alkali Metals', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(names, T_F_values, color='salmon', edgecolor='darkred', linewidth=2)
    ax2.set_ylabel('Fermi Temperature (10³ K)', fontsize=12)
    ax2.set_title('Fermi Temperature of Alkali Metals', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Discussion:**

  * Since E_F ∝ n^(2/3), lower electron density leads to smaller E_F
  * As atomic number increases (Li → Cs), atoms become larger and electron density n decreases
  * Therefore Li (smallest atom) has the highest E_F (4.74 eV)
  * Cs (largest atom) has the lowest E_F (~1.59 eV)
  * This correlates with softness (low melting point): Cs has low E_F, weak bonding, melting point 28.5°C

### Hard (Advanced)

Q7: Derive the density of states for 2D materials (such as graphene) in the free electron model. Discuss the difference from the 3D case D(E) ∝ √E.

See Answer

**Derivation:**

In 2D, wave vector space is a plane (k_x, k_y). Free electron energy:

E = ℏ²k²/(2m) = ℏ²(k_x² + k_y²)/(2m) 

Density of states in k-space (per unit area): 1/(2π)² = 1/(4π²)

States with energy E lie on a circle of radius k = √(2mE/ℏ²).

k-width corresponding to energy width dE: dk = (m/ℏ²k) dE

Number of states: dN = [circumference 2πk] × dk × [1/(4π²)] × 2 (spin)

D(E) = dN/dE = (2πk × dk/dE) / (2π²) = (2m)/(2πℏ²) = constant 

**Key differences:**

Dimension | D(E) | Feature  
---|---|---  
1D | ∝ E^(-1/2) | Diverges at low energy  
2D | = constant | Energy independent  
3D | ∝ E^(1/2) | Increases at high energy  
  
**Physical meaning:**

  * 2D systems show different temperature dependence of electronic specific heat C_v from 3D
  * Graphene has Dirac dispersion (E ∝ k), not free electron, so actually D(E) ∝ |E|
  * 2D material-specific phenomena like quantum Hall effect and superconductivity are observed

Q8: Diamond (C) and graphite (C) are made of the same element, yet one is an insulator and the other is a semimetal. From a band theory perspective, explain the structural factors causing this difference.

See Answer

**Structural differences:**

  * **Diamond:** sp³ hybridization, 3D network, C-C bond distance 1.54 Å
  * **Graphite:** sp² hybridization, 2D layered structure, intralayer bonds 1.42 Å, interlayer distance 3.35 Å

**Electronic structure differences:**

**Diamond:**

  * All 4 valence electrons form σ bonds in sp³ hybrid orbitals
  * Bonding orbitals (valence band) completely occupied
  * Antibonding orbitals (conduction band) completely empty
  * Band gap E_g = 5.5 eV (large)
  * → Insulator

**Graphite:**

  * 3 valence electrons in sp² hybridization for σ bonds, 1 remaining in π orbital
  * π orbitals form continuous band (π electrons are delocalized)
  * π and π* orbitals (antibonding) just touch at K point (Brillouin zone corner)
  * Band gap E_g = 0 eV (zero gap)
  * → Semimetal (density of states near zero but finite conductivity)

**Key points:**

  1. **Hybridization difference** (sp³ vs sp²) is decisive
  2. **Dimensionality** : Graphite's 2D layer structure promotes π electron delocalization
  3. **Interlayer interaction** : Isolating one layer of graphite produces "graphene" with a unique Dirac cone-type band structure

**Practical implications:**

  * Diamond: High voltage devices, optical windows as insulator
  * Graphite: Electrode material, lubricant, lithium-ion battery anode
  * Graphene: Ultra-high-speed transistors, transparent electrodes (research stage)

Q9: (Programming task) In the tight-binding model, calculate and plot how bandwidth and band gap change when varying the nearest-neighbor hopping integral t for a 1D chain. Discuss the physical meaning.

See Answer

**Code example:**
    
    
    def tight_binding_1d(k, a, t, epsilon_0=0):
        """
        Band structure of 1D tight-binding model
    
        Args:
            k (array): Wave vector
            a (float): Lattice constant
            t (float): Hopping integral
            epsilon_0 (float): On-site energy
    
        Returns:
            array: Energy
        """
        return epsilon_0 - 2 * t * np.cos(k * a)
    
    a = 3e-10  # 3 Å
    k = np.linspace(-np.pi/a, np.pi/a, 500)
    t_values = [0.5*e, 1.0*e, 2.0*e, 4.0*e]  # eV
    
    plt.figure(figsize=(10, 7))
    
    for t in t_values:
        E = tight_binding_1d(k, a, t)
        bandwidth = np.max(E) - np.min(E)
        plt.plot(k*a/np.pi, E/e, linewidth=2.5, label=f't = {t/e:.1f} eV (BW = {bandwidth/e:.2f} eV)')
    
    plt.xlabel('Wave vector k (π/a)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title('1D Tight-Binding Model: Effect of Hopping Parameter', fontsize=14, fontweight='bold')
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # t dependence of bandwidth
    t_range = np.linspace(0.1*e, 5*e, 50)
    bandwidth = [4*t for t in t_range]  # BW = 4t (theoretical value)
    
    plt.figure(figsize=(8, 6))
    plt.plot(t_range/e, np.array(bandwidth)/e, 'b-', linewidth=3)
    plt.xlabel('Hopping parameter t (eV)', fontsize=12)
    plt.ylabel('Bandwidth (eV)', fontsize=12)
    plt.title('Bandwidth vs Hopping Parameter', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("📊 Analysis results:")
    print(f"Bandwidth = 4t (larger t gives wider band)")
    print(f"Physical meaning: Large t → electron wavefunction strongly overlaps with adjacent sites")
    print(f"                → Electrons move easily in crystal (effective mass m* ∝ 1/t)")
    

**Discussion:**

  * **Physical meaning of t (hopping integral)** : Magnitude of probability for electrons to "tunnel" to adjacent atoms
  * **Bandwidth ∝ t** : Large t → electrons move easily → wide band → light effective mass
  * **Diatomic molecule analogy** : Close interatomic distance (large overlap) → large bonding-antibonding energy difference
  * **Real materials** : Transition metals (large d-orbital extent) have large t and wide bands → high conductivity

Q10: (Advanced problem) Mott insulators are predicted to be metals by band theory but are actually insulators (e.g., NiO, La₂CuO₄). Research why band theory alone cannot explain this phenomenon and briefly summarize the role of electron correlation.

See Answer

**What are Mott insulators:**

  * Band theory: Partially occupied d-orbital band → predicts metal
  * Experiment: Insulator (gap ~1-3 eV)
  * Examples: NiO (Ni²⁺ 3d⁸ configuration), La₂CuO₄ (parent compound of high-temperature superconductors)

**Why band theory fails:**

  1. **Ignores strong electron correlation**
     * Band theory assumes electrons move independently (one-electron approximation)
     * Actually, electron-electron Coulomb repulsion U can be larger than kinetic energy (bandwidth W)
  2. **Mott-Hubbard picture**
     * Cost to put 2 electrons on one site (U) is too large
     * Electrons localize, placing one per site
     * Moving electrons requires U energy → gap formation

**Phase diagram (U vs W):**

  * **U << W**: Kinetic energy dominates → band theory valid → metal
  * **U >> W**: Coulomb interaction dominates → electron localization → Mott insulator
  * **U ~ W** : Competition region → strongly correlated systems (high-temperature superconductors, heavy fermions, etc.)

**Success of Dynamical Mean Field Theory (DMFT):**

  * Theory developed in 1990s (Georges, Kotliar et al.)
  * Incorporates electron correlation while retaining band structure
  * Can describe Mott transition (metal-insulator transition)
  * Provides integrated picture of band theory + correlation effects

**Practical importance:**

  * **High-temperature superconductivity** : Superconductivity emerges by doping Mott insulators (cuprates, iron-based superconductors)
  * **Colossal magnetoresistance** : Manganese oxides (CMR effect)
  * **Mottronics** : Proposal of switching devices utilizing Mott transition

**Conclusion:**

Band theory is a powerful framework for understanding many materials, but is insufficient for systems with strong electron correlation (transition metal oxides, f-electron systems, etc.). To understand these, methods considering many-body effects (DMFT, quantum Monte Carlo, density functional theory+U, etc.) are necessary.

## Next Steps

In Chapter 1, we learned the basic concepts of band theory as the foundation of solid-state electronic theory. In the next chapter, we will learn **methods to calculate the electronic structure of actual materials** based on this knowledge.

**Chapter 2 Preview: Introduction to Density Functional Theory (DFT)**

  * Hohenberg-Kohn and Kohn-Sham theorems
  * Exchange-correlation functionals (LDA, GGA, hybrid)
  * Hands-on with Python: Band structure calculation using ASE + GPAW
  * Applications to real materials: Band structures of Si, GaAs, TiO₂

## References

### Fundamental Textbooks

  1. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_. Holt, Rinehart and Winston. (Classic masterpiece in solid-state physics)
  2. Kittel, C. (2004). _Introduction to Solid State Physics_ (8th ed.). Wiley. (Ideal as an introductory text)
  3. Marder, M. P. (2010). _Condensed Matter Physics_ (2nd ed.). Wiley. (Modern approach)

### Band Theory Details

  4. Harrison, W. A. (1980). _Electronic Structure and the Properties of Solids_. Freeman. (Detailed tight-binding method)
  5. Yu, P. Y., & Cardona, M. (2010). _Fundamentals of Semiconductors_ (4th ed.). Springer. (Standard semiconductor physics textbook)

### Computational Physical Chemistry

  6. Martin, R. M. (2004). _Electronic Structure: Basic Theory and Practical Methods_. Cambridge University Press. (DFT theory and practice)
  7. Sholl, D., & Steckel, J. A. (2009). _Density Functional Theory: A Practical Introduction_. Wiley. (DFT introduction)

### Online Resources

  8. Materials Project. (2024). Electronic Structure Calculations. <https://materialsproject.org>
  9. ASE (Atomic Simulation Environment). <https://wiki.fysik.dtu.dk/ase/>
  10. GPAW Documentation. <https://wiki.fysik.dtu.dk/gpaw/>

### Original Papers (Historically Important)

  11. Bloch, F. (1929). "Über die Quantenmechanik der Elektronen in Kristallgittern." _Zeitschrift für Physik_ , 52(7-8), 555-600. (Bloch's theorem)
  12. Wilson, A. H. (1931). "The Theory of Electronic Semi-Conductors." _Proceedings of the Royal Society A_ , 133(822), 458-491. (Foundation of semiconductor theory)

[← Series Index](<./index.html>) [Chapter 2: Introduction to DFT →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
