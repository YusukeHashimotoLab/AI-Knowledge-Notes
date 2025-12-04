---
title: "Chapter 2: Phase Diagrams and Phase Transformations of Alloys"
chapter_title: "Chapter 2: Phase Diagrams and Phase Transformations of Alloys"
subtitle: Phase Rule, Phase Diagrams, Solid Solutions, Intermetallic Compounds, Eutectoid and Peritectic Reactions
---

# Chapter 2: Phase Diagrams and Phase Transformations of Alloys

This chapter covers phase diagrams and phase transformations of alloys. You will learn essential concepts and techniques.

Phase Rule, Phase Diagrams, Solid Solutions, Intermetallic Compounds, Eutectoid and Peritectic Reactions

## 2.1 Fundamentals of Alloys and Phase Diagrams

An **alloy** is a metallic material composed of two or more elements. Compared to pure metals, alloys offer improved characteristics such as strength, corrosion resistance, and heat resistance. To understand alloy properties, **phase diagrams** are essential tools.

### 2.1.1 Phase Rule (Gibbs Phase Rule)

The **Gibbs Phase Rule** is the fundamental law that determines the degrees of freedom of a system at thermodynamic equilibrium.

**Gibbs Phase Rule:**

$$F = C - P + 2$$

where:

  * $F$: Degrees of freedom (number of independently variable parameters such as temperature, pressure, and composition)
  * $C$: Number of components (number of elements)
  * $P$: Number of phases

For a binary system ($C = 2$) at constant pressure, the degrees of freedom become:

$$F = C - P + 1 = 2 - P + 1 = 3 - P$$

**Examples:**

  * Single-phase region ($P = 1$): $F = 2$ - Both temperature and composition can be varied independently
  * Two-phase region ($P = 2$): $F = 1$ - Once temperature is fixed, the composition of each phase is uniquely determined
  * Three-phase coexistence point ($P = 3$): $F = 0$ - Both temperature and composition are fixed (invariant point)

    
    
    ```mermaid
    flowchart TD A[Gibbs Phase Rule: F = C - P + 2] -->B[Binary system at constant pressure] B -->C[F = 3 - P] C -->D[Single-phase region P=1] C -->E[Two-phase coexistence P=2] C -->F[Three-phase coexistence point P=3] D -->G[F = 2: Temperature and Composition variable] E -->H[F = 1: When temperature is fixed, composition is determined] F -->I[F = 0: Both temperature and composition are fixed]
    ```

### 2.1.2 Basic Elements of Phase Diagrams

A phase diagram shows the regions where different phases are stable as a function of temperature and composition.

  * **Phase boundary** : Lines separating regions of different stable phases
  * **Liquidus** : Above this line, the material is completely liquid
  * **Solidus** : Below this line, the material is completely solid
  * **Solid solution** : A state where two or more elements are dissolved in a single phase
  * **Eutectic point** : The point where liquid phase transforms into two solid phases simultaneously
  * **Eutectoid point** : The point where one solid phase transforms into two different solid phases

### 2.1.3 Types of Solid Solutions

Solid solutions are classified into two types based on how atoms are arranged.

  * **Substitutional solid solution** : Solute atoms substitute for solvent atoms at lattice sites
    * Conditions (Hume-Rothery Rules): Atomic radius difference < 15%, same crystal structure, similar electronegativity, similar valence
    * Examples: Cu-Ni, Au-Ag, Fe-Cr
  * **Interstitial solid solution** : Solute atoms occupy interstitial sites (spaces between lattice atoms)
    * Conditions: Small solute atoms (C, N, H, O, etc.)
    * Examples: Fe-C (Steel), Ti-O

#### Code Example 1: Basic Structure Visualization of Binary System Phase Diagram
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_simple_phase_diagram():
        """
        Binary system phase diagram (eutectic type) example.
        Shows an idealized eutectic phase diagram.
        Actual phase diagrams are obtained from experimental data
        and thermodynamic calculations.
        """
        # Composition range (mass% of element B)
        composition_B = np.linspace(0, 100, 300)
    
        # Liquidus (simplified)
        T_melt_A = 1000  # Melting point of element A (deg C)
        T_melt_B = 900   # Melting point of element B (deg C)
        T_eutectic = 700 # Eutectic temperature (deg C)
        C_eutectic = 60  # Eutectic composition (mass% B)
    
        # Liquidus calculation (simplified curve)
        liquidus = np.where(composition_B <= C_eutectic,
            T_melt_A - (T_melt_A - T_eutectic) * (composition_B / C_eutectic)**1.5,
            T_melt_B - (T_melt_B - T_eutectic) * ((100 - composition_B) / (100 - C_eutectic))**1.5)
    
        # Solidus (simplified)
        # Maximum solubility of alpha phase and beta phase boundaries
        C_alpha_max = 20  # Maximum solubility in alpha phase (mass% B)
        C_beta_min = 85   # Minimum solubility for beta phase (mass% B)
    
        solidus_alpha = T_eutectic * np.ones_like(composition_B)
        solidus_beta = T_eutectic * np.ones_like(composition_B)
    
        # Temperature dependence of solubility (simplified)
        for i, c in enumerate(composition_B):
            if c <= C_alpha_max:
                solidus_alpha[i] = T_eutectic + (T_melt_A - T_eutectic) * (1 - c / C_alpha_max)**2
            if c >= C_beta_min:
                solidus_beta[i] = T_eutectic + (T_melt_B - T_eutectic) * (1 - (100 - c) / (100 - C_beta_min))**2
    
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # Liquidus
        ax.plot(composition_B, liquidus, 'r-', linewidth=2.5, label='Liquidus')
    
        # Solidus
        ax.plot(composition_B[:len(composition_B)//3],
                solidus_alpha[:len(composition_B)//3],
                'b-', linewidth=2.5, label='Solidus (alpha phase)')
        ax.plot(composition_B[2*len(composition_B)//3:],
                solidus_beta[2*len(composition_B)//3:],
                'g-', linewidth=2.5, label='Solidus (beta phase)')
    
        # Eutectic temperature line
        ax.axhline(T_eutectic, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(50, T_eutectic - 30, f'Eutectic Temperature ({T_eutectic} deg C)',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
        # Eutectic point marker
        ax.plot(C_eutectic, T_eutectic, 'ko', markersize=12, label='Eutectic Point')
    
        # Region labels
        ax.text(10, 950, 'Liquid (L)', fontsize=13, fontweight='bold', color='red')
        ax.text(10, 800, 'alpha Solid Solution', fontsize=12, fontweight='bold', color='blue')
        ax.text(50, 650, 'alpha + beta', fontsize=12, fontweight='bold', color='purple')
        ax.text(90, 800, 'beta Solid Solution', fontsize=12, fontweight='bold', color='green')
        ax.text(50, 850, 'L + alpha', fontsize=11, fontweight='bold', color='orange', alpha=0.7)
        ax.text(75, 850, 'L + beta', fontsize=11, fontweight='bold', color='orange', alpha=0.7)
    
        ax.set_xlabel('Composition (mass% B)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Temperature (deg C)', fontsize=13, fontweight='bold')
        ax.set_title('Basic Structure of Binary System Phase Diagram (Eutectic Type)',
                     fontsize=15, fontweight='bold', pad=15)
        ax.set_xlim(0, 100)
        ax.set_ylim(600, 1100)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='upper right')
    
        plt.tight_layout()
        plt.savefig('simple_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        print("=" * 70)
        print("Basic Explanation of Binary System Phase Diagram")
        print("=" * 70)
        print("\n[Main Parameters]")
        print(f"  Melting point of element A: {T_melt_A} deg C")
        print(f"  Melting point of element B: {T_melt_B} deg C")
        print(f"  Eutectic temperature: {T_eutectic} deg C")
        print(f"  Eutectic composition: {C_eutectic} mass% B")
        print("\n[Phase Regions]")
        print("  L: Liquid phase")
        print("  alpha: Solid solution rich in element A")
        print("  beta: Solid solution rich in element B")
        print("  L + alpha: Two-phase coexistence of liquid and alpha")
        print("  L + beta: Two-phase coexistence of liquid and beta")
        print("  alpha + beta: Two-phase coexistence of alpha and beta")
        print("\n[Degrees of Freedom by Gibbs Phase Rule]")
        print("  Single-phase region (L, alpha, beta): F = 2 (Temperature and Composition)")
        print("  Two-phase region: F = 1 (When temperature is fixed, composition is determined)")
        print("  Eutectic point: F = 0 (Both temperature and composition are fixed)")
        print("=" * 70)
        print("\n Graph saved as 'simple_phase_diagram.png'")
    
    # Execute
    plot_simple_phase_diagram()

## 2.2 Fe-C Phase Diagram and Microstructure of Steel

The iron-carbon (Fe-C) phase diagram is one of the most important phase diagrams in materials science. The properties of steel are determined by carbon content and heat treatment, which can be explained using the Fe-C phase diagram.

### 2.2.1 Main Phases in the Fe-C Phase Diagram

  * **Ferrite (alpha-Fe)** : BCC structure, low carbon solubility (maximum 0.022 mass% at 727 deg C)
  * **Austenite (gamma-Fe)** : FCC structure, high carbon solubility (maximum 2.14 mass% at 1147 deg C)
  * **Cementite (Fe3C)** : Intermetallic compound, 6.67 mass% carbon, very hard and brittle
  * **Pearlite** : Lamellar structure of ferrite and cementite (eutectoid microstructure)
  * **Martensite** : Supersaturated solid solution formed by rapid cooling of austenite (non-equilibrium phase)

### 2.2.2 Eutectoid and Eutectic Reactions

**Eutectoid Reaction**

At 727 deg C and 0.76 mass% C (eutectoid point), the following reaction occurs:

$$\gamma \text{ (Austenite)} \rightarrow \alpha \text{ (Ferrite)} + \text{Fe}_3\text{C} \text{ (Cementite)}$$

The lamellar structure formed by this reaction is called **pearlite**.

**Eutectic Reaction**

At 1147 deg C and 4.3 mass% C (eutectic point), the following reaction occurs:

$$L \text{ (Liquid)} \rightarrow \gamma \text{ (Austenite)} + \text{Fe}_3\text{C} \text{ (Cementite)}$$

The microstructure formed by this reaction is called **ledeburite** (found in cast iron).

### 2.2.3 Classification of Steels

Iron-carbon alloys are classified as follows based on carbon content:

  * **Hypoeutectoid steel** : 0.022 - 0.76 mass% C - Proeutectoid ferrite + Pearlite
  * **Eutectoid steel** : 0.76 mass% C - Complete pearlite microstructure
  * **Hypereutectoid steel** : 0.76 - 2.14 mass% C - Proeutectoid cementite + Pearlite
  * **Cast iron** : 2.14 - 6.67 mass% C - Ledeburite microstructure

#### Code Example 2: Creating Fe-C Phase Diagram (Simplified Version)
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_fe_c_diagram():
        """
        Fe-C Phase Diagram (simplified version)
        The actual Fe-C phase diagram is more complex,
        but here we show the main features in a simplified form.
        """
        # Main points on the diagram
        # Temperature (deg C)
        T_alpha_max = 912      # alpha to gamma transformation temperature (pure iron)
        T_gamma_max = 1394     # gamma to delta transformation temperature (pure iron)
        T_eutectoid = 727      # Eutectoid temperature
        T_eutectic = 1147      # Eutectic temperature
    
        # Composition (mass% C)
        C_eutectoid = 0.76     # Eutectoid composition
        C_eutectic = 4.3       # Eutectic composition
        C_max_austenite = 2.14 # Maximum carbon solubility in gamma phase
        C_max_ferrite = 0.022  # Maximum carbon solubility in alpha phase
    
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
    
        # Eutectoid temperature line
        ax.axhline(T_eutectoid, color='purple', linestyle='--', linewidth=2,
                   alpha=0.7, label='Eutectoid Temperature (727 deg C)')
    
        # Eutectoid point
        ax.plot(C_eutectoid, T_eutectoid, 'ro', markersize=15,
                label=f'Eutectoid Point ({C_eutectoid}% C, {T_eutectoid} deg C)')
    
        # Eutectic point
        ax.plot(C_eutectic, T_eutectic, 'bs', markersize=15,
                label=f'Eutectic Point ({C_eutectic}% C, {T_eutectic} deg C)')
    
        # alpha/gamma boundary (simplified)
        C_alpha_gamma = np.linspace(0, C_max_ferrite, 50)
        T_alpha_gamma = T_alpha_max + (T_eutectoid - T_alpha_max) * (C_alpha_gamma / C_max_ferrite)**0.5
        ax.plot(C_alpha_gamma, T_alpha_gamma, 'b-', linewidth=2.5, label='alpha/gamma boundary')
    
        # gamma phase region boundary (gamma/gamma+Fe3C)
        C_gamma_right = np.linspace(C_eutectoid, C_max_austenite, 50)
        T_gamma_right = T_eutectoid + (T_eutectic - T_eutectoid) * (C_gamma_right - C_eutectoid) / (C_max_austenite - C_eutectoid)
        ax.plot(C_gamma_right, T_gamma_right, 'g-', linewidth=2.5, label='gamma/gamma+Fe3C boundary')
    
        # Liquidus (simplified)
        C_liquidus = np.linspace(0, C_eutectic, 100)
        T_liquidus = 1538 - 200 * C_liquidus  # Simplified linear approximation
        ax.plot(C_liquidus, T_liquidus, 'r-', linewidth=2.5, label='Liquidus')
    
        # Cementite line
        ax.axvline(6.67, color='black', linestyle='-', linewidth=2,
                   alpha=0.5, label='Cementite (Fe3C)')
    
        # Region labels
        ax.text(0.1, 1200, 'gamma\n(Austenite)', fontsize=14, fontweight='bold',
                ha='center', color='darkgreen')
        ax.text(0.3, 600, 'alpha + Pearlite', fontsize=12, fontweight='bold',
                ha='center', color='blue')
        ax.text(1.5, 600, 'Pearlite\n+ Fe3C', fontsize=12, fontweight='bold',
                ha='center', color='purple')
        ax.text(C_eutectoid, 820, 'Pearlite', fontsize=11, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
        # Steel classification regions
        ax.axvspan(0, C_eutectoid, alpha=0.15, color='blue', label='Hypoeutectoid Steel')
        ax.axvspan(C_eutectoid, C_max_austenite, alpha=0.15, color='red', label='Hypereutectoid Steel')
        ax.axvspan(C_max_austenite, 6.67, alpha=0.15, color='gray', label='Cast Iron')
    
        ax.set_xlabel('Carbon Content (mass% C)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Temperature (deg C)', fontsize=14, fontweight='bold')
        ax.set_title('Fe-C Phase Diagram (Simplified Version)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 7)
        ax.set_ylim(500, 1600)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right')
    
        # Important composition lines
        ax.axvline(C_eutectoid, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axvline(C_max_austenite, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    
        plt.tight_layout()
        plt.savefig('fe_c_phase_diagram_simplified.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Information output
        print("=" * 80)
        print("Main Points of Fe-C Phase Diagram")
        print("=" * 80)
        print(f"\nEutectoid Point: {C_eutectoid} mass% C, {T_eutectoid} deg C")
        print(f"  Reaction: gamma -> alpha + Fe3C (Pearlite formation)")
        print(f"\nEutectic Point: {C_eutectic} mass% C, {T_eutectic} deg C")
        print(f"  Reaction: L -> gamma + Fe3C (Ledeburite formation)")
        print(f"\nMaximum carbon solubility in alpha phase: {C_max_ferrite} mass% C ({T_eutectoid} deg C)")
        print(f"Maximum carbon solubility in gamma phase: {C_max_austenite} mass% C ({T_eutectic} deg C)")
        print("\n[Classification of Steels]")
        print(f"  Hypoeutectoid steel: 0 - {C_eutectoid} mass% C")
        print(f"  Eutectoid steel: {C_eutectoid} mass% C")
        print(f"  Hypereutectoid steel: {C_eutectoid} - {C_max_austenite} mass% C")
        print(f"  Cast iron: {C_max_austenite} - 6.67 mass% C")
        print("=" * 80)
        print("\n Graph saved as 'fe_c_phase_diagram_simplified.png'")
    
    # Execute
    plot_fe_c_diagram()

#### Code Example 3: Phase Fraction Calculation Using the Lever Rule
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def lever_rule_calculation(C_total, C_alpha, C_beta, T):
        """
        Lever Rule calculation for mass fraction of each phase
    
        Parameters:
        -----------
        C_total: float
            Overall composition (mass% B)
        C_alpha: float
            Composition of alpha phase (mass% B)
        C_beta: float
            Composition of beta phase (mass% B)
        T: float
            Temperature (deg C)
    
        Returns:
        --------
        f_alpha: float
            Mass fraction of alpha phase
        f_beta: float
            Mass fraction of beta phase
        """
        # Lever Rule
        # f_alpha = (C_beta - C_total) / (C_beta - C_alpha)
        # f_beta = (C_total - C_alpha) / (C_beta - C_alpha)
    
        if C_alpha == C_beta:
            # Single-phase region case
            if abs(C_total - C_alpha) < 1e-6:
                return 1.0, 0.0
            else:
                raise ValueError("Composition is outside phase boundaries")
    
        f_alpha = (C_beta - C_total) / (C_beta - C_alpha)
        f_beta = (C_total - C_alpha) / (C_beta - C_alpha)
    
        # Fractions must be in range 0-1
        f_alpha = np.clip(f_alpha, 0, 1)
        f_beta = np.clip(f_beta, 0, 1)
    
        return f_alpha, f_beta
    
    # Example Problem: Cooling Fe-0.4% C steel from 800 deg C to room temperature
    print("=" * 80)
    print("Phase Fraction Calculation Using the Lever Rule - Example Problem")
    print("=" * 80)
    print("\n[Problem Statement]")
    print("When Fe-0.4 mass% C steel (hypoeutectoid steel) is cooled to just below 727 deg C,")
    print("calculate the mass fractions of ferrite and pearlite.")
    
    # Given values
    C_steel = 0.4       # Carbon content of steel (mass% C)
    C_ferrite = 0.022   # Carbon content of ferrite (mass% C)
    C_eutectoid = 0.76  # Eutectoid composition (pearlite) (mass% C)
    T = 727             # Temperature (deg C)
    
    # Lever Rule calculation
    f_ferrite, f_pearlite = lever_rule_calculation(C_steel, C_ferrite, C_eutectoid, T)
    
    print(f"\n[Given Values]")
    print(f"  Steel composition: {C_steel} mass% C")
    print(f"  Ferrite composition: {C_ferrite} mass% C")
    print(f"  Pearlite composition: {C_eutectoid} mass% C (eutectoid composition)")
    print(f"  Temperature: {T} deg C (just below eutectoid temperature)")
    print(f"\n[Application of Lever Rule]")
    print(f"  f_Ferrite = (C_Pearlite - C_Steel) / (C_Pearlite - C_Ferrite)")
    print(f"           = ({C_eutectoid} - {C_steel}) / ({C_eutectoid} - {C_ferrite})")
    print(f"           = {f_ferrite:.4f} = {f_ferrite * 100:.2f}%")
    print(f"  ")
    print(f"  f_Pearlite = (C_Steel - C_Ferrite) / (C_Pearlite - C_Ferrite)")
    print(f"            = ({C_steel} - {C_ferrite}) / ({C_eutectoid} - {C_ferrite})")
    print(f"            = {f_pearlite:.4f} = {f_pearlite * 100:.2f}%")
    print(f"\n[Verification]")
    print(f"  f_Ferrite + f_Pearlite = {f_ferrite + f_pearlite:.4f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Composition markers
    compositions = [C_ferrite, C_steel, C_eutectoid]
    labels = ['Ferrite\n(alpha)', 'Steel Composition\n(0.4% C)', 'Pearlite\n(Eutectoid Composition)']
    colors = ['blue', 'red', 'purple']
    
    for c, label, color in zip(compositions, labels, colors):
        ax.axvline(c, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.text(c, 0.9, label, ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Lever Rule visualization
    ax.plot([C_ferrite, C_eutectoid], [0.5, 0.5], 'k-', linewidth=3, label='Tie line')
    ax.plot(C_steel, 0.5, 'ro', markersize=15, label='Fulcrum point (Steel composition)')
    
    # Distance annotations
    ax.annotate('', xy=(C_ferrite, 0.35), xytext=(C_steel, 0.35),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text((C_ferrite + C_steel) / 2, 0.3, f'{C_steel - C_ferrite:.3f}%',
            ha='center', fontsize=10, color='blue', fontweight='bold')
    
    ax.annotate('', xy=(C_steel, 0.35), xytext=(C_eutectoid, 0.35),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text((C_steel + C_eutectoid) / 2, 0.3, f'{C_eutectoid - C_steel:.3f}%',
            ha='center', fontsize=10, color='purple', fontweight='bold')
    
    # Mass fraction display
    ax.text(C_ferrite - 0.1, 0.7, f'Mass Fraction\n{f_ferrite * 100:.1f}%',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(C_eutectoid + 0.1, 0.7, f'Mass Fraction\n{f_pearlite * 100:.1f}%',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.set_xlabel('Carbon Content (mass% C)', fontsize=13, fontweight='bold')
    ax.set_ylabel('', fontsize=13)
    ax.set_title('Phase Fraction Calculation Using Lever Rule (Fe-0.4% C Steel)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lever_rule_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("[Conclusion]")
    print("=" * 80)
    print(f"When Fe-{C_steel} mass% C steel is cooled to just below the eutectoid temperature,")
    print(f"the microstructure consists of:")
    print(f"  Ferrite: {f_ferrite * 100:.1f} mass%")
    print(f"  Pearlite: {f_pearlite * 100:.1f} mass%")
    print("\nThis is a typical hypoeutectoid steel microstructure.")
    print("Proeutectoid ferrite (soft) and pearlite (lamellar structure)")
    print("form the two-phase structure. As carbon content increases toward 0.76%,")
    print("the pearlite fraction increases, resulting in higher strength and hardness.")
    print("=" * 80)
    print("\n Graph saved as 'lever_rule_example.png'")

## 2.3 Thermodynamics of Phase Transformations

A **phase transformation** is the process by which a material changes from one phase to another due to changes in temperature or composition. To understand phase transformations, the concept of Gibbs free energy is essential.

### 2.3.1 Gibbs Free Energy

**Definition of Gibbs Free Energy:**

$$G = H - TS$$

where: $G$ = Gibbs free energy, $H$ = enthalpy, $T$ = temperature, $S$ = entropy

At equilibrium, the Gibbs free energy of the system is minimized. When temperature changes, the phase with the lowest Gibbs free energy becomes the stable phase, and phase transformation occurs.

### 2.3.2 Driving Force for Phase Transformation

The driving force for a phase transformation is the difference in Gibbs free energy between two phases.

$$\Delta G = G_{\beta} - G_{\alpha}$$

When $\Delta G < 0$, the transformation from alpha phase to beta phase is thermodynamically favorable.

#### Code Example 4: Visualization of Gibbs Free Energy and Phase Stability
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def gibbs_energy_phases(T):
        """
        Calculate Gibbs free energy of two phases (alpha and beta) as a function of temperature.
        Simplified model: G_alpha = H_alpha - T * S_alpha
                          G_beta = H_beta - T * S_beta
        Here, beta phase has higher entropy than alpha phase.
        """
        # Parameters (arbitrary units)
        H_alpha = 1000  # Enthalpy of alpha phase
        H_beta = 1200   # Enthalpy of beta phase (higher than alpha)
        S_alpha = 5     # Entropy of alpha phase
        S_beta = 8      # Entropy of beta phase (higher than alpha)
    
        G_alpha = H_alpha - T * S_alpha
        G_beta = H_beta - T * S_beta
    
        return G_alpha, G_beta
    
    # Temperature range
    temperatures = np.linspace(0, 500, 1000)
    
    # Calculate Gibbs free energy at each temperature
    G_alpha_values = []
    G_beta_values = []
    
    for T in temperatures:
        G_alpha, G_beta = gibbs_energy_phases(T)
        G_alpha_values.append(G_alpha)
        G_beta_values.append(G_beta)
    
    G_alpha_values = np.array(G_alpha_values)
    G_beta_values = np.array(G_beta_values)
    
    # Equilibrium temperature (where G_alpha = G_beta)
    # H_alpha - T_eq * S_alpha = H_beta - T_eq * S_beta
    # T_eq = (H_beta - H_alpha) / (S_beta - S_alpha)
    H_alpha = 1000
    H_beta = 1200
    S_alpha = 5
    S_beta = 8
    T_equilibrium = (H_beta - H_alpha) / (S_beta - S_alpha)
    
    print("=" * 70)
    print("Gibbs Free Energy and Phase Transformation")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  alpha phase: H_alpha = {H_alpha}, S_alpha = {S_alpha}")
    print(f"  beta phase: H_beta = {H_beta}, S_beta = {S_beta}")
    print(f"\nEquilibrium Temperature T_eq = (H_beta - H_alpha) / (S_beta - S_alpha)")
    print(f"                           = ({H_beta} - {H_alpha}) / ({S_beta} - {S_alpha})")
    print(f"                           = {T_equilibrium:.2f}")
    print(f"\nPhase Stability:")
    print(f"  T < {T_equilibrium:.2f}: G_alpha < G_beta -> alpha phase stable")
    print(f"  T = {T_equilibrium:.2f}: G_alpha = G_beta -> equilibrium")
    print(f"  T > {T_equilibrium:.2f}: G_alpha > G_beta -> beta phase stable")
    print("=" * 70)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left figure: Temperature dependence of Gibbs free energy
    ax1.plot(temperatures, G_alpha_values, 'b-', linewidth=2.5, label='alpha phase')
    ax1.plot(temperatures, G_beta_values, 'r-', linewidth=2.5, label='beta phase')
    ax1.axvline(T_equilibrium, color='green', linestyle='--', linewidth=2,
                label=f'Equilibrium Temperature ({T_equilibrium:.1f})')
    
    # Stable phase regions
    ax1.fill_between(temperatures, G_alpha_values.min(), G_alpha_values.max(),
                     where=(temperatures < T_equilibrium), alpha=0.2, color='blue',
                     label='alpha phase stable region')
    ax1.fill_between(temperatures, G_alpha_values.min(), G_alpha_values.max(),
                     where=(temperatures >= T_equilibrium), alpha=0.2, color='red',
                     label='beta phase stable region')
    
    # Equilibrium point
    ax1.plot(T_equilibrium, gibbs_energy_phases(T_equilibrium)[0], 'go', markersize=12)
    
    ax1.set_xlabel('Temperature (K)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Gibbs Free Energy G (arb. units)', fontsize=13, fontweight='bold')
    ax1.set_title('Temperature Dependence of Gibbs Free Energy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right figure: Gibbs free energy difference Delta G = G_beta - G_alpha
    delta_G = G_beta_values - G_alpha_values
    
    ax2.plot(temperatures, delta_G, 'purple', linewidth=2.5, label='Delta G = G_beta - G_alpha')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(T_equilibrium, color='green', linestyle='--', linewidth=2,
                label=f'Equilibrium Temperature ({T_equilibrium:.1f})')
    
    # Delta G < 0 and Delta G > 0 regions
    ax2.fill_between(temperatures, 0, delta_G,
                     where=(delta_G < 0), alpha=0.3, color='blue',
                     label='Delta G < 0 (alpha->beta unfavorable)')
    ax2.fill_between(temperatures, 0, delta_G,
                     where=(delta_G >= 0), alpha=0.3, color='red',
                     label='Delta G > 0 (alpha->beta unfavorable)')
    
    ax2.set_xlabel('Temperature (K)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Delta G = G_beta - G_alpha (arb. units)', fontsize=13, fontweight='bold')
    ax2.set_title('Driving Force for Phase Transformation (Gibbs Free Energy Difference)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gibbs_energy_phase_transformation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n Graph saved as 'gibbs_energy_phase_transformation.png'")
    print("\n[Physical Interpretation]")
    print("-" * 70)
    print("The beta phase has higher enthalpy (H_beta > H_alpha),")
    print("so it is energetically unfavorable at low temperatures.")
    print("")
    print("However, the beta phase has higher entropy (S_beta > S_alpha),")
    print("so at high temperatures the -TS contribution causes its Gibbs free energy")
    print("to decrease, making the beta phase stable.")
    print("")
    print("This explains phase transformations such as melting (solid to liquid)")
    print("and BCC to FCC transitions in iron.")
    print("-" * 70)

## 2.4 Phase Diagram Calculation Using pycalphad

For actual materials, the **CALPHAD (CALculation of PHAse Diagrams)** method is used to calculate phase diagrams from thermodynamic databases. In Python, the `pycalphad` library provides this functionality.

#### What is pycalphad?

`pycalphad` is a Python library for calculating multicomponent phase diagrams, phase equilibria, and thermodynamic properties. It provides similar functionality to commercial software (such as Thermo-Calc, FactSage) as an open-source tool.

#### Code Example 5: Al-Cu System Phase Diagram Calculation Using pycalphad
    
    
    """
    Phase Diagram Calculation for Al-Cu System Using pycalphad
    
    Note: pycalphad installation is required
    pip install pycalphad
    
    A thermodynamic database (TDB file) is also required.
    Here we show a simplified demonstration.
    """
    
    try:
        from pycalphad import Database, equilibrium, variables as v
        from pycalphad.plot.binary import binplot
        import matplotlib.pyplot as plt
        import numpy as np
    
        print("=" * 80)
        print("Al-Cu System Phase Diagram Calculation Demo Using pycalphad")
        print("=" * 80)
    
        # Simplified thermodynamic database for demonstration (TDB format)
        # For actual calculations, detailed TDB files should be used
        tdb_string = """
        ELEMENT AL FCC_A1 26.98 4540.0 28.3!
        ELEMENT CU FCC_A1 63.546 5004.0 33.15!
    
        PHASE FCC_A1 % 2 1 1!
        CONSTITUENT FCC_A1:AL,CU: VA:!
    
        PHASE LIQUID % 1 1.0!
        CONSTITUENT LIQUID:AL,CU:!
    
        FUNCTION GHSERAL 298.15 -7976.15+137.093038*T-24.3671976*T*LN(T)
            -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1); 700.00 Y
            -11276.24+223.048446*T-38.5844296*T*LN(T)
            +18.531982E-3*T**2-5.764227E-6*T**3+74092*T**(-1); 933.47 Y
            -11278.378+188.684153*T-31.748192*T*LN(T)
            -1.234264E28*T**(-9); 2900.00 N!
    
        FUNCTION GHSERCU 298.15 -7770.458+130.485235*T-24.112392*T*LN(T)
            -2.65684E-3*T**2+0.129223E-6*T**3+52478*T**(-1); 1357.77 Y
            -13542.026+183.803828*T-31.38*T*LN(T)
            +3.64167E29*T**(-9); 3200.00 N!
    
        PARAMETER G(FCC_A1,AL:VA;0) 298.15 +GHSERAL; 6000 N!
        PARAMETER G(FCC_A1,CU:VA;0) 298.15 +GHSERCU; 6000 N!
        PARAMETER G(FCC_A1,AL,CU:VA;0) 298.15 -53520+7.2*T; 6000 N!
    
        PARAMETER G(LIQUID,AL;0) 298.15 +11005.553-11.840873*T+GHSERAL+7.934E-20*T**7; 933.47 Y
            +10481.974-11.252014*T+GHSERAL+1.234264E28*T**(-9); 6000 N!
        PARAMETER G(LIQUID,CU;0) 298.15 +12964.735-9.511904*T+GHSERCU-5.8489E-21*T**7; 1357.77 Y
            +13495.481-9.922344*T+GHSERCU-3.64167E29*T**(-9); 3200 N!
        PARAMETER G(LIQUID,AL,CU;0) 298.15 -66220+7.5*T; 6000 N!
        """
    
        # Create database
        db = Database(tdb_string)
    
        # Calculation conditions
        components = ['AL', 'CU', 'VA']
        phases = ['FCC_A1', 'LIQUID']
    
        # Temperature range
        T_range = (500, 1500, 10)  # 500-1500 K, 10 K step
    
        # Composition range (mole fraction of Cu)
        X_range = (0, 1, 0.01)  # 0-1, 0.01 step
    
        print("\nCalculation Conditions:")
        print(f"  Components: {components}")
        print(f"  Phases: {phases}")
        print(f"  Temperature range: {T_range[0]}-{T_range[1]} K")
        print(f"  Composition range: X(CU) = {X_range[0]}-{X_range[1]}")
        print("\nStarting calculation (may take some time)...")
    
        # Equilibrium calculation
        # Note: Actual calculation may take time
        eq_result = equilibrium(db, components, phases,
                                {v.X('CU'): X_range, v.T: T_range, v.P: 101325})
    
        print("Calculation complete")
    
        # Phase diagram plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
    
        # Use pycalphad's built-in plotting function
        binplot(db, components, phases,
                {v.X('CU'): X_range, v.T: T_range, v.P: 101325},
                eq_result=eq_result, ax=ax)
    
        ax.set_xlabel('Composition X(Cu)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Temperature (K)', fontsize=13, fontweight='bold')
        ax.set_title('Al-Cu System Phase Diagram (Calculated with pycalphad)',
                     fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('al_cu_phase_diagram_pycalphad.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        print("\n Graph saved as 'al_cu_phase_diagram_pycalphad.png'")
    
        print("\n" + "=" * 80)
        print("[About the Al-Cu System]")
        print("=" * 80)
        print("The Al-Cu system is the basis for aluminum alloys (such as duralumin).")
        print("Main features:")
        print("  - FCC solid solution (alpha-Al) forms on the Al-rich side")
        print("  - FCC solid solution (alpha-Cu) forms on the Cu-rich side")
        print("  - Multiple intermetallic compounds form at intermediate compositions")
        print("  - Age hardening treatment enables high strength (precipitation hardening)")
        print("=" * 80)
    
    except ImportError:
        print("Error: pycalphad is not installed")
        print("Installation: pip install pycalphad")
        print("\npycalphad provides the following functionality:")
        print("  - Multicomponent phase diagram calculation")
        print("  - Phase equilibrium calculation")
        print("  - Thermodynamic property calculation (enthalpy, entropy, etc.)")
        print("  - Precipitation phase stability evaluation")
        print("\nAlternatively, simplified phase diagrams can be created (see Code Example 2)")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        print("pycalphad calculations require thermodynamic databases (TDB)")
        print("Detailed TDB files can be obtained from NIST, GTT-Technologies, Thermo-Calc, etc.")

### Exercises

#### Exercise 2.1

Easy

**Problem:** For a binary system phase diagram at constant pressure, calculate the degrees of freedom $F$ for each of the following cases.

  1. Single-phase region (liquid phase only)
  2. Two-phase region (liquid phase + alpha solid phase)
  3. Eutectic point (liquid phase + alpha solid phase + beta solid phase)

Show Solution
    
    
    print("=" * 70)
    print("Degrees of Freedom Calculation Using Gibbs Phase Rule")
    print("=" * 70)
    
    # Gibbs Phase Rule: F = C - P + 2
    # For binary system at constant pressure: F = C - P + 1 = 2 - P + 1 = 3 - P
    
    C = 2  # Number of components
    pressure_fixed = True  # Constant pressure
    
    print(f"\nGibbs Phase Rule: F = C - P + 2")
    print(f"Number of components C = {C}")
    print(f"At constant pressure: F = C - P + 1 = 3 - P")
    
    cases = [("Single-phase region (liquid phase only)", 1),
             ("Two-phase region (liquid + alpha solid)", 2),
             ("Eutectic point (liquid + alpha + beta)", 3)]
    
    print("\n" + "=" * 70)
    print("[Calculation Results]")
    print("=" * 70)
    
    for i, (description, P) in enumerate(cases, 1):
        F = 3 - P
        print(f"\n({i}) {description}")
        print(f"    Number of phases P = {P}")
        print(f"    Degrees of freedom F = 3 - {P} = {F}")
    
        if F == 2:
            print(f"    -> Both temperature and composition can be varied independently")
        elif F == 1:
            print(f"    -> When temperature is fixed, composition of each phase is determined")
            print(f"       (or when composition is fixed, temperature is determined)")
        elif F == 0:
            print(f"    -> Both temperature and composition are fixed (invariant point)")
    
    print("\n" + "=" * 70)
    print("[Physical Interpretation]")
    print("=" * 70)
    print("\nThe degrees of freedom F represents the number of independent variables")
    print("needed to specify the state of the system.")
    print("")
    print("F = 2: In single-phase regions, both temperature and composition can be varied.")
    print("       Represented as an area (2D region) on the phase diagram.")
    print("")
    print("F = 1: In two-phase regions, when temperature is fixed, the composition")
    print("       of each phase is thermodynamically determined (lever rule gives fractions).")
    print("       Represented as a line (1D) on the phase diagram.")
    print("")
    print("F = 0: At three-phase coexistence points, both temperature and composition are fixed.")
    print("       Represented as a point (0D) on the phase diagram.")
    print("       Examples: Eutectic point, eutectoid point, peritectic point")
    print("=" * 70)
    
    # Figure
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Simplified phase diagram
    composition = np.linspace(0, 100, 300)
    T_liquidus_1 = 1000 - 5 * composition[:150]
    T_liquidus_2 = 250 + 5 * (composition[150:] - 50)
    liquidus = np.concatenate([T_liquidus_1, T_liquidus_2])
    
    ax.plot(composition, liquidus, 'r-', linewidth=3, label='Liquidus')
    ax.axhline(250, color='blue', linestyle='--', linewidth=2, label='Eutectic Temperature')
    
    # Region labels
    ax.text(50, 800, 'F = 2\n(Single-phase region)', fontsize=14, fontweight='bold',
            ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(25, 400, 'F = 1\n(Two-phase coexistence)', fontsize=13, fontweight='bold',
            ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(75, 400, 'F = 1\n(Two-phase coexistence)', fontsize=13, fontweight='bold',
            ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.plot(50, 250, 'ko', markersize=15, label='Eutectic Point (F=0)')
    
    ax.set_xlabel('Composition (mass% B)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (deg C)', fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Degrees of Freedom and Phase Diagram',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(200, 1100)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gibbs_phase_rule_degrees_of_freedom.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n Graph saved as 'gibbs_phase_rule_degrees_of_freedom.png'")

#### Exercise 2.2

Medium

**Problem:** Fe-0.6 mass% C steel is slowly cooled from 900 deg C to room temperature. Calculate the microstructure (mass fractions of proeutectoid ferrite and pearlite) just below 727 deg C using the lever rule.

Given values:

  * Carbon content of steel: 0.6 mass% C
  * Carbon content of ferrite (at 727 deg C): 0.022 mass% C
  * Eutectoid composition: 0.76 mass% C

Show Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Phase fraction calculation for Fe-0.6% C steel
    
    Purpose: Demonstrate lever rule application
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: numpy, matplotlib
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Given values
    C_steel = 0.6       # Carbon content of steel (mass% C)
    C_ferrite = 0.022   # Carbon content of ferrite (mass% C)
    C_pearlite = 0.76   # Eutectoid composition (pearlite) (mass% C)
    T = 727             # Temperature (deg C)
    
    print("=" * 80)
    print("Microstructure Calculation for Fe-0.6 mass% C Steel (Lever Rule)")
    print("=" * 80)
    print("\n[Problem Statement]")
    print(f"Steel composition: {C_steel} mass% C (Hypoeutectoid Steel)")
    print(f"Cooling temperature: 900 deg C -> room temperature")
    print(f"Evaluation temperature: {T} deg C (just below eutectoid temperature)")
    
    print("\n[Given Values]")
    print(f"  C_Steel: {C_steel} mass% C")
    print(f"  C_Ferrite: {C_ferrite} mass% C")
    print(f"  C_Pearlite: {C_pearlite} mass% C (eutectoid composition)")
    
    # Lever Rule application
    # f_alpha = (C_Pearlite - C_Steel) / (C_Pearlite - C_Ferrite)
    # f_Pearlite = (C_Steel - C_Ferrite) / (C_Pearlite - C_Ferrite)
    
    f_ferrite = (C_pearlite - C_steel) / (C_pearlite - C_ferrite)
    f_pearlite = (C_steel - C_ferrite) / (C_pearlite - C_ferrite)
    
    print("\n[Lever Rule Calculation]")
    print("\nFerrite mass fraction:")
    print(f"  f_Ferrite = (C_Pearlite - C_Steel) / (C_Pearlite - C_Ferrite)")
    print(f"           = ({C_pearlite} - {C_steel}) / ({C_pearlite} - {C_ferrite})")
    print(f"           = {C_pearlite - C_steel:.2f} / {C_pearlite - C_ferrite:.3f}")
    print(f"           = {f_ferrite:.4f}")
    print(f"           = {f_ferrite * 100:.2f}%")
    
    print("\nPearlite mass fraction:")
    print(f"  f_Pearlite = (C_Steel - C_Ferrite) / (C_Pearlite - C_Ferrite)")
    print(f"            = ({C_steel} - {C_ferrite}) / ({C_pearlite} - {C_ferrite})")
    print(f"            = {C_steel - C_ferrite:.3f} / {C_pearlite - C_ferrite:.3f}")
    print(f"            = {f_pearlite:.4f}")
    print(f"            = {f_pearlite * 100:.2f}%")
    
    print("\n[Verification]")
    total = f_ferrite + f_pearlite
    print(f"  f_Ferrite + f_Pearlite = {f_ferrite:.4f} + {f_pearlite:.4f}")
    print(f"                        = {total:.4f}")
    if abs(total - 1.0) < 1e-6:
        print("  Confirmed to equal 1.0")
    else:
        print(f"  Error: {total - 1.0:.6f}")
    
    # Detailed analysis of pearlite microstructure
    # Pearlite = Ferrite (0.022% C) + Cementite (6.67% C)
    # Pearlite composition = 0.76% C
    C_cementite = 6.67  # Carbon content of cementite
    
    f_ferrite_in_pearlite = (C_cementite - C_pearlite) / (C_cementite - C_ferrite)
    f_cementite_in_pearlite = (C_pearlite - C_ferrite) / (C_cementite - C_ferrite)
    
    print("\n[Detailed Analysis of Pearlite Microstructure]")
    print("Pearlite is a lamellar structure of ferrite and cementite.")
    print(f"\nFerrite fraction in pearlite:")
    print(f"  = ({C_cementite} - {C_pearlite}) / ({C_cementite} - {C_ferrite})")
    print(f"  = {f_ferrite_in_pearlite:.4f} = {f_ferrite_in_pearlite * 100:.2f}%")
    print(f"\nCementite fraction in pearlite:")
    print(f"  = ({C_pearlite} - {C_ferrite}) / ({C_cementite} - {C_ferrite})")
    print(f"  = {f_cementite_in_pearlite:.4f} = {f_cementite_in_pearlite * 100:.2f}%")
    
    # Total phase fractions in steel
    f_ferrite_total = f_ferrite + f_pearlite * f_ferrite_in_pearlite
    f_cementite_total = f_pearlite * f_cementite_in_pearlite
    
    print("\n[Total Phase Mass Fractions in Steel]")
    print(f"  Proeutectoid ferrite: {f_ferrite * 100:.2f}%")
    print(f"  Pearlite: {f_pearlite * 100:.2f}%")
    print(f"    - Ferrite in pearlite: {f_pearlite * f_ferrite_in_pearlite * 100:.2f}%")
    print(f"    - Cementite: {f_cementite_total * 100:.2f}%")
    print(f"  ----------------------------------------")
    print(f"  Total ferrite: {f_ferrite_total * 100:.2f}%")
    print(f"  Total cementite: {f_cementite_total * 100:.2f}%")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left figure: Lever Rule diagram
    compositions = [C_ferrite, C_steel, C_pearlite]
    labels = ['Ferrite\n(alpha)', 'Steel Composition\n(0.6% C)', 'Pearlite\n(Eutectoid)']
    colors = ['blue', 'red', 'purple']
    
    for c, label, color in zip(compositions, labels, colors):
        ax1.axvline(c, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(c, 0.9, label, ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax1.plot([C_ferrite, C_pearlite], [0.5, 0.5], 'k-', linewidth=3)
    ax1.plot(C_steel, 0.5, 'ro', markersize=15, label='Fulcrum point')
    
    ax1.annotate('', xy=(C_ferrite, 0.35), xytext=(C_steel, 0.35),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2.5))
    ax1.text((C_ferrite + C_steel) / 2, 0.28, f'{C_steel - C_ferrite:.3f}%',
             ha='center', fontsize=11, color='blue', fontweight='bold')
    
    ax1.annotate('', xy=(C_steel, 0.35), xytext=(C_pearlite, 0.35),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2.5))
    ax1.text((C_steel + C_pearlite) / 2, 0.28, f'{C_pearlite - C_steel:.2f}%',
             ha='center', fontsize=11, color='purple', fontweight='bold')
    
    ax1.text(C_ferrite - 0.1, 0.7, f'Mass Fraction\n{f_ferrite * 100:.1f}%',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    ax1.text(C_pearlite + 0.1, 0.7, f'Mass Fraction\n{f_pearlite * 100:.1f}%',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
    
    ax1.set_xlabel('Carbon Content (mass% C)', fontsize=13, fontweight='bold')
    ax1.set_title('Lever Rule Calculation', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.grid(axis='x', alpha=0.3)
    
    # Right figure: Microstructure pie chart
    phases = ['Proeutectoid Ferrite', f'Pearlite\n({f_pearlite * 100:.1f}%)']
    sizes = [f_ferrite * 100, f_pearlite * 100]
    colors_pie = ['lightblue', 'lightcoral']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=phases,
                                        colors=colors_pie, autopct='%1.1f%%',
                                        startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax2.set_title('Fe-0.6% C Steel Microstructure Composition',
                  fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('fe_06c_microstructure_calculation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("[Conclusion]")
    print("=" * 80)
    print(f"When Fe-{C_steel} mass% C steel is slowly cooled to just below the eutectoid temperature:")
    print(f"  Proeutectoid ferrite: {f_ferrite * 100:.1f} mass%")
    print(f"  Pearlite: {f_pearlite * 100:.1f} mass%")
    print("")
    print("This is a typical hypoeutectoid steel microstructure.")
    print("Proeutectoid ferrite is soft, while pearlite (lamellar structure) is harder,")
    print(f"so this steel has moderate strength and hardness.")
    print("")
    print(f"As carbon content approaches 0.76% (eutectoid steel), the pearlite fraction increases,")
    print("resulting in higher strength and hardness.")
    print("=" * 80)
    print("\n Graph saved as 'fe_06c_microstructure_calculation.png'")

#### Exercise 2.3

Hard

**Problem:** In the Fe-C phase diagram, explain the microstructure changes when Fe-1.2 mass% C steel (hypereutectoid steel) is slowly cooled from 1000 deg C to room temperature at each temperature range. Also, calculate the microstructure (mass fractions of proeutectoid cementite and pearlite) just below 727 deg C using the lever rule.

Show Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Phase transformation of Fe-1.2% C hypereutectoid steel
    
    Purpose: Demonstrate cooling transformation and lever rule
    Target: Intermediate to Advanced
    Execution time: 10-30 seconds
    Dependencies: numpy, matplotlib
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Given values
    C_steel = 1.2       # Carbon content of steel (mass% C) - Hypereutectoid steel
    C_eutectoid = 0.76  # Eutectoid composition (mass% C)
    C_cementite = 6.67  # Carbon content of cementite (mass% C)
    T_eutectoid = 727   # Eutectoid temperature (deg C)
    
    print("=" * 80)
    print("Cooling Transformation of Fe-1.2 mass% C Steel (Hypereutectoid Steel)")
    print("=" * 80)
    
    print("\n[Steel Classification]")
    print(f"Carbon content: {C_steel} mass% C")
    print(f"Classification: Hypereutectoid steel ({C_eutectoid} mass% C < C < 2.14 mass% C)")
    
    # Cooling process explanation
    cooling_stages = [
        ("1000 deg C", "gamma phase (austenite) single phase"),
        ("~850 deg C", "gamma + Fe3C boundary line; proeutectoid cementite starts to precipitate"),
        ("850-727 deg C", "gamma + Fe3C two-phase coexistence; proeutectoid Fe3C precipitation continues"),
        ("727 deg C (eutectoid temperature)", "Eutectoid reaction: gamma -> alpha + Fe3C (pearlite forms)"),
        ("Below 727 deg C", "Proeutectoid Fe3C + pearlite; room temperature microstructure")
    ]
    
    print("\n[Microstructure Changes During Cooling]")
    print("=" * 80)
    for i, (temp, description) in enumerate(cooling_stages, 1):
        print(f"\n{i}. {temp}")
        print(f"   {description}")
    
    # Lever Rule calculation (just below 727 deg C)
    # For hypereutectoid steel: Proeutectoid cementite + Pearlite
    # f_Fe3C = (C_Steel - C_Eutectoid) / (C_Cementite - C_Eutectoid)
    # f_Pearlite = (C_Cementite - C_Steel) / (C_Cementite - C_Eutectoid)
    
    f_cementite = (C_steel - C_eutectoid) / (C_cementite - C_eutectoid)
    f_pearlite = (C_cementite - C_steel) / (C_cementite - C_eutectoid)
    
    print("\n" + "=" * 80)
    print("[Lever Rule Calculation (just below 727 deg C)]")
    print("=" * 80)
    
    print(f"\nOverall composition: {C_steel} mass% C")
    print(f"Pearlite composition (gamma phase at eutectoid): {C_eutectoid} mass% C")
    print(f"Cementite composition (Fe3C): {C_cementite} mass% C")
    
    print("\n[Phase Fraction Calculation]")
    print("\nProeutectoid cementite:")
    print(f"  f_Fe3C = (C_Steel - C_Eutectoid) / (C_Cementite - C_Eutectoid)")
    print(f"        = ({C_steel} - {C_eutectoid}) / ({C_cementite} - {C_eutectoid})")
    print(f"        = {f_cementite:.4f} = {f_cementite * 100:.2f}%")
    
    print("\nPearlite:")
    print(f"  f_Pearlite = (C_Cementite - C_Steel) / (C_Cementite - C_Eutectoid)")
    print(f"            = ({C_cementite} - {C_steel}) / ({C_cementite} - {C_eutectoid})")
    print(f"            = {f_pearlite:.4f} = {f_pearlite * 100:.2f}%")
    
    print(f"\n[Verification]")
    total = f_cementite + f_pearlite
    print(f"  f_Fe3C + f_Pearlite = {total:.4f}")
    if abs(total - 1.0) < 1e-6:
        print("  Confirmed to equal 1.0")
    
    # Pearlite internal structure analysis
    C_ferrite_eutectoid = 0.022
    f_ferrite_in_pearlite = (C_cementite - C_eutectoid) / (C_cementite - C_ferrite_eutectoid)
    f_cementite_in_pearlite = (C_eutectoid - C_ferrite_eutectoid) / (C_cementite - C_ferrite_eutectoid)
    
    print("\n[Pearlite Internal Structure]")
    print(f"  Ferrite in pearlite: {f_ferrite_in_pearlite * 100:.2f}%")
    print(f"  Cementite in pearlite: {f_cementite_in_pearlite * 100:.2f}%")
    
    # Total cementite in steel
    total_cementite = f_cementite + f_pearlite * f_cementite_in_pearlite
    
    print("\n[Total Phase Composition in Steel]")
    print(f"  Proeutectoid cementite: {f_cementite * 100:.2f}%")
    print(f"  Pearlite: {f_pearlite * 100:.2f}%")
    print(f"  Total cementite: {total_cementite * 100:.2f}%")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left figure: Cooling curve with microstructure changes
    temperatures = [1000, 900, 850, 727, 500, 25]
    time = np.arange(len(temperatures))
    
    ax1.plot(time, temperatures, 'ro-', linewidth=2.5, markersize=10)
    ax1.axhline(T_eutectoid, color='purple', linestyle='--', linewidth=2, alpha=0.7,
                label='Eutectoid Temperature (727 deg C)')
    ax1.axhline(850, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label='gamma + Fe3C boundary (~850 deg C)')
    
    ax1.text(0.5, 1050, 'gamma phase\n(Austenite)', fontsize=12, fontweight='bold',
             ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax1.text(2, 800, 'gamma + Proeutectoid Fe3C\nPrecipitation', fontsize=11, fontweight='bold',
             ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.text(4, 650, 'Proeutectoid Fe3C\n+ Pearlite', fontsize=12, fontweight='bold',
             ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax1.set_xlabel('Cooling Time (arbitrary units)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Temperature (deg C)', fontsize=13, fontweight='bold')
    ax1.set_title('Fe-1.2% C Steel Microstructure Changes During Cooling',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0, 1100)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right figure: Microstructure pie chart
    phases = ['Proeutectoid Cementite', f'Pearlite\n({f_pearlite * 100:.1f}%)']
    sizes = [f_cementite * 100, f_pearlite * 100]
    colors_pie = ['darkgray', 'lightcoral']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=phases,
                                        colors=colors_pie, autopct='%1.2f%%',
                                        startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax2.set_title('Fe-1.2% C Steel Microstructure Composition (Lever Rule)',
                  fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('fe_12c_hypereutectoid_cooling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("[Hypereutectoid Steel Characteristics]")
    print("=" * 80)
    print("Hypereutectoid steel differs from hypoeutectoid steel in that:")
    print("  - Proeutectoid cementite (hard, brittle) forms instead of proeutectoid ferrite (soft)")
    print("  - Due to the hard cementite network, steel has high hardness (HV 200+)")
    print("  - However, the continuous cementite network reduces toughness")
    print("  - Spheroidizing annealing can improve toughness by fragmenting cementite")
    
    print("\n[Comparison with Hypoeutectoid Steel]")
    print("Hypoeutectoid steel: Proeutectoid ferrite (soft) + Pearlite")
    print("  - Good balance of ductility and strength")
    print("Hypereutectoid steel: Proeutectoid cementite (hard) + Pearlite")
    print("  - High hardness, lower toughness")
    print("=" * 80)
    print("\n Graph saved as 'fe_12c_hypereutectoid_cooling.png'")

### Learning Objectives Checklist

#### Level 1: Basic Understanding (Knowledge Acquisition)

  * Understand and apply the Gibbs Phase Rule
  * Explain basic phase diagram elements (liquidus, solidus, eutectic point, eutectoid point)
  * Distinguish between substitutional and interstitial solid solutions
  * Identify main phases in Fe-C phase diagram (ferrite, austenite, cementite)
  * Classify hypoeutectoid, eutectoid, and hypereutectoid steels

#### Level 2: Practical Skills (Calculation and Application)

  * Calculate degrees of freedom using Gibbs Phase Rule
  * Calculate phase mass fractions using the lever rule
  * Predict steel microstructure from Fe-C phase diagram
  * Explain relationship between Gibbs free energy and phase stability
  * Create simplified phase diagrams using Python
  * Perform basic phase diagram calculations using pycalphad

#### Level 3: Application Ability (Problem Solving and Optimization)

  * Analyze and predict microstructure of alloy systems
  * Predict microstructure changes based on cooling rate
  * Design heat treatment conditions to achieve desired microstructure and properties
  * Perform advanced phase diagram calculations using pycalphad and CALPHAD
  * Combine experimental data and thermodynamic calculations for materials design

## References

  1. Porter, D.A., Easterling, K.E., Sherif, M.Y. (2009). _Phase Transformations in Metals and Alloys_ , 3rd ed. CRC Press, pp. 1-102, 215-289.
  2. Callister, W.D., Jr., Rethwisch, D.G. (2020). _Materials Science and Engineering: An Introduction_ , 10th ed. Wiley, pp. 269-345.
  3. Krauss, G. (2015). _Steels: Processing, Structure, and Performance_ , 2nd ed. ASM International, pp. 1-89, 155-234.
  4. Hillert, M. (2008). _Phase Equilibria, Phase Diagrams and Phase Transformations: Their Thermodynamic Basis_ , 2nd ed. Cambridge University Press, pp. 1-78, 234-301.
  5. Massalski, T.B., ed. (1990). _Binary Alloy Phase Diagrams_ , 2nd ed. ASM International.
  6. Otis, R., Liu, Z.-K. (2017). "pycalphad: CALPHAD-based Computational Thermodynamics in Python". _Journal of Open Research Software_ , 5(1), p.1. DOI: 10.5334/jors.140
  7. pycalphad documentation. <https://pycalphad.org/>

[Previous: Chapter 1: Metallic Bonding and Crystal Structure](<chapter-1.html>)[Next: Chapter 3: Strengthening Mechanisms of Metallic Materials](<chapter-3.html>)
