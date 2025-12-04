---
title: "Chapter 4: Reading and Analyzing Binary Phase Diagrams"
chapter_title: "Chapter 4: Reading and Analyzing Binary Phase Diagrams"
subtitle: "Understanding phase diagrams of practical alloys: Cu-Ni complete solid solution, Al-Si eutectic system, Pb-Sn peritectic system"
---

This chapter covers Reading and Analyzing Binary Phase Diagrams. You will learn essential concepts and techniques.

## Learning Objectives

In this chapter, we will learn how to read and analyze **binary phase diagrams** , one of the most important tools in materials science. Phase diagrams visually represent phase states as a function of composition and temperature, and are essential for alloy design and heat treatment process design.

#### Skills to Master in This Chapter

  * Reading phase diagrams of complete solid solution systems (Cu-Ni system, etc.)
  * Characteristics and microstructure formation of eutectic systems (Al-Si, Pb-Sn systems, etc.)
  * Reaction mechanisms of peritectic and monotectic systems
  * Analysis of complex phase diagrams containing intermetallic compounds
  * Quantitative calculation of phase fractions using the lever rule
  * Relationship between cooling curves and microstructure formation
  * Composition-microstructure mapping of practical alloys

#### =¡ Importance of Binary Phase Diagrams

Binary phase diagrams show equilibrium states of alloy systems consisting of two components. Many industrially important materials such as Fe-C system for steel, Al-Cu system and Al-Si system for aluminum alloys are based on binary phase diagrams. The skill to accurately read phase diagrams is essential for materials developers.

## Flow for Reading Binary Phase Diagrams

When reading a phase diagram, follow these steps in order:
    
    
    ```mermaid
    graph TD
        A[1. Check axes] --> B[2. Identify phase regions]
        B --> C[3. Trace phase boundaries]
        C --> D[4. Check invariant points]
        D --> E[5. Set composition point]
        E --> F[6. Determine equilibrium state at temperature]
        F --> G[7. Apply lever rule]
        G --> H[8. Trace cooling path]
    
        style A fill:#f093fb,stroke:#f5576c,color:#fff
        style H fill:#f093fb,stroke:#f5576c,color:#fff
    ```

#### Basic Principles of Phase Diagram Reading

  1. **Horizontal axis (composition)** : Left end is component A (0 wt%), right end is component B (100 wt%)
  2. **Vertical axis (temperature)** : Temperature increases from bottom to top
  3. **Phase regions** : Distinguished by labels such as ±, ², L (liquid phase)
  4. **Phase boundaries** : Liquidus, solidus, solubility curves, etc.
  5. **Invariant points** : Reaction points at specific temperature and composition such as eutectic point, peritectic point, monotectic point

## 1\. Complete Solid Solution System (Cu-Ni System)

In complete solid solution systems, continuous solid solution occurs over the entire composition range from liquid to solid phase. The Cu-Ni system is a typical example where complete solid solution is possible because both components have the same FCC structure and similar atomic radii.

### 1.1 Characteristics of Cu-Ni Phase Diagram

  * **Liquidus** : Completely liquid (L phase) above this line
  * **Solidus** : Completely solid (± phase) below this line
  * **Two-phase region (L + ±)** : Temperature range where liquid and solid phases coexist
  * **Equilibrium solidification** : With extremely slow cooling, gradual solidification occurs from liquidus to solidus

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: 1.1 Characteristics of Cu-Ni Phase Diagram
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
                <h4>Code Example 1: Creating Cu-Ni Complete Solid Solution Phase Diagram</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    
    # Experimental data for Cu-Ni system (approximate values)
    compositions = np.array([0, 20, 40, 60, 80, 100])  # wt% Ni
    liquidus = np.array([1085, 1160, 1260, 1350, 1410, 1455])  # 
    solidus = np.array([1085, 1130, 1220, 1310, 1390, 1455])   # 
    
    # Generate smooth curves with spline interpolation
    comp_smooth = np.linspace(0, 100, 200)
    liquidus_interp = interp1d(compositions, liquidus, kind='cubic')
    solidus_interp = interp1d(compositions, solidus, kind='cubic')
    
    liquidus_smooth = liquidus_interp(comp_smooth)
    solidus_smooth = solidus_interp(comp_smooth)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Fill phase regions
    ax.fill_between(comp_smooth, liquidus_smooth, 1600,
                    alpha=0.3, color='red', label='Liquid (L)')
    ax.fill_between(comp_smooth, solidus_smooth, liquidus_smooth,
                    alpha=0.3, color='yellow', label='L + ± (two-phase region)')
    ax.fill_between(comp_smooth, 0, solidus_smooth,
                    alpha=0.3, color='blue', label='± solid solution')
    
    # Plot liquidus and solidus lines
    ax.plot(comp_smooth, liquidus_smooth, 'r-', linewidth=2.5, label='Liquidus')
    ax.plot(comp_smooth, solidus_smooth, 'b-', linewidth=2.5, label='Solidus')
    
    # Experimental data points
    ax.scatter(compositions, liquidus, c='red', s=80, zorder=5, edgecolors='black')
    ax.scatter(compositions, solidus, c='blue', s=80, zorder=5, edgecolors='black')
    
    # Lever rule application example (60 wt% Ni, 1300)
    example_comp = 60
    example_temp = 1300
    ax.plot(example_comp, example_temp, 'ko', markersize=10, zorder=10,
            label=f'Example: {example_comp} wt% Ni, {example_temp}')
    
    # Horizontal tie line
    liquidus_comp = np.interp(example_temp, liquidus_smooth[::-1], comp_smooth[::-1])
    solidus_comp = np.interp(example_temp, solidus_smooth[::-1], comp_smooth[::-1])
    ax.plot([solidus_comp, liquidus_comp], [example_temp, example_temp],
            'k--', linewidth=1.5, alpha=0.7)
    ax.plot([solidus_comp, liquidus_comp], [example_temp, example_temp],
            'ko', markersize=6)
    
    # Labels and formatting
    ax.set_xlabel('Composition (wt% Ni)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature ()', fontsize=13, fontweight='bold')
    ax.set_title('Cu-Ni Complete Solid Solution Phase Diagram', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(1000, 1600)
    
    # Annotate melting points of pure components
    ax.annotate('Cu melting point\n1085', xy=(0, 1085), xytext=(15, 1050),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center')
    ax.annotate('Ni melting point\n1455', xy=(100, 1455), xytext=(85, 1500),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('cu_ni_phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Cu-Ni phase diagram created.")
    print(f"\nLever rule application example:")
    print(f"Composition: {example_comp} wt% Ni, Temperature: {example_temp}")
    print(f"Liquid phase composition: {liquidus_comp:.1f} wt% Ni")
    print(f"Solid phase composition: {solidus_comp:.1f} wt% Ni")
    
    # Lever rule calculation
    f_liquid = (example_comp - solidus_comp) / (liquidus_comp - solidus_comp)
    f_solid = 1 - f_liquid
    print(f"Liquid phase fraction: {f_liquid:.3f} ({f_liquid*100:.1f}%)")
    print(f"Solid phase fraction: {f_solid:.3f} ({f_solid*100:.1f}%)")
    

#### =¡ Principle of the Lever Rule

The lever rule is a method to calculate the fraction of each phase in a two-phase region. Similar to the principle of a lever, the inverse ratio of distances from the composition point to each phase boundary gives the phase fraction:

$$f_L = \frac{C_0 - C_\alpha}{C_L - C_\alpha}, \quad f_\alpha = \frac{C_L - C_0}{C_L - C_\alpha}$$

where \\(C_0\\) is overall composition, \\(C_L\\) is liquid phase composition, and \\(C_\alpha\\) is solid phase composition.

## 2\. Eutectic Systems (Al-Si System, Pb-Sn System)

In eutectic systems, a **eutectic reaction** occurs at a specific composition (eutectic composition) where two solid phases crystallize simultaneously from the liquid phase:

$$L \rightarrow \alpha + \beta \quad (\text{upon cooling})$$

### 2.1 Characteristics of Eutectic Systems

  * **Eutectic Point** : Temperature and composition where eutectic reaction occurs
  * **Eutectic temperature** : Lowest melting point in the system
  * **Solubility limit** : Maximum composition that can dissolve in ± and ² phases respectively
  * **Eutectic microstructure** : Fine microstructure with ± and ² phases in lamellar or rod-like arrangement

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: 2.1 Characteristics of Eutectic Systems
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
                <h4>Code Example 2: Al-Si Eutectic Phase Diagram and Lever Rule Calculation</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    # Phase diagram data for Al-Si system (simplified)
    def create_al_si_phase_diagram():
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # Composition range
        comp = np.linspace(0, 100, 300)
    
        # Liquidus (Al side and Si side)
        Al_side = 660 - 5.5 * comp[:150]  # Al-side liquidus
        Si_side = 1414 - 9.0 * (100 - comp[150:])  # Si-side liquidus
        liquidus = np.concatenate([Al_side, Si_side])
    
        # Eutectic point
        eutectic_comp = 12.6  # wt% Si
        eutectic_temp = 577   # 
    
        # Solubility limits (temperature dependence simplified)
        solubility_Al = np.full(300, 1.65)  # Si solubility limit in ± phase
        solubility_Si = np.full(300, 0.05)  # Al solubility limit in Si phase (nearly 0)
    
        # Fill phase regions
        ax.fill_between(comp, liquidus, 1500, alpha=0.3, color='red', label='Liquid (L)')
        ax.fill_between(comp[:150], eutectic_temp, Al_side,
                        alpha=0.3, color='yellow', label='L + ±')
        ax.fill_between(comp[150:], eutectic_temp, Si_side,
                        alpha=0.3, color='orange', label='L + ²(Si)')
        ax.fill_between(comp, 0, eutectic_temp, alpha=0.2, color='blue', label='± + ²')
    
        # Plot liquidus
        ax.plot(comp, liquidus, 'r-', linewidth=2.5, label='Liquidus')
    
        # Eutectic line
        ax.plot([0, 100], [eutectic_temp, eutectic_temp],
                'g--', linewidth=2, label='Eutectic line')
    
        # Eutectic point
        ax.plot(eutectic_comp, eutectic_temp, 'ro', markersize=12,
                zorder=10, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'Eutectic point\n({eutectic_comp} wt% Si, {eutectic_temp})',
                    xy=(eutectic_comp, eutectic_temp),
                    xytext=(eutectic_comp + 15, eutectic_temp + 80),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
        # Solubility limits
        ax.plot([solubility_Al[0], solubility_Al[0]], [0, eutectic_temp],
                'b--', linewidth=1.5, alpha=0.7)
        ax.plot([100-solubility_Si[0], 100-solubility_Si[0]], [0, eutectic_temp],
                'b--', linewidth=1.5, alpha=0.7)
    
        # Lever rule application example (8 wt% Si, 600)
        example_comp = 8.0
        example_temp = 600
        ax.plot(example_comp, example_temp, 'ko', markersize=10, zorder=10)
    
        # Tie line
        liquidus_comp_at_temp = eutectic_comp + (example_temp - eutectic_temp) * 0.2
        ax.plot([solubility_Al[0], liquidus_comp_at_temp],
                [example_temp, example_temp], 'k--', linewidth=1.5)
        ax.plot([solubility_Al[0], liquidus_comp_at_temp],
                [example_temp, example_temp], 'ko', markersize=6)
    
        # Labels and formatting
        ax.set_xlabel('Composition (wt% Si)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Temperature ()', fontsize=13, fontweight='bold')
        ax.set_title('Al-Si Eutectic Phase Diagram', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
        ax.set_ylim(500, 750)
    
        # Melting point of pure component
        ax.annotate('Al melting point\n660', xy=(0, 660), xytext=(5, 720),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=10)
    
        plt.tight_layout()
        plt.savefig('al_si_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Lever rule calculation
        C0 = example_comp
        C_alpha = solubility_Al[0]
        C_L = liquidus_comp_at_temp
    
        f_L = (C0 - C_alpha) / (C_L - C_alpha)
        f_alpha = 1 - f_L
    
        print(f"\n=== Lever Rule Calculation Results ===")
        print(f"Overall composition: {C0} wt% Si")
        print(f"Temperature: {example_temp}")
        print(f"± phase composition: {C_alpha} wt% Si")
        print(f"Liquid phase composition: {C_L:.1f} wt% Si")
        print(f"Liquid phase fraction: {f_L:.3f} ({f_L*100:.1f}%)")
        print(f"± phase fraction: {f_alpha:.3f} ({f_alpha*100:.1f}%)")
    
    create_al_si_phase_diagram()
    

### 2.2 Pb-Sn Eutectic System and Cooling Curves

The Pb-Sn system is a eutectic alloy widely used as solder material. At the eutectic composition (61.9 wt% Sn), transformation occurs directly from liquid phase to eutectic microstructure, resulting in a clear thermal arrest in the cooling curve.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 3: Cooling Curve Simulation for Pb-Sn Eutectic System</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_cooling_curve(composition_Sn, initial_temp=300, final_temp=100,
                               cooling_rate=0.5, latent_heat_factor=20):
        """
        Simulate cooling curve for Pb-Sn system
    
        Parameters:
        -----------
        composition_Sn : float
            Sn composition (wt%)
        initial_temp : float
            Initial temperature ()
        final_temp : float
            Final temperature ()
        cooling_rate : float
            Cooling rate (/s)
        latent_heat_factor : float
            Strength of temperature arrest due to latent heat
        """
        # Phase diagram parameters
        eutectic_comp = 61.9  # wt% Sn
        eutectic_temp = 183   # 
        Pb_melting = 327      # 
        Sn_melting = 232      # 
    
        # Estimate liquidus temperature from composition
        if composition_Sn < eutectic_comp:
            liquidus_temp = Pb_melting - (Pb_melting - eutectic_temp) * (composition_Sn / eutectic_comp)
        else:
            liquidus_temp = Sn_melting - (Sn_melting - eutectic_temp) * ((100 - composition_Sn) / (100 - eutectic_comp))
    
        # Time array
        time = np.arange(0, (initial_temp - final_temp) / cooling_rate, 0.1)
        temperature = np.zeros_like(time)
    
        for i, t in enumerate(time):
            # Basic cooling
            temp = initial_temp - cooling_rate * t
    
            # Arrest at liquidus (solidification start)
            if abs(temp - liquidus_temp) < 5:
                temp += latent_heat_factor * np.exp(-((temp - liquidus_temp)**2) / 10)
    
            # Arrest at eutectic temperature
            if abs(temp - eutectic_temp) < 5:
                # Arrest more pronounced closer to eutectic composition
                eutectic_factor = 1.0 - abs(composition_Sn - eutectic_comp) / eutectic_comp
                temp += latent_heat_factor * eutectic_factor * np.exp(-((temp - eutectic_temp)**2) / 10)
    
            temperature[i] = temp
    
        return time, temperature
    
    # Calculate cooling curves for different compositions
    compositions = [20, 40, 61.9, 80]  # wt% Sn
    colors = ['blue', 'green', 'red', 'purple']
    labels = [f'{comp} wt% Sn' for comp in compositions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot cooling curves
    for comp, color, label in zip(compositions, colors, labels):
        time, temp = simulate_cooling_curve(comp)
        if comp == 61.9:
            ax1.plot(time, temp, color=color, linewidth=2.5, label=label + ' (eutectic)', linestyle='-')
        else:
            ax1.plot(time, temp, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax1.axhline(y=183, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Eutectic temperature (183)')
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temperature ()', fontsize=12, fontweight='bold')
    ax1.set_title('Cooling Curves for Pb-Sn System', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(100, 350)
    
    # Cooling rate (time derivative of temperature)
    ax2.set_title('Cooling Rate (dT/dt)', fontsize=14, fontweight='bold')
    for comp, color, label in zip(compositions, colors, labels):
        time, temp = simulate_cooling_curve(comp)
        cooling_rate = np.gradient(temp, time)
        if comp == 61.9:
            ax2.plot(temp, -cooling_rate, color=color, linewidth=2.5, label=label + ' (eutectic)')
        else:
            ax2.plot(temp, -cooling_rate, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax2.axvline(x=183, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Temperature ()', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute cooling rate (/s)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(100, 350)
    
    plt.tight_layout()
    plt.savefig('pb_sn_cooling_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Characteristics of Cooling Curves ===")
    print("1. Arrest at liquidus: Latent heat release due to solidification start")
    print("2. Arrest at eutectic temperature: Significant latent heat release from eutectic reaction (L ’ ± + ²)")
    print("3. Most clear arrest at eutectic composition (61.9 wt% Sn)")
    print("4. Two-stage arrest observed for compositions away from eutectic")
    

## 3\. Peritectic and Monotectic Systems

### 3.1 Peritectic Reaction

A peritectic reaction is where a liquid phase reacts with a solid phase to form a new solid phase:

$$L + \alpha \rightarrow \beta \quad (\text{upon cooling})$$

This is observed in many alloy systems such as Pt-Ag, Fe-Ni, and Cu-Zn systems.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: This is observed in many alloy systems such as Pt-Ag, Fe-Ni,
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
                <h4>Code Example 4: Visualization of Peritectic Reaction (Pt-Ag System Model)</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    
    def create_peritectic_diagram():
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # Composition range
        comp = np.linspace(0, 100, 500)
    
        # Liquidus of peritectic system (simplified model)
        # Pt side
        liquidus_left = 1769 - 15 * comp[:200]
        # Ag side
        liquidus_right = 961 + 7 * (100 - comp[300:])
        # Near peritectic point
        liquidus_middle = np.linspace(liquidus_left[-1], liquidus_right[0], 100)
        liquidus = np.concatenate([liquidus_left, liquidus_middle, liquidus_right])
    
        # Peritectic point parameters
        peritectic_comp = 42  # wt% Ag
        peritectic_temp = 1186  # 
    
        # Solidus
        solidus_left = np.full(200, peritectic_temp)
        solidus_middle = np.full(100, peritectic_temp)
        solidus_right = 961 + 5 * (100 - comp[300:])
        solidus = np.concatenate([solidus_left, solidus_middle, solidus_right])
    
        # ± phase solubility limit
        alpha_limit = 15  # wt% Ag
    
        # Fill phase regions
        ax.fill_between(comp, liquidus, 2000, alpha=0.3, color='red', label='Liquid (L)')
        ax.fill_between(comp[:200], solidus[:200], liquidus[:200],
                        alpha=0.3, color='yellow', label='L + ±')
        ax.fill_between(comp[200:], solidus[200:], liquidus[200:],
                        alpha=0.3, color='orange', label='L + ²')
        ax.fill_between(comp, 800, solidus, alpha=0.2, color='blue', label='² phase')
    
        # ± phase region
        ax.fill_between(comp[:100], 800, peritectic_temp,
                        where=(comp[:100] <= alpha_limit),
                        alpha=0.3, color='cyan', label='± phase')
        ax.fill_between(comp[:100], 800, peritectic_temp,
                        where=(comp[:100] > alpha_limit),
                        alpha=0.3, color='lightblue', label='± + ²')
    
        # Plot liquidus and solidus lines
        ax.plot(comp, liquidus, 'r-', linewidth=2.5, label='Liquidus')
        ax.plot(comp, solidus, 'b-', linewidth=2.5, label='Solidus')
    
        # Peritectic line
        ax.plot([0, 100], [peritectic_temp, peritectic_temp],
                'g--', linewidth=2, label='Peritectic line')
    
        # Peritectic point
        ax.plot(peritectic_comp, peritectic_temp, 'ro', markersize=14,
                zorder=10, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'Peritectic point\n({peritectic_comp} wt% Ag, {peritectic_temp})',
                    xy=(peritectic_comp, peritectic_temp),
                    xytext=(peritectic_comp + 15, peritectic_temp + 150),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
        # Arrows for peritectic reaction
        arrow1 = FancyArrowPatch((peritectic_comp - 10, peritectic_temp + 50),
                                (peritectic_comp, peritectic_temp + 10),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='red', label='Liquid')
        arrow2 = FancyArrowPatch((peritectic_comp - 20, peritectic_temp + 50),
                                (peritectic_comp, peritectic_temp + 10),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='cyan', label='± phase')
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
    
        ax.text(peritectic_comp + 5, peritectic_temp - 30, 'L + ± ’ ²',
                fontsize=13, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2))
    
        # Labels and formatting
        ax.set_xlabel('Composition (wt% Ag)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Temperature ()', fontsize=13, fontweight='bold')
        ax.set_title('Peritectic Phase Diagram (Pt-Ag System Model)', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 80)
        ax.set_ylim(900, 1900)
    
        # Melting point of pure component
        ax.annotate('Pt melting point\n1769', xy=(0, 1769), xytext=(10, 1850),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=10)
    
        plt.tight_layout()
        plt.savefig('peritectic_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        print("\n=== Characteristics of Peritectic Reaction ===")
        print("1. L + ± ’ ²: Liquid phase reacts with previously crystallized ± phase to form ² phase")
        print("2. Below peritectic point, ± phase is surrounded by ² phase (core-shell structure)")
        print("3. Long-time diffusion required for complete equilibrium state")
        print("4. With rapid cooling, ± phase remains as residual core")
    
    create_peritectic_diagram()
    

#### =¡ Practical Importance of Peritectic Reactions

Peritectic reactions frequently appear in industrially important processes such as steel solidification (´-Fe + L ’ ³-Fe) and synthesis of high-temperature superconductors (YBCO). In peritectic reactions, diffusion becomes rate-controlling because the previously crystallized phase is surrounded by the later-formed phase, requiring considerable time to reach equilibrium.

## 4\. Systems Containing Intermetallic Compounds

Intermetallic compounds are compounds with ordered structure formed at specific stoichiometric compositions. Representative examples include ¸ phase (Al‚Cu) in Al-Cu system and NiƒAl in Ni-Al system.

### 4.1 Characteristics of Al-Cu System

  * **± phase** : Cu solid solution in Al (FCC structure)
  * **¸ phase (Al‚Cu)** : Stoichiometric compound
  * **Eutectic reaction** : L ’ ± + ¸ (548)
  * **Age hardening** : Strengthening through ¸' precipitation from supersaturated solid solution

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: 4.1 Characteristics of Al-Cu System
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
                <h4>Code Example 5: System with Intermetallic Compounds (Al-Cu System Model)</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    def create_al_cu_phase_diagram():
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # Composition range (wt% Cu)
        comp = np.linspace(0, 60, 600)
    
        # Al-side liquidus
        liquidus_Al = 660 - 8 * comp[:250]
        # Near ¸ phase (Al2Cu)
        theta_comp = 33.2  # Stoichiometric composition of Al2Cu (wt% Cu)
        liquidus_theta = np.full(100, 591)  # Near melting point of ¸ phase
        # Cu-side liquidus
        liquidus_Cu = 1085 - 15 * (60 - comp[350:])
        # Intermediate parts
        liquidus_mid = np.linspace(liquidus_Al[-1], liquidus_theta[0], 50)
        liquidus_mid2 = np.linspace(liquidus_theta[-1], liquidus_Cu[0], 50)
        liquidus = np.concatenate([liquidus_Al, liquidus_mid, liquidus_theta, liquidus_mid2, liquidus_Cu])
    
        # Eutectic temperature
        eutectic_temp = 548  # 
        eutectic_comp = 32.5  # wt% Cu
    
        # Solubility limit (temperature dependence simplified)
        solubility_Al_at_eutectic = 5.65  # wt% Cu at 548
        solubility_Al_at_room_temp = 0.1   # wt% Cu at 25
    
        # Fill phase regions
        ax.fill_between(comp, liquidus, 1200, alpha=0.3, color='red', label='Liquid (L)')
        ax.fill_between(comp[:250], eutectic_temp, liquidus[:250],
                        alpha=0.3, color='yellow', label='L + ±')
        ax.fill_between(comp[350:], eutectic_temp, liquidus[350:],
                        alpha=0.3, color='orange', label='L + ¸')
    
        # ± + ¸ two-phase region
        ax.fill_between(comp[:400], 0, eutectic_temp,
                        where=(comp[:400] > solubility_Al_at_room_temp),
                        alpha=0.2, color='purple', label='± + ¸')
    
        # ± single-phase region
        ax.fill_between(comp[:100], 0, eutectic_temp,
                        where=(comp[:100] <= solubility_Al_at_eutectic),
                        alpha=0.3, color='cyan', label='± (Al solid solution)')
    
        # ¸ phase region
        ax.fill_between(comp[300:500], 0, liquidus_theta[0],
                        alpha=0.3, color='lightgreen', label='¸ (Al‚Cu)')
    
        # Plot liquidus
        ax.plot(comp, liquidus, 'r-', linewidth=2.5, label='Liquidus')
    
        # Eutectic line
        ax.plot([0, 60], [eutectic_temp, eutectic_temp],
                'g--', linewidth=2, label='Eutectic line')
    
        # Eutectic point
        ax.plot(eutectic_comp, eutectic_temp, 'ro', markersize=12,
                zorder=10, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'Eutectic point\n({eutectic_comp} wt% Cu, {eutectic_temp})',
                    xy=(eutectic_comp, eutectic_temp),
                    xytext=(eutectic_comp - 10, eutectic_temp + 80),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
        # Stoichiometric composition of ¸ phase
        ax.axvline(x=theta_comp, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(theta_comp + 1, 1000, 'Al‚Cu\n(33.2 wt% Cu)',
                fontsize=10, fontweight='bold', color='green')
    
        # Solubility limit curve (simplified)
        solubility_curve_T = np.linspace(25, eutectic_temp, 100)
        solubility_curve_C = solubility_Al_at_room_temp + \
                             (solubility_Al_at_eutectic - solubility_Al_at_room_temp) * \
                             ((solubility_curve_T - 25) / (eutectic_temp - 25))**0.5
        ax.plot(solubility_curve_C, solubility_curve_T, 'b-', linewidth=2,
                label='Solubility limit of ± phase')
    
        # Example of age hardening treatment (4 wt% Cu alloy)
        aging_comp = 4.0
        solution_treat_temp = 520  # Solution treatment temperature
        aging_temp = 190           # Aging treatment temperature
    
        # Arrows for treatment path
        ax.annotate('', xy=(aging_comp, solution_treat_temp),
                    xytext=(aging_comp, 25),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2.5, linestyle='--'))
        ax.plot(aging_comp, solution_treat_temp, 'bs', markersize=10,
                label='Solution treatment', zorder=10)
        ax.plot(aging_comp, 25, 'b^', markersize=10,
                label='Quench (supersaturated solid solution)', zorder=10)
        ax.plot(aging_comp, aging_temp, 'b*', markersize=15,
                label='Aging treatment (precipitation hardening)', zorder=10)
    
        # Labels and formatting
        ax.set_xlabel('Composition (wt% Cu)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Temperature ()', fontsize=13, fontweight='bold')
        ax.set_title('Al-Cu Phase Diagram and Age Hardening Treatment', fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 750)
    
        plt.tight_layout()
        plt.savefig('al_cu_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        print("\n=== Age Hardening Treatment of Al-Cu System ===")
        print(f"1. Solution treatment: Heat to {solution_treat_temp} to completely dissolve Cu")
        print(f"2. Quench: Rapid cooling to room temperature to form supersaturated ± solid solution")
        print(f"3. Aging treatment: Heat to {aging_temp} to precipitate ¸' phase (metastable phase)")
        print("4. ¸' precipitates obstruct dislocation movement ’ Strengthening (Duralumin)")
        print(f"5. Over-aging: Growth to ¸ phase (equilibrium phase) with long aging ’ Strength decrease")
    
    create_al_cu_phase_diagram()
    

#### Principle of Age Hardening

Al-Cu system alloys (Duralumin) are strengthened through the following process:

  1. **Solution treatment** : Heat in ± single-phase region (around 520) to completely dissolve Cu in ± phase
  2. **Quench** : Water quench to form supersaturated solid solution (non-equilibrium state)
  3. **Aging treatment** : Heat at around 190 to form fine ¸' precipitates (GP zone ’ ¸'')
  4. **Precipitation strengthening** : nm to tens of nm precipitates obstruct dislocation movement, improving strength by 2-3 times

## 5\. Temperature Dependence of Solubility Limit

The solubility limit is the maximum solute concentration that can dissolve in a solid solution at a given temperature. In most systems, the solubility limit increases with temperature.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 6: Temperature Dependence of Solubility Limit (Al-Cu System ± Phase)</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Experimental data (solubility limit of ± phase in Al-Cu system)
    temp_data = np.array([25, 100, 200, 300, 400, 500, 548])  # 
    solubility_data = np.array([0.1, 0.5, 1.5, 2.8, 4.0, 5.2, 5.65])  # wt% Cu
    
    # Arrhenius-type fitting function
    def solubility_model(T, A, B):
        """
        Temperature dependence of solubility (simplified model)
        C = A * exp(B / T)
        """
        T_kelvin = T + 273.15
        return A * np.exp(B / T_kelvin)
    
    # Fitting
    popt, pcov = curve_fit(solubility_model, temp_data, solubility_data, p0=[0.01, 3000])
    A_fit, B_fit = popt
    
    # Smooth curve
    temp_smooth = np.linspace(25, 548, 200)
    solubility_smooth = solubility_model(temp_smooth, A_fit, B_fit)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Solubility limit curve
    ax1.plot(temp_smooth, solubility_smooth, 'b-', linewidth=2.5, label='Fitting curve')
    ax1.scatter(temp_data, solubility_data, s=100, c='red', zorder=5,
                edgecolors='black', linewidths=2, label='Experimental data')
    
    ax1.set_xlabel('Temperature ()', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Solubility limit (wt% Cu)', fontsize=13, fontweight='bold')
    ax1.set_title('Solubility Limit of ± Phase in Al-Cu System', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 600)
    ax1.set_ylim(0, 6.5)
    
    # Show composition range for age hardening
    ax1.fill_between(temp_smooth, 0, solubility_smooth, alpha=0.2, color='green',
                     label='Age hardenable region')
    ax1.axhline(y=4.0, color='orange', linestyle='--', linewidth=2,
                label='Typical alloy composition (4 wt% Cu)')
    
    # Arrhenius plot (ln(C) vs 1/T)
    temp_kelvin = temp_data + 273.15
    ln_solubility = np.log(solubility_data)
    
    ax2.scatter(1000/temp_kelvin, ln_solubility, s=100, c='red', zorder=5,
                edgecolors='black', linewidths=2, label='Experimental data')
    
    # Fitting line
    temp_kelvin_smooth = temp_smooth + 273.15
    ln_solubility_smooth = np.log(solubility_model(temp_smooth, A_fit, B_fit))
    ax2.plot(1000/temp_kelvin_smooth, ln_solubility_smooth, 'b-', linewidth=2.5,
             label='Linear fit')
    
    ax2.set_xlabel('1000/T (K{¹)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('ln(solubility limit)', fontsize=13, fontweight='bold')
    ax2.set_title('Arrhenius Plot', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Display fitting parameters
    textstr = f'C = A·exp(B/T)\nA = {A_fit:.4f}\nB = {B_fit:.1f} K'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('al_cu_solubility_limit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate optimal temperature range for age hardening treatment
    print("\n=== Solubility Limit Data Analysis ===")
    print(f"Fitting parameters: A = {A_fit:.4f}, B = {B_fit:.1f} K")
    print(f"\nSolubility limit by temperature:")
    for T in [25, 100, 200, 300, 400, 500]:
        C = solubility_model(T, A_fit, B_fit)
        print(f"  {T}: {C:.2f} wt% Cu")
    
    print(f"\nFor 4 wt% Cu alloy:")
    target_composition = 4.0
    # Calculate temperature where solubility limit equals 4.0 wt%
    def find_solvus_temp(C_target):
        T_kelvin = B_fit / np.log(C_target / A_fit)
        return T_kelvin - 273.15
    
    solvus_temp = find_solvus_temp(target_composition)
    print(f"  Solvus temperature: {solvus_temp:.1f}")
    print(f"  Solution treatment temperature: above {solvus_temp + 20:.1f} (typically 520)")
    print(f"  Aging treatment temperature: 150-200 (maintain supersaturation)")
    print(f"  Precipitable Cu amount: {target_composition - solubility_model(190, A_fit, B_fit):.2f} wt%")
    

## 6\. Composition-Microstructure Mapping of Practical Materials

Correlating predicted microstructure from phase diagrams with actual microstructure is extremely important in materials design. Here, we map the relationship between composition and microstructure for Al-Si casting alloys.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Correlating predicted microstructure from phase diagrams wit
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
                <h4>Code Example 7: Composition-Microstructure Mapping of Practical Materials (Al-Si Casting Alloys)</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Composition and microstructure data for Al-Si casting alloys
    alloy_data = {
        'Alloy': ['A356', 'A380', 'A383', 'A390', 'Silumin', 'Hypereutectic Al-Si'],
        'Si_content': [7.0, 8.5, 10.5, 17.0, 12.6, 18.0],  # wt% Si
        'Mg_content': [0.35, 0.0, 0.0, 0.55, 0.0, 0.0],
        'Microstructure': ['Hypoeutectic', 'Hypoeutectic', 'Hypoeutectic', 'Hypereutectic', 'Eutectic', 'Hypereutectic'],
        'Primary_phase': ['±-Al', '±-Al', '±-Al', 'Primary Si', 'Eutectic', 'Primary Si'],
        'UTS_MPa': [228, 317, 310, 283, 180, 250],  # Ultimate Tensile Strength
        'Elongation': [8.0, 3.5, 3.0, 1.0, 5.0, 0.5],  # %
        'Application': ['Auto parts', 'Die casting', 'Die casting', 'Engine', 'General casting', 'Wear resistant']
    }
    
    df = pd.DataFrame(alloy_data)
    
    # Phase diagram based plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Composition and microstructure relationship
    ax1 = fig.add_subplot(gs[0, :])
    
    # Phase diagram background
    eutectic_comp = 12.6
    ax1.axvline(x=eutectic_comp, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label='Eutectic composition')
    ax1.axvspan(0, eutectic_comp, alpha=0.2, color='blue', label='Hypoeutectic region')
    ax1.axvspan(eutectic_comp, 25, alpha=0.2, color='orange', label='Hypereutectic region')
    
    # Position of each alloy
    colors = {'Hypoeutectic': 'blue', 'Eutectic': 'red', 'Hypereutectic': 'orange'}
    for idx, row in df.iterrows():
        color = colors[row['Microstructure']]
        marker = 'o' if row['Mg_content'] == 0 else '^'
        ax1.scatter(row['Si_content'], idx + 1, s=200, c=color, marker=marker,
                    edgecolors='black', linewidths=2, zorder=5)
        ax1.text(row['Si_content'] + 0.5, idx + 1, row['Alloy'],
                 fontsize=11, va='center', fontweight='bold')
    
    ax1.set_xlabel('Si content (wt%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Alloy number (order only)', fontsize=13, fontweight='bold')
    ax1.set_title('Composition Distribution of Al-Si Casting Alloys', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 22)
    ax1.set_ylim(0, len(df) + 1)
    ax1.set_yticks([])
    
    # 2. Tensile strength and Si content relationship
    ax2 = fig.add_subplot(gs[1, 0])
    
    for idx, row in df.iterrows():
        color = colors[row['Microstructure']]
        marker = 'o' if row['Mg_content'] == 0 else '^'
        ax2.scatter(row['Si_content'], row['UTS_MPa'], s=150, c=color,
                    marker=marker, edgecolors='black', linewidths=2)
        ax2.text(row['Si_content'], row['UTS_MPa'] + 10, row['Alloy'],
                 fontsize=9, ha='center')
    
    ax2.axvline(x=eutectic_comp, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Si content (wt%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tensile strength (MPa)', fontsize=12, fontweight='bold')
    ax2.set_title('Mechanical Strength and Si Content', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(5, 20)
    ax2.set_ylim(150, 350)
    
    # 3. Elongation and Si content relationship
    ax3 = fig.add_subplot(gs[1, 1])
    
    for idx, row in df.iterrows():
        color = colors[row['Microstructure']]
        marker = 'o' if row['Mg_content'] == 0 else '^'
        ax3.scatter(row['Si_content'], row['Elongation'], s=150, c=color,
                    marker=marker, edgecolors='black', linewidths=2)
        ax3.text(row['Si_content'], row['Elongation'] + 0.3, row['Alloy'],
                 fontsize=9, ha='center')
    
    ax3.axvline(x=eutectic_comp, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Si content (wt%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Elongation (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Ductility and Si Content', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(5, 20)
    ax3.set_ylim(0, 10)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=10, label='Hypoeutectic (±-Al primary)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, label='Eutectic (fine microstructure)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=10, label='Hypereutectic (Si primary)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=10, label='Mg-added alloy'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.savefig('al_si_composition_microstructure_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Data table output
    print("\n=== Composition-Microstructure-Property Mapping of Al-Si Casting Alloys ===\n")
    print(df.to_string(index=False))
    
    print("\n\n=== Composition and Microstructure Relationship ===")
    print("1. Hypoeutectic alloys (< 12.6 wt% Si):")
    print("   - ±-Al primary + Al-Si eutectic microstructure")
    print("   - Relatively high ductility (3-8%)")
    print("   - Applications: Auto parts, die casting")
    
    print("\n2. Eutectic alloys (12.6 wt% Si):")
    print("   - Complete eutectic microstructure (±-Al + Si lamellar structure)")
    print("   - Lowest melting point (577) ’ Good castability")
    print("   - Applications: General castings, decorative items")
    
    print("\n3. Hypereutectic alloys (> 12.6 wt% Si):")
    print("   - Coarse Si primary + Al-Si eutectic microstructure")
    print("   - High wear resistance, low ductility (< 1%)")
    print("   - Applications: Engine blocks, wear-resistant parts")
    print("   - P addition enables refinement of primary Si")
    
    print("\n4. Effect of Mg addition (A356, A390):")
    print("   - Age hardening through Mg‚Si precipitation")
    print("   - Strength increase (200 ’ 280 MPa)")
    print("   - Maximum strength with T6 treatment (solution + aging)")
    

#### Selection Criteria for Practical Alloys

Required Property | Recommended Composition Range | Representative Alloy | Considerations  
---|---|---|---  
High strength | 7-10 wt% Si + Mg | A356, A380 | Age hardening treatment required  
High ductility | 7-9 wt% Si | A356 | Ductility decreases with Mg addition  
Castability | 10-13 wt% Si | Silumin, A380 | Best near eutectic composition  
Wear resistance | 17-18 wt% Si | A390 | P addition for primary refinement  
  
## Exercises

#### Exercise 1: Lever Rule Calculation for Cu-Ni System

**Problem:** A Cu-60wt%Ni alloy is cooled very slowly from 1300. Determine the equilibrium state at 1250.

**Given information:**

  * Liquidus composition at 1250: 55 wt% Ni
  * Solidus composition at 1250: 68 wt% Ni
  * Overall composition: 60 wt% Ni

**Find:**

  1. Fraction of liquid phase
  2. Fraction of solid phase
  3. Composition of each phase

View solution

**Solution:**

Apply lever rule:

$$f_L = \frac{C_\alpha - C_0}{C_\alpha - C_L} = \frac{68 - 60}{68 - 55} = \frac{8}{13} = 0.615$$

$$f_\alpha = \frac{C_0 - C_L}{C_\alpha - C_L} = \frac{60 - 55}{68 - 55} = \frac{5}{13} = 0.385$$

**Answer:**

  * Liquid phase fraction: 61.5% (composition: 55 wt% Ni)
  * Solid phase fraction: 38.5% (composition: 68 wt% Ni)

#### Exercise 2: Microstructure Prediction for Al-Si Eutectic Alloy

**Problem:** Predict the final microstructure when an Al-10wt%Si alloy is equilibrium cooled from 700 to room temperature.

**Given information:**

  * Eutectic temperature: 577
  * Eutectic composition: 12.6 wt% Si
  * Si solubility limit in ± phase at room temperature: 0.1 wt% Si
  * Si solubility limit in ± phase at 577: 1.65 wt% Si

**Find:**

  1. Phase state just above 577
  2. Phase state and fraction of each phase just below 577
  3. Final microstructure at room temperature

View solution

**Solution:**

**1\. Just above 577:** L + ± (two-phase region)

**2\. Just below 577 (immediately after eutectic reaction):**

  * Primary ± phase (crystallized above 577) + eutectic microstructure (± + Si lamellar structure)
  * Lever rule: \\( f_{\text{primary}\alpha} = \frac{12.6 - 10}{12.6 - 1.65} = 0.237 \\) (23.7%)
  * Eutectic microstructure: 76.3%

**3\. Final microstructure at room temperature:**

  * Primary ± phase (Si: decreased to 0.1 wt%) + eutectic ± (Si: 0.1 wt%) + eutectic Si (no change) + precipitated Si (precipitated from ± phase)
  * During cooling, Si precipitates from ± phase (1.65 ’ 0.1 wt%), dispersing fine Si particles

#### Exercise 3: Design of Age Hardening Treatment

**Problem:** Design optimal heat treatment conditions for age hardening an Al-4.5wt%Cu alloy.

**Given information:**

  * Eutectic temperature: 548
  * Cu solubility limit in ± phase at 548: 5.65 wt% Cu
  * Cu solubility limit in ± phase at room temperature: 0.1 wt% Cu

**Find:**

  1. Temperature range for solution treatment
  2. Recommended temperature range for aging treatment
  3. Theoretical amount of precipitable Cu

View solution

**Solution:**

**1\. Solution treatment temperature:**

  * Above temperature where 4.5 wt% Cu completely dissolves (approximately above 500)
  * Recommended: 520-540 (set at least 10 below eutectic temperature)
  * Reason: Risk of partial melting above eutectic temperature

**2\. Aging treatment temperature:**

  * Recommended: 150-200 (typically 190)
  * Reason: Maintain supersaturation while achieving appropriate diffusion rate for ¸' precipitation
  * Low temperature (< 150): Aging time becomes long
  * High temperature (> 200): ¸' grows to ¸ causing strength decrease (over-aging)

**3\. Theoretical precipitation amount:**

  * After solution treatment: 4.5 wt% Cu in supersaturated state
  * Equilibrium solubility at 190: approximately 0.5 wt% Cu
  * Precipitable amount: 4.5 - 0.5 = 4.0 wt% Cu
  * This amount precipitates as ¸' (Al‚Cu metastable phase) to improve strength

#### Exercise 4: Tracing Cooling Path from Phase Diagram

**Problem:** Describe the phase transformation sequence when a Pb-30wt%Sn alloy is equilibrium cooled from 300 to 100.

**Given information:**

  * Pb melting point: 327
  * Eutectic temperature: 183
  * Eutectic composition: 61.9 wt% Sn
  * Sn solubility limit in ± phase (Pb-rich) at 183: 19.2 wt% Sn

**Find:**

  1. Phase state in each temperature range
  2. Major phase transformation reactions
  3. Final microstructure composition

View solution

**Solution:**

**Phase transformation sequence:**

  1. **300:** Complete liquid phase (L)
  2. **Approximately 270 (liquidus):** L ’ L + ± (crystallization of ± phase begins)
  3. **270 - 183:** L + ± (two-phase region) 
     * ± phase increases, liquid phase decreases with cooling
     * ± phase composition: Sn gradually increases
     * Liquid phase composition: approaches eutectic composition (61.9 wt% Sn)
  4. **183 (eutectic temperature):** L ’ ± + ² (eutectic reaction) 
     * Residual liquid transforms to eutectic microstructure
     * Lever rule: \\( f_{\text{primary}\alpha} = \frac{61.9 - 30}{61.9 - 19.2} = 0.747 \\) (74.7%)
     * Eutectic microstructure: 25.3%
  5. **183 - 100:** ± + ² (two-phase region) 
     * Due to temperature dependence of solubility limit, Sn precipitates from ± phase to ² phase
     * Fine ² particles disperse in ± phase

**Final microstructure (100):**

  * Primary ± phase (coarse): approximately 75%
  * Eutectic microstructure (± + ² lamellar structure): approximately 25%
  * Precipitated ² phase (fine, dispersed in ± phase): small amount

## Summary

In this chapter, we learned in detail how to read and analyze binary phase diagrams using practical alloy systems as examples.

#### Review of Key Points

  1. **Complete solid solution system (Cu-Ni)** : Continuous solidification between liquidus and solidus. Calculate phase fractions using lever rule.
  2. **Eutectic systems (Al-Si, Pb-Sn)** : Two solid phases crystallize simultaneously from liquid at eutectic point. Minimum melting point at eutectic composition. Clear arrest in cooling curve.
  3. **Peritectic system (Pt-Ag)** : Peritectic reaction L + ± ’ ². Diffusion-controlled requiring time to reach equilibrium.
  4. **Intermetallic compound system (Al-Cu)** : Compound formation at specific composition. Strengthening through age hardening treatment (Duralumin).
  5. **Temperature dependence of solubility limit** : Solubility limit increases with temperature. Principle of age hardening.
  6. **Composition-microstructure-property mapping** : Predict microstructure from phase diagram and correlate with mechanical properties.

#### =¡ Next Steps

In the next chapter, we will learn about **ternary phase diagrams and complex phase transformations**. We will understand phase equilibria in more practical alloy systems such as Fe-C-Cr system (stainless steel) and Al-Cu-Mg system (high-strength aluminum alloys). We will also master advanced phase diagram reading methods for non-equilibrium states including metastable phase diagrams, continuous cooling transformation (CCT) diagrams, and TTT diagrams.

#### Learning Verification

Check if you can answer the following questions:

  * Can you calculate the liquid fraction at 1250 for a 60 wt% Ni alloy in Cu-Ni system using lever rule?
  * Can you explain the differences in microstructure between hypoeutectic, eutectic, and hypereutectic in Al-Si system?
  * Can you explain why arrest at eutectic temperature is prominent in cooling curve of Pb-Sn eutectic alloy?
  * Can you explain differences between peritectic and eutectic reactions using chemical formulas and phase diagrams?
  * Can you explain the mechanism of age hardening in Al-Cu system Duralumin with correlation to phase diagram?
  * Can you explain why solubility limit temperature dependence follows Arrhenius type?
  * Can you propose composition selection (hypoeutectic vs hypereutectic) for practical Al-Si alloys according to application?

[� Chapter 3: Phase Equilibria and Phase Diagram Fundamentals](<chapter-3.html>) [Chapter 5: Ternary Phase Diagrams and Complex Phase Transformations ’](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
