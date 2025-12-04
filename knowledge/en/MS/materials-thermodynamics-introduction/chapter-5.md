---
title: "Chapter 5: Ternary Phase Diagrams and CALPHAD Method"
chapter_title: "Chapter 5: Ternary Phase Diagrams and CALPHAD Method"
subtitle: Ternary Phase Diagrams of Fe-Cr-Ni and Al-Cu-Mg Systems and Principles of Phase Diagram Calculation Using CALPHAD Method
---

This chapter covers Ternary Phase Diagrams and CALPHAD Method. You will learn essential concepts and techniques.

## Learning Objectives

In this chapter, you will learn how to read **ternary phase diagrams** , which are essential for practical materials, and the principles of the **CALPHAD method (CALculation of PHAse Diagrams)** , which forms the foundation of computational materials science. Although ternary systems are more complex than binary systems, many industrial materials such as stainless steels (Fe-Cr-Ni) and high-strength aluminum alloys (Al-Cu-Mg) are ternary or higher-order multicomponent systems.

#### Skills You Will Acquire in This Chapter

  * Composition representation using Gibbs triangle
  * Reading isothermal sections
  * Analysis of liquidus projections
  * Creation and interpretation of vertical sections (pseudo-binary sections)
  * Determination of ternary eutectic and peritectic points
  * Principles of CALPHAD method and structure of thermodynamic databases
  * Multicomponent system modeling using Redlich-Kister equations
  * Practice of phase diagram calculation workflow

#### 💡 Importance of Ternary Phase Diagrams and CALPHAD Method

Ternary phase diagrams represent the equilibrium state of three-component alloys in three-dimensional space (two composition axes + temperature axis). However, since 3D diagrams are difficult to read, they are typically visualized as 2D sections such as isothermal sections, liquidus projections, and vertical sections. The CALPHAD method is a technique for calculating phase diagrams using thermodynamic databases and is indispensable for predicting phase diagrams in regions where experiments are difficult and for designing new alloys.

## 1\. Composition Representation Using Gibbs Triangle

The composition of a ternary system A-B-C is represented by the **Gibbs triangle**. Each vertex of the equilateral triangle corresponds to a pure component (A, B, C), and each edge represents a binary system (A-B, B-C, C-A).

### 1.1 Principles of Triangular Coordinate System

  * **Vertices** : A (100% A), B (100% B), C (100% C)
  * **Edges** : AB edge (C = 0%), BC edge (A = 0%), CA edge (B = 0%)
  * **Interior points** : Ternary alloys (A + B + C = 100%)
  * **Iso-composition lines** : Lines of equal composition for each component drawn parallel to the edges

#### How to Read Composition

The composition of point P (\\(x_A, x_B, x_C\\)) is read by the following procedure:

  1. The length of the perpendicular from point P to edge BC is proportional to \\(x_A\\)
  2. The length of the perpendicular from point P to edge CA is proportional to \\(x_B\\)
  3. The length of the perpendicular from point P to edge AB is proportional to \\(x_C\\)
  4. \\(x_A + x_B + x_C = 1\\)(or 100%)

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 1: Composition Representation in Gibbs Triangle</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    def ternary_to_cartesian(a, b, c):
        """Convert ternary coordinates to Cartesian coordinates"""
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return x, y
    
    # Draw Gibbs triangle
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Triangle vertices (A, B, C)
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Vertex labels
    ax.text(-0.05, -0.05, 'A (Fe)', fontsize=14, fontweight='bold')
    ax.text(1.05, -0.05, 'B (Cr)', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'C (Ni)', fontsize=14, fontweight='bold', ha='center')
    
    # Draw iso-composition lines (grid)
    for i in range(1, 10):
        t = i / 10
        # Iso-composition lines for component A (parallel to BC edge)
        x1, y1 = ternary_to_cartesian(t, 1-t, 0)
        x2, y2 = ternary_to_cartesian(t, 0, 1-t)
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)
    
        # Iso-composition lines for component B (parallel to CA edge)
        x1, y1 = ternary_to_cartesian(1-t, t, 0)
        x2, y2 = ternary_to_cartesian(0, t, 1-t)
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)
    
        # Iso-composition lines for component C (parallel to AB edge)
        x1, y1 = ternary_to_cartesian(1-t, 0, t)
        x2, y2 = ternary_to_cartesian(0, 1-t, t)
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)
    
    # Plot sample composition point (Fe-18Cr-8Ni: SUS304 stainless steel)
    a_sample, b_sample, c_sample = 0.74, 0.18, 0.08  # Mole fraction
    x_sample, y_sample = ternary_to_cartesian(a_sample, b_sample, c_sample)
    ax.plot(x_sample, y_sample, 'ro', markersize=10, label='SUS304 (Fe-18Cr-8Ni)')
    ax.text(x_sample + 0.03, y_sample, 'SUS304', fontsize=11, color='red')
    
    # Other important composition points
    compositions = {
        'SUS316': (0.68, 0.17, 0.12),  # Fe-17Cr-12Ni
        'SUS430': (0.83, 0.17, 0.00),  # Fe-17Cr (Ferritic)
    }
    
    for name, (a, b, c) in compositions.items():
        x, y = ternary_to_cartesian(a, b, c)
        ax.plot(x, y, 'bs', markersize=8)
        ax.text(x + 0.03, y, name, fontsize=10, color='blue')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right')
    ax.set_title('Gibbs Triangle: Composition Representation of Fe-Cr-Ni Ternary System', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gibbs_triangle.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 Composition Confirmation:")
    print(f"SUS304: Fe={a_sample*100:.1f}%, Cr={b_sample*100:.1f}%, Ni={c_sample*100:.1f}%")
    print(f"Total: {(a_sample + b_sample + c_sample)*100:.1f}%")

#### 💡 Practical Example: Composition Representation of Stainless Steels

Stainless steels are representative examples of Fe-Cr-Ni ternary systems. SUS304 (Fe-18Cr-8Ni) is austenitic stainless steel, and SUS430 (Fe-17Cr) is ferritic stainless steel. By confirming which phase region of the phase diagram these compositions are in on the Gibbs triangle, the crystal structure at room temperature (austenite FCC or ferrite BCC) can be predicted.

## 2\. Isothermal Sections

An **isothermal section** is a cross-sectional diagram showing the phase equilibrium of a ternary system at a specific temperature. Each phase region and phase boundary is drawn on the Gibbs triangle.

### 2.1 How to Read Isothermal Sections

  * **Single-phase regions** : Only one phase is stable (α, β, γ, L (liquid phase), etc.)
  * **Two-phase regions** : Two phases coexist (α+β, L+α, etc.)
  * **Three-phase regions** : Three phases coexist (α+β+γ, etc., drawn as triangular regions)
  * **Tie-line** : A straight line connecting the compositions of two phases in equilibrium within a two-phase region
  * **Tie-triangle** : A triangle connecting the compositions of three phases in equilibrium within a three-phase region

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 2: Creating Isothermal Sections</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import LineCollection
    
    def ternary_to_cartesian(a, b, c):
        """Convert ternary coordinates to Cartesian coordinates"""
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return x, y
    
    # Simplified isothermal section of Fe-Cr-Ni system at 1200°C (schematic)
    fig, ax = plt.subplots(figsize=(11, 10))
    
    # Triangle vertices
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Vertex labels
    ax.text(-0.05, -0.05, 'Fe', fontsize=14, fontweight='bold')
    ax.text(1.05, -0.05, 'Cr', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Ni', fontsize=14, fontweight='bold', ha='center')
    
    # Definition of phase regions (simplified)
    # Liquid phase region (L)
    liquid_region = np.array([
        ternary_to_cartesian(0.2, 0.3, 0.5),
        ternary_to_cartesian(0.1, 0.5, 0.4),
        ternary_to_cartesian(0.15, 0.6, 0.25),
        ternary_to_cartesian(0.25, 0.45, 0.3),
    ])
    liquid_patch = Polygon(liquid_region, alpha=0.3, facecolor='lightblue',
                           edgecolor='blue', linewidth=1.5, label='Liquid (L)')
    ax.add_patch(liquid_patch)
    
    # Austenite phase region (γ-FCC)
    austenite_region = np.array([
        ternary_to_cartesian(0.7, 0.1, 0.2),
        ternary_to_cartesian(0.5, 0.15, 0.35),
        ternary_to_cartesian(0.6, 0.05, 0.35),
        ternary_to_cartesian(0.8, 0.05, 0.15),
    ])
    austenite_patch = Polygon(austenite_region, alpha=0.3, facecolor='lightgreen',
                              edgecolor='green', linewidth=1.5, label='γ (FCC)')
    ax.add_patch(austenite_patch)
    
    # Ferrite phase region (α-BCC)
    ferrite_region = np.array([
        ternary_to_cartesian(0.9, 0.1, 0.0),
        ternary_to_cartesian(0.8, 0.2, 0.0),
        ternary_to_cartesian(0.7, 0.25, 0.05),
        ternary_to_cartesian(0.85, 0.12, 0.03),
    ])
    ferrite_patch = Polygon(ferrite_region, alpha=0.3, facecolor='lightyellow',
                            edgecolor='orange', linewidth=1.5, label='α (BCC)')
    ax.add_patch(ferrite_patch)
    
    # L + γ two-phase region
    L_gamma_region = np.array([
        ternary_to_cartesian(0.25, 0.45, 0.3),
        ternary_to_cartesian(0.35, 0.3, 0.35),
        ternary_to_cartesian(0.5, 0.15, 0.35),
        ternary_to_cartesian(0.2, 0.3, 0.5),
    ])
    L_gamma_patch = Polygon(L_gamma_region, alpha=0.2, facecolor='cyan',
                            edgecolor='blue', linestyle='--', linewidth=1, label='L + γ')
    ax.add_patch(L_gamma_patch)
    
    # Examples of tie-lines
    tie_lines = [
        [ternary_to_cartesian(0.3, 0.35, 0.35), ternary_to_cartesian(0.45, 0.2, 0.35)],
        [ternary_to_cartesian(0.25, 0.4, 0.35), ternary_to_cartesian(0.48, 0.18, 0.34)],
    ]
    
    for tie_line in tie_lines:
        xs, ys = zip(*tie_line)
        ax.plot(xs, ys, 'k--', linewidth=1, alpha=0.6)
    
    # Phase labels
    ax.text(*ternary_to_cartesian(0.15, 0.45, 0.4), 'L', fontsize=12, fontweight='bold', ha='center')
    ax.text(*ternary_to_cartesian(0.65, 0.1, 0.25), 'γ', fontsize=12, fontweight='bold', ha='center')
    ax.text(*ternary_to_cartesian(0.82, 0.15, 0.03), 'α', fontsize=12, fontweight='bold', ha='center')
    ax.text(*ternary_to_cartesian(0.35, 0.3, 0.35), 'L+γ', fontsize=10, style='italic', ha='center')
    
    # Representative composition points
    compositions = {
        'SUS304': (0.74, 0.18, 0.08),
        'SUS316': (0.68, 0.17, 0.15),
    }
    
    for name, (a, b, c) in compositions.items():
        x, y = ternary_to_cartesian(a, b, c)
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x + 0.03, y, name, fontsize=10, color='red')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Isothermal Section of Fe-Cr-Ni Ternary System (1200°C, schematic)', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('isothermal_section.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 Phase State at 1200°C:")
    print("• SUS304 (Fe-18Cr-8Ni): γ phase (austenite) region")
    print("• High Cr region: α phase (ferrite) is stable")
    print("• L+γ two-phase region: Liquid phase and austenite coexist")

#### 💡 Tie-lines and Lever Rule

At composition point P within a two-phase region, two phases (α and β) on the tie-line coexist. The fraction of each phase can be calculated using the lever rule, as in binary systems:

\\[ f_\alpha = \frac{|\text{P-β}|}{|\text{α-β}|}, \quad f_\beta = \frac{|\text{P-α}|}{|\text{α-β}|} \\]

However, the distance is the distance in composition space on the Gibbs triangle.

## 3\. Liquidus Projection

A **liquidus projection** is a diagram showing the temperature at which the liquid phase begins to solidify (liquidus temperature) as contour lines on the Gibbs triangle. It is useful for understanding cooling paths and solidification processes.

### 3.1 Components of Liquidus Projection

  * **Liquidus isotherms** : Curves connecting points with the same liquidus temperature
  * **Primary crystallization lines** : Boundary lines distinguishing the phase that crystallizes first from the liquid
  * **Eutectic valley** : Valley line where the liquidus temperature reaches a minimum
  * **Ternary eutectic point** : Invariant point where the liquid decomposes into three solid phases

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 3: Visualization of Liquidus Projection</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.interpolate import griddata
    
    def ternary_to_cartesian(a, b, c):
        """Convert ternary coordinates to Cartesian coordinates"""
        total = a + b + c
        x = 0.5 * (2*b + c) / total
        y = (np.sqrt(3)/2) * c / total
        return x, y
    
    # Simplified model of liquidus temperature (simulating Al-Cu-Mg system)
    fig, ax = plt.subplots(figsize=(11, 10))
    
    # Triangle vertices
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Vertex labels and melting points
    ax.text(-0.08, -0.05, 'Al\n(660℃)', fontsize=13, fontweight='bold', ha='right')
    ax.text(1.08, -0.05, 'Cu\n(1085℃)', fontsize=13, fontweight='bold', ha='left')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Mg\n(650℃)', fontsize=13, fontweight='bold', ha='center')
    
    # Calculate liquidus temperature at grid points (simplified model)
    n_points = 100
    grid_a = []
    grid_b = []
    grid_c = []
    liquidus_temps = []
    
    for i in range(n_points):
        for j in range(n_points - i):
            k = n_points - i - j
            a, b, c = i/n_points, j/n_points, k/n_points
    
            if a + b + c > 0.99 and a + b + c < 1.01:  # Only inside triangle
                # Simplified liquidus temperature model (actually calculated by CALPHAD)
                T_liquidus = (660*a + 1085*b + 650*c) - 200*a*b - 150*b*c - 100*c*a
    
                grid_a.append(a)
                grid_b.append(b)
                grid_c.append(c)
                liquidus_temps.append(T_liquidus)
    
    # Convert to Cartesian coordinates
    grid_x = []
    grid_y = []
    for a, b, c in zip(grid_a, grid_b, grid_c):
        x, y = ternary_to_cartesian(a, b, c)
        grid_x.append(x)
        grid_y.append(y)
    
    # Create grid for contour plotting
    xi = np.linspace(0, 1, 200)
    yi = np.linspace(0, np.sqrt(3)/2, 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # griddata interpolation
    zi = griddata((grid_x, grid_y), liquidus_temps, (xi_grid, yi_grid), method='cubic')
    
    # Plot contour lines
    levels = np.arange(550, 1100, 50)
    contour = ax.contour(xi_grid, yi_grid, zi, levels=levels, colors='black', linewidths=0.5, alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%d℃')
    
    # Fill contours
    contourf = ax.contourf(xi_grid, yi_grid, zi, levels=levels, cmap='coolwarm', alpha=0.5)
    cbar = plt.colorbar(contourf, ax=ax, label='Liquidus Temperature (°C)', pad=0.02)
    
    # Primary crystallization lines (schematic)
    primary_lines = [
        # Boundary between Al and Cu primary crystallization regions
        [ternary_to_cartesian(0.8, 0.2, 0), ternary_to_cartesian(0.5, 0.3, 0.2)],
        # Boundary between Cu and Mg primary crystallization regions
        [ternary_to_cartesian(0.2, 0.8, 0), ternary_to_cartesian(0.3, 0.4, 0.3)],
        # Boundary between Al and Mg primary crystallization regions
        [ternary_to_cartesian(0.7, 0, 0.3), ternary_to_cartesian(0.4, 0.1, 0.5)],
    ]
    
    for line in primary_lines:
        xs, ys = zip(*line)
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
    
    # Ternary eutectic point (schematic)
    eutectic_point = ternary_to_cartesian(0.5, 0.3, 0.2)
    ax.plot(*eutectic_point, 'r*', markersize=15, label='Ternary eutectic point (~520°C)')
    
    # Compositions of practical alloys
    alloys = {
        '2024': (0.935, 0.043, 0.015),  # Al-4.3Cu-1.5Mg
        '7075': (0.90, 0.016, 0.025),   # Al-1.6Cu-2.5Mg-Zn
    }
    
    for name, (a, b, c) in alloys.items():
        x, y = ternary_to_cartesian(a, b, c)
        ax.plot(x, y, 'ko', markersize=8)
        ax.text(x + 0.03, y, name, fontsize=10, color='black', fontweight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Liquidus Projection of Al-Cu-Mg Ternary System (schematic)', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('liquidus_projection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 How to Read Liquidus Projection:")
    print("• Contour lines: Show liquidus temperature (temperature at which solidification begins during cooling)")
    print("• Primary crystallization lines: Boundaries distinguishing the first crystallizing phase")
    print("• Ternary eutectic point: Invariant point where L → α + β + γ reaction occurs")

#### Tracking Cooling Paths

On the liquidus projection, the cooling path of an alloy can be tracked as follows:

  1. Start cooling from composition point P
  2. When the liquidus temperature is reached, primary crystals (α, β, or γ) begin to crystallize
  3. As cooling progresses, the composition of the liquid changes along the primary crystallization line
  4. Descend the primary crystallization line and reach the eutectic valley
  5. Descend the eutectic valley and completely solidify at the ternary eutectic point

## 4\. Vertical Sections (Pseudo-Binary Sections)

A **vertical section** is a temperature-composition diagram along a specific line on the Gibbs triangle (e.g., a line from edge A-B to vertex C). It has a similar appearance to binary phase diagrams.

### 4.1 Applications of Vertical Sections

  * Compositions of practical alloysDetailed analysis of phase transformations in the composition range
  * Investigation of the effect of the third component C when a specific composition ratio (e.g., A:B = 1:1) is fixed
  * Understanding local behavior of ternary systems in a format similar to binary phase diagrams

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: 4.1 Applications of Vertical Sections
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
                <h4>Code Example 4: Vertical Section (Pseudo-Binary Section)</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    # Vertical section of Fe-Cr-Ni system with Cr/Ni = 2:1 ratio fixed (schematic)
    # Horizontal axis: Fe content (100% → 0%), Vertical axis: Temperature
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Fe content (wt%)
    fe_content = np.linspace(100, 0, 100)
    # Since Cr/Ni = 2:1, Cr = 2(100-Fe)/3, Ni = (100-Fe)/3
    
    # Liquidus and solidus (simplified model)
    liquidus = 1536 - 5*fe_content + 0.02*fe_content**2  # Fe side is 1536°C
    solidus = 1450 - 3*fe_content + 0.015*fe_content**2
    
    # γ/α phase boundary (boundary between ferrite and austenite)
    gamma_alpha_boundary = 1400 - 6*fe_content + 0.03*fe_content**2
    
    # Plot
    ax.plot(fe_content, liquidus, 'b-', linewidth=2, label='Liquidus')
    ax.plot(fe_content, solidus, 'r-', linewidth=2, label='Solidus')
    ax.plot(fe_content, gamma_alpha_boundary, 'g--', linewidth=2, label='γ/α phase boundary')
    
    # Phase region labels
    ax.text(50, 1600, 'L (Liquid)', fontsize=12, ha='center', fontweight='bold')
    ax.text(50, 1500, 'L + γ', fontsize=11, ha='center', style='italic')
    ax.text(70, 1350, 'γ (FCC)', fontsize=12, ha='center', fontweight='bold', color='green')
    ax.text(30, 1250, 'α (BCC)', fontsize=12, ha='center', fontweight='bold', color='orange')
    ax.text(50, 1300, 'γ + α', fontsize=10, ha='center', style='italic')
    
    # Fill phase regions
    ax.fill_between(fe_content, liquidus, 1700, alpha=0.2, color='lightblue', label='L')
    ax.fill_between(fe_content, solidus, liquidus, alpha=0.2, color='cyan', label='L+γ')
    ax.fill_between(fe_content, gamma_alpha_boundary, solidus,
                    where=(gamma_alpha_boundary < solidus), alpha=0.2, color='lightgreen', label='γ')
    ax.fill_between(fe_content, 1100, gamma_alpha_boundary, alpha=0.2, color='lightyellow', label='α')
    
    # Composition point of SUS304 (Fe-18Cr-8Ni → Cr/Ni ≈ 2.25:1)
    fe_304 = 74  # wt% Fe
    ax.axvline(fe_304, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(fe_304 + 2, 1150, 'SUS304', fontsize=11, color='red', fontweight='bold', rotation=90)
    
    ax.set_xlabel('Fe content (wt%)', fontsize=13)
    ax.set_ylabel('Temperature (°C)', fontsize=13)
    ax.set_title('Vertical Section of Fe-Cr-Ni Ternary System (Cr/Ni = 2:1 fixed, schematic)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(1100, 1700)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('vertical_section.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 Interpretation of Vertical Section:")
    print("• SUS304 (Fe-18Cr-8Ni): γ phase (austenite) is stable at high temperature")
    print("• As Fe content increases, α phase (ferrite) stabilizes")
    print("• L+γ two-phase region: γ phase crystallizes during solidification")

#### 💡 Application Example of Vertical Sections

In welding of stainless steels, the phase state during solidification is important. By using vertical sections, it is possible to predict the solidification path at a specific Cr/Ni ratio and design compositions that avoid the formation of harmful phases (such as σ phase) that cause weld cracking.

## 5\. Ternary Eutectic Points and Invariant Reactions

In ternary systems, there are invariant reactions such as **ternary eutectic reactions** \\( L \rightarrow \alpha + \beta + \gamma \\). According to Gibbs' phase rule, these reactions occur at specific temperatures and compositions.

### 5.1 Invariant Reactions in Ternary Systems

Reaction Type | Reaction Equation | Characteristics  
---|---|---  
Ternary Eutectic | \\( L \rightarrow \alpha + \beta + \gamma \\) | Liquid decomposes into three solid phases  
Ternary Peritectic | \\( L + \alpha + \beta \rightarrow \gamma \\) | Liquid and two solid phases react to form a new solid phase  
Ternary Monotectic | \\( L_1 \rightarrow L_2 + \alpha + \beta \\) | Liquid separates into two liquids and a solid phase  
Quasi-Peritectic | \\( L + \alpha \rightarrow \beta + \gamma \\) | Liquid and solid phase react to form two solid phases  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 5: Determination of Ternary Eutectic Point</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d import Axes3D
    
    def ternary_to_cartesian(a, b, c):
        """Convert ternary coordinates to Cartesian coordinates"""
        total = a + b + c
        x = 0.5 * (2*b + c) / total
        y = (np.sqrt(3)/2) * c / total
        return x, y
    
    # Ternary Eutecticreaction visualization
    fig = plt.figure(figsize=(14, 6))
    
    # Left: Ternary eutectic point on Gibbs triangle
    ax1 = fig.add_subplot(121)
    
    # Triangle vertices
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(triangle)
    
    ax1.text(-0.05, -0.05, 'A', fontsize=14, fontweight='bold')
    ax1.text(1.05, -0.05, 'B', fontsize=14, fontweight='bold')
    ax1.text(0.5, np.sqrt(3)/2 + 0.05, 'C', fontsize=14, fontweight='bold', ha='center')
    
    # Ternary eutectic point
    eutectic_comp = (0.40, 0.35, 0.25)  # A, B, C
    e_x, e_y = ternary_to_cartesian(*eutectic_comp)
    ax1.plot(e_x, e_y, 'r*', markersize=20, label='Ternary eutectic point E')
    
    # Compositions of three solid phases in equilibrium
    alpha_comp = (0.85, 0.10, 0.05)
    beta_comp = (0.15, 0.75, 0.10)
    gamma_comp = (0.20, 0.15, 0.65)
    
    alpha_x, alpha_y = ternary_to_cartesian(*alpha_comp)
    beta_x, beta_y = ternary_to_cartesian(*beta_comp)
    gamma_x, gamma_y = ternary_to_cartesian(*gamma_comp)
    
    ax1.plot(alpha_x, alpha_y, 'go', markersize=10, label='α phase')
    ax1.plot(beta_x, beta_y, 'bo', markersize=10, label='β phase')
    ax1.plot(gamma_x, gamma_y, 'mo', markersize=10, label='γ phase')
    
    # Tie-triangle
    tie_triangle = Polygon(
        [ternary_to_cartesian(*alpha_comp),
         ternary_to_cartesian(*beta_comp),
         ternary_to_cartesian(*gamma_comp)],
        fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Tie-triangle'
    )
    ax1.add_patch(tie_triangle)
    
    # Composition labels
    ax1.text(alpha_x + 0.05, alpha_y, 'α', fontsize=11, color='green', fontweight='bold')
    ax1.text(beta_x + 0.05, beta_y, 'β', fontsize=11, color='blue', fontweight='bold')
    ax1.text(gamma_x + 0.05, gamma_y, 'γ', fontsize=11, color='purple', fontweight='bold')
    ax1.text(e_x + 0.03, e_y + 0.05, 'E', fontsize=11, color='red', fontweight='bold')
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.0)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('(a) Ternary eutectic point and Tie-triangle', fontsize=13, fontweight='bold')
    
    # Right: Cooling curves
    ax2 = fig.add_subplot(122)
    
    time = np.linspace(0, 100, 500)
    
    # Cooling curve (ternary eutectic composition)
    temp_eutectic = 900 - 5*time
    temp_eutectic[temp_eutectic < 550] = 550  # Arrest at eutectic temperature
    temp_eutectic[time > 60] = 550 - 3*(time[time > 60] - 60)
    
    # Cooling curve (non-eutectic composition)
    temp_noneutectic = 950 - 5*time
    temp_noneutectic[(temp_noneutectic < 600) & (temp_noneutectic > 550)] = \
        600 - 0.5*(time[(temp_noneutectic < 600) & (temp_noneutectic > 550)] - 50)
    temp_noneutectic[temp_noneutectic < 550] = 550
    temp_noneutectic[time > 70] = 550 - 3*(time[time > 70] - 70)
    
    ax2.plot(time, temp_eutectic, 'r-', linewidth=2, label='Eutectic composition (point E)')
    ax2.plot(time, temp_noneutectic, 'b-', linewidth=2, label='Non-eutectic composition')
    
    # Eutectic temperature line
    ax2.axhline(550, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(10, 560, 'Ternary eutectic temperature T_E', fontsize=10, color='gray')
    
    ax2.set_xlabel('Time (arbitrary units)', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_title('(b) Cooling Curves: Arrest at Eutectic Temperature', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(400, 1000)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ternary_eutectic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 Characteristics of ternary eutectic reaction:")
    print(f"• Eutectic composition: A={eutectic_comp[0]*100:.0f}%, B={eutectic_comp[1]*100:.0f}%, C={eutectic_comp[2]*100:.0f}%")
    print("• Reaction: L → α + β + γ")
    print("• Cooling curve: Significant arrest at eutectic temperature (release of latent heat)")
    print("• Tie-triangle: 3triangle connecting equilibrium compositions of three solid phases")

#### Gibbs' Phase Rule and Ternary Systems

Applying Gibbs' phase rule \\( F = C - P + 2 \\) to ternary systems:

  * **Number of components** \\( C = 3 \\)
  * **Single-phase region (P=1)** : \\( F = 4 \\) (temperature, pressure, 2 composition variables)
  * **Two-phase region (P=2)** : \\( F = 3 \\) (1 composition variable is free under isothermal and isobaric conditions)
  * **Three-phase region (P=3)** : \\( F = 2 \\) (zero degrees of freedom under isothermal and isobaric conditions → composition fixed)
  * **Four-phase coexistence (P=4)** : \\( F = 1 \\) (invariant point: temperature and composition fixed)

In ternary eutectic reactions, since liquid L and three solid phases α, β, γ coexist (P=4), they occur only at specific temperatures and compositions.

## 6\. Principles of CALPHAD Method

The **CALPHAD method** (CALculation of PHAse Diagrams) is a technique for calculating phase diagrams using thermodynamic databases. It combines experimental data and theoretical models to predict phase equilibria in complex multicomponent systems.

### 6.1 Basic Concepts of CALPHAD Method

  * **Gibbs free energy minimization** : In equilibrium, the total Gibbs free energy G of the system is minimized
  * **Phase models** : Express Gibbs free energy of each phase (solid, liquid, compound, etc.) as a function of composition and temperature
  * **Thermodynamic database** : Stores model parameters optimized from experimental data
  * **Phase equilibrium calculation** : Calculate G for all phases and determine the combination of phases that gives the minimum

    
    
    ```mermaid
    graph TD
        A[Experimental data] --> B[Thermodynamic Modeling]
        B --> C[Parameter Optimization]
        C --> D[Thermodynamic DatabaseTDB file]
        D --> E[Phase Diagram Calculation EngineThermo-Calc, pycalphad]
        E --> F[Phase Diagram Output]
        E --> G[Thermodynamic Property Calculation]
    
        H[New Material Design] --> E
        I[Process Simulation] --> E
    
        style D fill:#f093fb,stroke:#f5576c,color:#fff
        style E fill:#f093fb,stroke:#f5576c,color:#fff
    ```

#### Advantages of CALPHAD Method

  1. **Extrapolation capability** : Predict phase diagrams in regions without experimental data (high temperature, extreme compositions)
  2. **Extension to multicomponent systems** : Extrapolate from binary system data to ternary, quaternary, and higher-order systems
  3. **Time and cost reduction** : Significantly reduce the number of experimental trials
  4. **Integrated approach** : Can calculate not only phase diagrams but also heat capacity, activity, chemical potential, etc.

### 6.2 Gibbs Free Energy Model for Phases

The Gibbs free energy of a solution phase in binary system A-B is expressed in the following form:

\\[ G_m = x_A {}^0G_A + x_B {}^0G_B + RT(x_A \ln x_A + x_B \ln x_B) + {}^{\text{ex}}G_m \\]

  * \\( {}^0G_A, {}^0G_B \\): Gibbs free energy of pure components A and B (standard state)
  * \\( RT(x_A \ln x_A + x_B \ln x_B) \\): Mixing entropy term for ideal solution
  * \\( {}^{\text{ex}}G_m \\): Excess Gibbs free energy (expresses non-ideality)

The excess Gibbs free energy is approximated by the **Redlich-Kister polynomial** :

\\[ {}^{\text{ex}}G_m = x_A x_B \sum_{i=0}^{n} {}^iL_{A,B} (x_A - x_B)^i \\]

  * \\( {}^iL_{A,B} \\): Interaction parameter (temperature-dependent: \\( L = a + bT + cT\ln T + \cdots \\))
  * \\( i \\): Degree of polynomial (typically 0 to 2)

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 6: Ternary System Modeling Using Redlich-Kister Equations</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    # Implementation of Redlich-Kister polynomial
    def redlich_kister_binary(x_A, L0, L1=0, L2=0):
        """Redlich-Kister excess Gibbs free energy for binary system A-B"""
        x_B = 1 - x_A
        ex_G = x_A * x_B * (L0 + L1*(x_A - x_B) + L2*(x_A - x_B)**2)
        return ex_G
    
    def gibbs_binary(x_A, G0_A, G0_B, T, L0, L1=0, L2=0):
        """Gibbs free energy of binary system"""
        R = 8.314  # J/(mol·K)
        x_B = 1 - x_A
    
        # Avoid division by zero
        x_A = np.clip(x_A, 1e-10, 1-1e-10)
        x_B = np.clip(x_B, 1e-10, 1-1e-10)
    
        # Ideal mixing term
        G_ideal = x_A * G0_A + x_B * G0_B
        G_mix = R * T * (x_A * np.log(x_A) + x_B * np.log(x_B))
    
        # Excess term
        G_ex = redlich_kister_binary(x_A, L0, L1, L2)
    
        return G_ideal + G_mix + G_ex
    
    # Gibbs Free Energy Curves for Binary Systems
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Ideal Solution vs Non-ideal Solution
    ax1 = axes[0]
    x = np.linspace(0.001, 0.999, 200)
    T = 1000  # K
    R = 8.314
    
    G0_A = 0      # Pure A
    G0_B = 5000   # Pure B (5 kJ/mol higher)
    
    # Ideal solution
    G_ideal = x * G0_A + (1-x) * G0_B + R*T*(x*np.log(x) + (1-x)*np.log(1-x))
    
    # Non-ideal solution (Positive deviation → phase separation tendency）
    L0_positive = 15000  # J/mol
    G_positive = gibbs_binary(x, G0_A, G0_B, T, L0_positive)
    
    # Non-ideal solution (Negative deviation → compound formation tendency）
    L0_negative = -10000  # J/mol
    G_negative = gibbs_binary(x, G0_A, G0_B, T, L0_negative)
    
    ax1.plot(x*100, G_ideal/1000, 'k-', linewidth=2, label='Ideal solution (L₀=0)')
    ax1.plot(x*100, G_positive/1000, 'r-', linewidth=2, label=f'Positive deviation (L₀={L0_positive/1000:.0f} kJ/mol)')
    ax1.plot(x*100, G_negative/1000, 'b-', linewidth=2, label=f'Negative deviation (L₀={L0_negative/1000:.0f} kJ/mol)')
    
    ax1.set_xlabel('B content (at%)', fontsize=12)
    ax1.set_ylabel('Gibbs Free Energy (kJ/mol)', fontsize=12)
    ax1.set_title('(a) Redlich-Kisterequation: Deviation from Ideal solution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # Right: Temperature Dependence
    ax2 = axes[1]
    temperatures = [800, 1000, 1200, 1400]  # K
    L0 = 15000  # J/mol
    
    for T in temperatures:
        G = gibbs_binary(x, G0_A, G0_B, T, L0)
        ax2.plot(x*100, G/1000, linewidth=2, label=f'T = {T} K')
    
    ax2.set_xlabel('B content (at%)', fontsize=12)
    ax2.set_ylabel('Gibbs Free Energy (kJ/mol)', fontsize=12)
    ax2.set_title('(b) Temperature Dependence of Gibbs Free Energy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig('redlich_kister.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 Interpretation of Redlich-Kister Equation:")
    print("• L₀ > 0: Positive deviation → weak A-B interaction → phase separation tendency")
    print("• L₀ < 0: Negative deviation → strong A-B interaction → compound formation tendency")
    print("• Temperature increase: contribution of mixing entropy term (RT ln x) increases → mixing becomes favorable")

#### 💡 Model Extension to Ternary Systems

In ternary system A-B-C, in addition to binary parameters (A-B, B-C, C-A), a **ternary interaction parameter** is introduced:

\\[ {}^{\text{ex}}G_m^{\text{ABC}} = {}^{\text{ex}}G_m^{\text{AB}} + {}^{\text{ex}}G_m^{\text{BC}} + {}^{\text{ex}}G_m^{\text{CA}} + x_A x_B x_C L_{\text{ABC}} \\]

Here, \\( L_{\text{ABC}} \\) is the ternary interaction parameter. In many cases, extrapolation from binary data provides sufficient accuracy, so \\( L_{\text{ABC}} = 0 \\) is approximated.

## 7\. CALPHAD Method Workflow

Phase diagram calculation using the CALPHAD method is performed in the following steps:

#### 5 Steps of CALPHAD Workflow

  1. **Literature review** : Collect existing experimental data (phase diagrams, heat capacity, activity, etc.)
  2. **Model selection** : Select appropriate thermodynamic models for each phase (liquid, solid solution, compound)
  3. **Parameter Optimization** : Experimental dataOptimize model parameters to best fit
  4. **Database construction** : Store optimized parameters in TDB file (Thermo-Calc DataBase)
  5. **Phase diagram calculation and verification** : Calculate phase diagrams using the database and verify by comparison with experimental data

### 7.1 Structure of Thermodynamic Databases

In the CALPHAD method, thermodynamic data are stored in **TDB files** (Thermo-Calc DataBase format). Representative databases:

  * **SGTE (Scientific Group Thermodata Europe)** : Pure substance database
  * **SSUB (SGTE Substance Database)** : Thermodynamic data for over 1500 pure substances
  * **TCFE (Thermo-Calc Steel and Fe-alloys Database)** : Steel and iron alloy database
  * **TCAL (Thermo-Calc Al-alloys Database)** : Aluminum alloy database

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: In the CALPHAD method, thermodynamic data are stored inTDB f
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
                <h4>Code Example 7: Simulation of CALPHAD Workflow (Simplified Version)</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    # Simplified CALPHAD workflow: Parameter optimization demonstration
    
    # Step 1: Experimental data (simulated)
    # Cu-NiExperimental data for liquidus and solidus of system
    exp_compositions = np.array([0, 20, 40, 60, 80, 100])  # wt% Ni
    exp_liquidus = np.array([1085, 1160, 1260, 1350, 1410, 1455])  # ℃
    exp_solidus = np.array([1085, 1130, 1220, 1310, 1390, 1455])   # ℃
    
    # Step 2: Thermodynamic model (simplified: Redlich-Kister model)
    def calculate_phase_diagram(L0_liquid, L0_solid):
        """
        Calculation of binary phase diagram (extremely simplified model)
        Actual CALPHAD requires more complex calculations
        """
        compositions = np.linspace(0, 100, 100)
    
        # Simplified model for liquidus and solidus
        liquidus = 1085 + (1455-1085)*compositions/100 + L0_liquid*compositions*(100-compositions)/10000
        solidus = 1085 + (1455-1085)*compositions/100 + L0_solid*compositions*(100-compositions)/10000
    
        return compositions, liquidus, solidus
    
    # Step 3: Parameter Optimization
    def objective_function(params):
        """Objective function: Minimize error with experimental data"""
        L0_liquid, L0_solid = params
    
        comp_calc, liq_calc, sol_calc = calculate_phase_diagram(L0_liquid, L0_solid)
    
        # Experimental dataInterpolate calculated values at points
        liq_interp = np.interp(exp_compositions, comp_calc, liq_calc)
        sol_interp = np.interp(exp_compositions, comp_calc, sol_calc)
    
        # Sum of squared errors
        error = np.sum((liq_interp - exp_liquidus)**2) + np.sum((sol_interp - exp_solidus)**2)
    
        return error
    
    # Initial estimate
    initial_params = [0.0, 0.0]
    
    # Execute optimization
    print("🔧 Optimizing parameters...")
    result = minimize(objective_function, initial_params, method='Nelder-Mead')
    optimal_L0_liquid, optimal_L0_solid = result.x
    
    print(f"✅ Optimization complete!")
    print(f"   Optimal parameters: L0_liquid = {optimal_L0_liquid:.2f}, L0_solid = {optimal_L0_solid:.2f}")
    print(f"   Error: {result.fun:.2f} K²")
    
    # Step 4: Phase diagram calculation and visualization
    comp_calc, liq_calc, sol_calc = calculate_phase_diagram(optimal_L0_liquid, optimal_L0_solid)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calculated phase diagram
    ax.plot(comp_calc, liq_calc, 'b-', linewidth=2, label='Liquidus (calculated)')
    ax.plot(comp_calc, sol_calc, 'r-', linewidth=2, label='Solidus (calculated)')
    
    # Experimental data
    ax.plot(exp_compositions, exp_liquidus, 'bo', markersize=8, label='Liquidus (experimental)')
    ax.plot(exp_compositions, exp_solidus, 'ro', markersize=8, label='Solidus (experimental)')
    
    # Fill phase regions
    ax.fill_between(comp_calc, liq_calc, 1500, alpha=0.2, color='lightblue', label='L (Liquid)')
    ax.fill_between(comp_calc, sol_calc, liq_calc, alpha=0.2, color='lightgreen', label='L + α (Two-phase)')
    ax.fill_between(comp_calc, 1050, sol_calc, alpha=0.2, color='lightyellow', label='α (Solid)')
    
    ax.set_xlabel('Ni content (wt%)', fontsize=13)
    ax.set_ylabel('Temperature (°C)', fontsize=13)
    ax.set_title('CALPHAD Workflow: Cu-Ni System Phase Diagram Calculation (Simplified Model)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(1050, 1500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('calphad_workflow.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 5: Database output (TDB file format image)
    print("\n📄 TDB file output (simplified version):")
    print("=" * 50)
    print("$ Cu-Ni system optimized parameters")
    print("$ Database: DEMO_CU_NI")
    print("$ Date: 2025-10-27")
    print("$")
    print("ELEMENT Cu  FCC    63.546   5004.0  33.15  !")
    print("ELEMENT Ni  FCC    58.69    6536.0  29.87  !")
    print("$")
    print("PHASE LIQUID % 1 1.0 !")
    print(f"PARAMETER G(LIQUID,Cu;0)  298.15  +12964.7-9.511*T !")
    print(f"PARAMETER G(LIQUID,Ni;0)  298.15  +16414.7-9.397*T !")
    print(f"PARAMETER L(LIQUID,Cu,Ni;0)  298.15  {optimal_L0_liquid:.2f} !")
    print("$")
    print("PHASE FCC_A1 % 1 1.0 !")
    print(f"PARAMETER G(FCC_A1,Cu;0)  298.15  -7770.5+130.485*T !")
    print(f"PARAMETER G(FCC_A1,Ni;0)  298.15  -5179.2+117.854*T !")
    print(f"PARAMETER L(FCC_A1,Cu,Ni;0)  298.15  {optimal_L0_solid:.2f} !")
    print("=" * 50)

#### 💡 Actual CALPHAD Calculation Software

The above is an extremely simplified example for educational purposes. In actual CALPHAD calculations, the following software is used:

  * **Thermo-Calc** : Commercial, most widely used (industry and academia)
  * **FactSage** : Commercial, strong in high-temperature processes
  * **Pandat** : Commercial, specialized in phase transformation simulation
  * **pycalphad** : Open source (Python), will be studied in detail in the next chapter

In the next chapter, we will perform practical phase diagram calculations using pycalphad.

## Exercises

#### Exercise 1: Reading Composition on Gibbs Triangle

**Problem:** In the Fe-Cr-Ni ternary system, plot point P (Fe: 70%, Cr: 20%, Ni: 10%) on the Gibbs triangle and calculate the composition distance to SUS304 (Fe: 74%, Cr: 18%, Ni: 8%).

Hint

Convert ternary coordinates to Cartesian coordinates, then calculate the Euclidean distance. The distance in composition space serves as an indicator of similarity in actual material properties.

Sample Answer
    
    
    # Composition distance calculation on Gibbs triangle
    def ternary_to_cartesian(a, b, c):
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return x, y
    
    # Composition 1: Point P
    P_comp = (0.70, 0.20, 0.10)
    P_x, P_y = ternary_to_cartesian(*P_comp)
    
    # Composition 2: SUS304
    SUS304_comp = (0.74, 0.18, 0.08)
    SUS304_x, SUS304_y = ternary_to_cartesian(*SUS304_comp)
    
    # Euclidean distance
    distance = np.sqrt((P_x - SUS304_x)**2 + (P_y - SUS304_y)**2)
    
    print(f"Point P: ({P_x:.4f}, {P_y:.4f})")
    print(f"SUS304: ({SUS304_x:.4f}, {SUS304_y:.4f})")
    print(f"Composition distance: {distance:.4f}(normalized distance on triangle)")
    print(f"Composition difference: ΔFe={abs(0.70-0.74)*100:.1f}%, ΔCr={abs(0.20-0.18)*100:.1f}%, ΔNi={abs(0.10-0.08)*100:.1f}%")

#### Exercise 2: Tie-line in Isothermal Section

**Problem:** In the isothermal section of the Fe-Cr-Ni system at 1200°C, assume an alloy with composition Fe: 60%, Cr: 25%, Ni: 15% is in the L+γ two-phase region. When the liquid composition is Fe: 50%, Cr: 30%, Ni: 20%, and the γ phase composition is Fe: 65%, Cr: 22%, Ni: 13%, find the fraction of each phase.

Hint

Apply the lever rule in ternary coordinates. The inverse ratio of distances from the alloy composition point to each phase gives the phase fraction.

Sample Answer
    
    
    # Lever rule in ternary coordinates
    alloy_comp = np.array([0.60, 0.25, 0.15])  # Fe, Cr, Ni
    L_comp = np.array([0.50, 0.30, 0.20])
    gamma_comp = np.array([0.65, 0.22, 0.13])
    
    # Vector distance calculation
    dist_alloy_to_L = np.linalg.norm(alloy_comp - L_comp)
    dist_alloy_to_gamma = np.linalg.norm(alloy_comp - gamma_comp)
    dist_L_to_gamma = np.linalg.norm(L_comp - gamma_comp)
    
    # Lever rule
    f_gamma = dist_alloy_to_L / dist_L_to_gamma
    f_L = dist_alloy_to_gamma / dist_L_to_gamma
    
    print(f"Fraction of liquid (L): {f_L*100:.1f}%")
    print(f"γ phasefraction: {f_gamma*100:.1f}%")
    print(f"Total: {(f_L + f_gamma)*100:.1f}%")
    
    # Verification: Mass conservation
    reconstructed_comp = f_L * L_comp + f_gamma * gamma_comp
    print(f"\nVerification (Mass conservation law):")
    print(f"Original composition: Fe={alloy_comp[0]*100:.1f}%, Cr={alloy_comp[1]*100:.1f}%, Ni={alloy_comp[2]*100:.1f}%")
    print(f"Reconstructed composition: Fe={reconstructed_comp[0]*100:.1f}%, Cr={reconstructed_comp[1]*100:.1f}%, Ni={reconstructed_comp[2]*100:.1f}%")

#### Exercise 3: Fitting Redlich-Kister Equation

**Problem:** Activity data for a binary system A-B (at 1000 K, excess Gibbs free energies at x_B = 0.2, 0.4, 0.6, 0.8 are 5000, 8000, 8000, 5000 J/mol) have been obtained. Determine the Redlich-Kister parameters L0, L1 by the least squares method.

Hint

Redlich-Kisterequation \\( {}^{\text{ex}}G_m = x_A x_B (L_0 + L_1(x_A - x_B)) \\) to fit the experimental data. scipy.optimize.curve_fit is convenient.

Sample Answer
    
    
    from scipy.optimize import curve_fit
    
    # Experimental data
    x_B_data = np.array([0.2, 0.4, 0.6, 0.8])
    ex_G_data = np.array([5000, 8000, 8000, 5000])  # J/mol
    
    # Redlich-Kister model (L0, L1)
    def redlich_kister_model(x_B, L0, L1):
        x_A = 1 - x_B
        return x_A * x_B * (L0 + L1 * (x_A - x_B))
    
    # Fitting
    popt, pcov = curve_fit(redlich_kister_model, x_B_data, ex_G_data)
    L0_fit, L1_fit = popt
    
    print(f"Optimized parameters:")
    print(f"  L0 = {L0_fit:.1f} J/mol")
    print(f"  L1 = {L1_fit:.1f} J/mol")
    
    # Visualization
    x_B_fine = np.linspace(0.01, 0.99, 100)
    ex_G_fit = redlich_kister_model(x_B_fine, L0_fit, L1_fit)
    
    plt.figure(figsize=(9, 6))
    plt.plot(x_B_data, ex_G_data, 'ro', markersize=10, label='Experimental data')
    plt.plot(x_B_fine, ex_G_fit, 'b-', linewidth=2, label=f'Fit (L₀={L0_fit:.0f}, L₁={L1_fit:.0f})')
    plt.xlabel('x_B', fontsize=13)
    plt.ylabel('Excess Gibbs Free Energy (J/mol)', fontsize=13)
    plt.title('Optimization of Redlich-Kister Parameters', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#### Exercise 4: Application of CALPHAD Method

**Problem:** Consider the composition of SUS316L (Fe-17Cr-12Ni-2.5Mo) in the Fe-Cr-Ni ternary system. Discuss how Mo addition changes the solidification temperature from the CALPHAD perspective. (Qualitative discussion is acceptable)

Hint

Mo is a high melting point element (2623°C) and acts as a solid solution strengthening element in steel. Consider the effect on liquid phase stability.

Sample Answer

**Qualitative Discussion:**

  * **Effect of Mo addition** : Mo has a high melting point (2623°C), and the interaction parameter with Fe is positive (phase separation tendency).
  * **Effect on liquidus** : Mo addition tends to **increase** the liquidus temperature (solidification temperature range expands).
  * **Solidification segregation** : Mo tends to concentrate in the liquid during solidification (partition coefficient k < 1), so Mo concentration increases in the final stage of solidification.
  * **Practical significance** : As the solidification temperature range expands, the risk of solidification cracking during welding increases. It is important to predict the solidification path with CALPHAD and optimize the Mo amount.

**CALPHAD calculation (concept)** : Using Thermo-Calc's TCFE (steel) database, vertical sections of Fe-17Cr-12Ni-xMo (x = 0, 1, 2, 3 wt%) can be calculated to quantify changes in liquidus and solidus.

## Summary

In this chapter, we learned how to read ternary phase diagrams and the principles of the CALPHAD method.

#### Review of Key Points

  1. **Gibbs triangle** : Represents ternary composition on a 2D equilateral triangle. Each vertex is a pure component, each edge is a binary system.
  2. **Isothermal section** : Shows phase equilibrium at specific temperature. Represents two-phase and three-phase regions with tie-lines and tie-triangles.
  3. **Liquidus projection** : Contour map of liquidus temperature. Primary crystallization lines and eutectic valleys are important. Useful for tracking cooling paths.
  4. **Vertical section** : Temperature-composition diagram with fixed composition ratio. Can be analyzed in the same format as binary phase diagrams.
  5. **Ternary eutectic reaction** : Invariant reaction L → α + β + γ. Tie-triangle represents equilibrium compositions of three phases.
  6. **CALPHAD method** : Phase diagram calculation by Gibbs free energy minimization. Utilizes thermodynamic database (TDB).
  7. **Redlich-Kister equation** : Models excess Gibbs free energy of solution phase. Expresses non-ideality with interaction parameter L.
  8. **CALPHAD workflow** : Literature review → Model selection → Parameter optimization → Database construction → Phase diagram calculation and verification.

#### 💡 Next Steps

In the next chapter, we will learn **practical phase diagram calculation using pycalphad**. Using the open-source Python library pycalphad, we will calculate phase diagrams from actual thermodynamic databases (TDB) and quantitatively analyze the relationship between temperature, composition, and phase fraction. Let's experience the power of the CALPHAD method through phase diagram calculations of practical materials such as Fe-C system, Al-Cu system, and Ni-based superalloys.

#### Learning Check

Check if you can answer the following questions:

  * Can you plot the composition point of Fe-18Cr-8Ni on the Gibbs triangle and accurately read the composition?
  * 1200℃isothermal section, can you calculate the phase fraction of each phase in an alloy within the L+γ two-phase region using the lever rule?
  * Can you track the solidification initiation temperature and cooling path of an alloy with a specific composition from the liquidus projection?
  * Can you analyze the effect of Fe content when the Cr/Ni ratio is fixed using a vertical section?
  * Can you explain the characteristics of ternary eutectic reactions and the meaning of tie-triangles?
  * Can you explain what physical phenomena positive and negative deviations correspond to in the Redlich-Kister equation?
  * Can you explain the parameter optimization process of the CALPHAD method, including comparison with experimental data?
  * Can you explain the structure of TDB files and the role of thermodynamic databases?

[← Chapter 4: Reading and Analysis of Binary Phase Diagrams](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
