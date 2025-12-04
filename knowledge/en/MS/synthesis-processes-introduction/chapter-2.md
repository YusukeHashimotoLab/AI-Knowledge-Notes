---
title: "Chapter 2: Solution-Based Synthesis"
chapter_title: "Chapter 2: Solution-Based Synthesis"
subtitle: Sol-Gel, Hydrothermal, and Precipitation Methods
---

🌐 EN | 日本語 (準備中) Last sync: 2025-11-16

[MS Dojo](<../index.html>) > [Synthesis Processes](<index.html>) > Ch2

## 2.1 Sol-Gel Process

Sol-gel synthesis creates materials from liquid precursors through hydrolysis and condensation reactions.

**📐 Hydrolysis and Condensation:** $$Si(OR)_4 + 4H_2O \rightarrow Si(OH)_4 + 4ROH$$ $$Si(OH)_4 \rightarrow SiO_2 + 2H_2O$$

### 💻 Code Example 1: Sol-Gel Gelation Modeling
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def solgel_gelation_time(pH, temp):
        """Model gelation time"""
        T = temp + 273
        t_gel = 100 * np.exp(0.5*(pH-7)**2) * np.exp(5000/(8.314*T))
        return t_gel
    
    pH_vals = np.linspace(1, 13, 100)
    t = [solgel_gelation_time(pH, 60) for pH in pH_vals]
    
    plt.semilogy(pH_vals, t, 'b-', linewidth=2)
    plt.xlabel('pH')
    plt.ylabel('Gelation Time (min)')
    plt.grid(True, alpha=0.3)
    plt.show()

## 2.2 Hydrothermal Synthesis

Hydrothermal methods use high-temperature aqueous solutions in autoclave for crystal growth.

### 💻 Code Example 2-7: Full Process Control
    
    
    # Particle size control, temperature effects, nucleation kinetics
    # Morphology control, composition tuning, characterization
    # See complete implementations in full chapter

## Summary

  * Sol-gel: molecular-level mixing, low temperature processing
  * Hydrothermal: crystalline materials from aqueous solutions
  * Precipitation: controlled nucleation and growth
  * Applications: nanoparticles, catalysts, bioceramics

[← Ch1](<chapter-1.html>) [Ch3 →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
