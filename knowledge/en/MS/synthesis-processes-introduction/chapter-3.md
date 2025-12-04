---
title: "Chapter 3: Vapor Phase Synthesis (CVD, PVD)"
chapter_title: "Chapter 3: Vapor Phase Synthesis (CVD, PVD)"
subtitle: Chemical and Physical Vapor Deposition
---

🌐 EN | 日本語 (準備中) Last sync: 2025-11-16

[MS Dojo](<../index.html>) > [Synthesis Processes](<index.html>) > Ch3

## 3.1 Chemical Vapor Deposition (CVD)

CVD deposits thin films through chemical reactions of gas-phase precursors.

**📐 Deposition Rate:** $$r = k_s P \exp\left(-\frac{E_a}{RT}\right)$$

### 💻 Code Example 1: CVD Rate Modeling
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def cvd_rate(T, P, Ea=150000):
        """CVD deposition rate"""
        R = 8.314
        rate = 1e6 * P * np.exp(-Ea/(R*(T+273)))
        return rate
    
    temps = np.linspace(400, 800, 100)
    rates = [cvd_rate(T, 100) for T in temps]
    
    plt.semilogy(temps, rates, 'b-', linewidth=2)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Deposition Rate (nm/min)')
    plt.grid(True, alpha=0.3)
    plt.show()

## 3.2 Physical Vapor Deposition (PVD)

PVD includes sputtering and evaporation methods.

## Summary

  * CVD: chemical reactions in gas phase
  * PVD: physical processes (sputtering, evaporation)
  * Applications: semiconductors, optical coatings

[← Ch2](<chapter-2.html>) [Ch4 →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
