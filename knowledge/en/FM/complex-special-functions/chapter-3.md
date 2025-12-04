---
title: "Chapter 3: Laurent Series and Residue Theorem"
chapter_title: "Chapter 3: Laurent Series and Residue Theorem"
subtitle: Laurent Series and Residue Theorem
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/complex-special-functions/chapter-3.html>) | Last sync: 2025-11-16

[Fundamentals Mathematics Dojo](<../index.html>) > [Complex Functions and Special Functions](<index.html>) > Chapter 3 

## 3.1 Taylor Series and Maclaurin Expansion

Analytic functions can be expanded into Taylor series within the circle of convergence. 

**📐 Definition: Taylor Series Expansion**  
$$f(z) = \sum_{n=0}^{\infty} \frac{f^{(n)}(z_0)}{n!} (z - z_0)^n$$ **Maclaurin expansion ($z_0 = 0$):** $$f(z) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} z^n$$ 

### 💻 Code Example 1: Calculation of Taylor Series Expansion

Python Implementation: Function Approximation by Taylor Series
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 💻 Code Example 1: Calculation of Taylor Series Expansion
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import factorial
    import sympy as sp
    
    # Symbolic computation with SymPy
    z = sp.Symbol('z')
    z0 = sp.Symbol('z0')
    
    # Function definitions
    functions_sym = {
        'e^z': sp.exp(z),
        'sin(z)': sp.sin(z),
        'cos(z)': sp.cos(z),
        '1/(1-z)': 1/(1-z),
    }
    
    print("=== Taylor Series Expansion (Maclaurin expansion, z0=0) ===\n")
    
    for name, f_sym in functions_sym.items():
        print(f"f(z) = {name}")
        # Taylor expansion (up to 10th order)
        taylor_series = sp.series(f_sym, z, 0, n=6).removeO()
        print(f"Taylor series: {taylor_series}")
        print()
    
    # Visualization omitted (see original code)

## 3.2 Laurent Series Expansion

In regions containing singularities, expansion is done with Laurent series including negative powers. 

**📐 Definition: Laurent Series Expansion**  
$$f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n$$ Separated into regular part and principal part: $$f(z) = \underbrace{\sum_{n=0}^{\infty} a_n (z - z_0)^n}_{\text{Regular part}} + \underbrace{\sum_{n=1}^{\infty} \frac{a_{-n}}{(z - z_0)^n}}_{\text{Principal part}}$$ 

## 3.3 Classification of Singularities

Singularities are classified into three types: removable singularity, pole, and essential singularity. 

**📐 Theorem: Classification of Singularities**  

  * **Removable singularity:** Principal part is 0 → $\lim_{z \to z_0} f(z)$ is finite
  * **Pole of order $m$:** Principal part has finite terms up to $(z-z_0)^{-m}$
  * **Essential singularity:** Principal part has infinite terms

## 3.4 Calculation of Residues

The residue is the coefficient of $(z-z_0)^{-1}$ in Laurent expansion and is important for calculating complex integrals. 

**📐 Definition: Residue**  
$$\text{Res}(f, z_0) = a_{-1}$$ where $a_{-1}$ is coefficient of $(z-z_0)^{-1}$ in Laurent expansion $f(z) = \sum a_n (z-z_0)^n$  
  
**For pole of order $m$:** $$\text{Res}(f, z_0) = \frac{1}{(m-1)!} \lim_{z \to z_0} \frac{d^{m-1}}{dz^{m-1}} [(z-z_0)^m f(z)]$$ 

## 3.5 Residue Theorem

The residue theorem allows calculating complex integrals as sum of residues. 

**📐 Theorem: Residue Theorem**  
$$\oint_C f(z) dz = 2\pi i \sum_{k} \text{Res}(f, z_k)$$ where $z_k$ are singularities inside $C$ 

## 3.6 Applications to Real Integrals (1): Rational Functions

Using residue theorem, complex real integrals can be calculated by converting to complex integrals. 

**🔬 Application Example:** Real integrals of rational functions  
$$\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)} dx = 2\pi i \sum_{\text{upper half-plane}} \text{Res}(f, z_k)$$ converges when $\deg Q \geq \deg P + 2$ 

## 3.7 Applications to Real Integrals (2): Integrals with Trigonometric Functions

By substitution $z = e^{i\theta}$, integrals containing trigonometric functions can be converted to complex integrals. 

**📐 Definition: Transformation of Trigonometric Integrals**  
$$z = e^{i\theta}, \quad \cos\theta = \frac{z + z^{-1}}{2}, \quad \sin\theta = \frac{z - z^{-1}}{2i}$$ $$\int_0^{2\pi} R(\cos\theta, \sin\theta) d\theta = \oint_{|z|=1} R\left(\frac{z+z^{-1}}{2}, \frac{z-z^{-1}}{2i}\right) \frac{dz}{iz}$$ 

## 3.8 Applications to Real Integrals (3): Fourier-type Integrals

Residue theorem is also effective for integrals containing $e^{iax}$. 

**📐 Theorem: Fourier-type Integrals**  
$$\int_{-\infty}^{\infty} f(x) e^{iax} dx = 2\pi i \sum_{\text{Im}(z_k)>0} \text{Res}(f(z)e^{iaz}, z_k) \quad (a > 0)$$ 

## 3.9 Applications to Materials Science: Lattice Vibrations and Phonon Dispersion

In solid state physics, complex function theory is used when analyzing dispersion relations of lattice vibrations. 

**📝 Physical Significance:**

  * Poles of Green's function → Lattice vibration modes (phonons)
  * Spectral function → Density of states
  * Complex frequency → Damping of vibrations

## 📝 Chapter Exercises

**✏️ Exercises**

  1. Find the Laurent expansion of $f(z) = \frac{e^z}{z^3}$ around $z=0$.
  2. Calculate residues of $f(z) = \frac{1}{z(z-1)(z-2)}$ at all singularities.
  3. Calculate $\int_{-\infty}^{\infty} \frac{dx}{1+x^4}$ using residue theorem.
  4. Calculate $\int_0^{2\pi} \frac{d\theta}{3 + 2\cos\theta}$ using residue theorem.

## Summary

  * Laurent series provides function representation near singularities
  * Complex real integrals can be calculated using residue theorem
  * Wide applications in physics (quantum mechanics, statistical mechanics)
  * Understanding residues is important even in numerical computation

[← Chapter 2: Complex Integration](<chapter-2.html>) [Chapter 4: Fourier Transform →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
