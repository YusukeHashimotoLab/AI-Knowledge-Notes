---
title: "Chapter 5: Special Functions and Boundary Value Problems"
chapter_title: "Chapter 5: Special Functions and Boundary Value Problems"
subtitle: Special Functions and Boundary Value Problems
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/complex-special-functions/chapter-5.html>) | Last sync: 2025-11-16

[Fundamentals Mathematics Dojo](<../index.html>) > [Complex Functions and Special Functions](<index.html>) > Chapter 5 

## 5.1 Definition and Properties of Bessel Functions

Bessel functions appear as solutions to wave equations and diffusion equations in cylindrical coordinate systems. 

**📐 Definition: Bessel Differential Equation**  
$$x^2 \frac{d^2y}{dx^2} + x\frac{dy}{dx} + (x^2 - \nu^2)y = 0$$ **Bessel function of the first kind $J_\nu(x)$:** $$J_\nu(x) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m!\Gamma(m+\nu+1)} \left(\frac{x}{2}\right)^{2m+\nu}$$ 

### 💻 Code Example 1: Calculation and Visualization of Bessel Functions

Python Implementation: Basic Properties of Bessel Functions
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 💻 Code Example 1: Calculation and Visualization of Bessel Fu
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import jv, yv, jn_zeros
    
    # Visualization of Bessel functions
    x = np.linspace(0, 20, 1000)
    orders = [0, 1, 2, 3, 4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Bessel function of the first kind J_ν(x)
    ax = axes[0, 0]
    for nu in orders:
        y = jv(nu, x)
        ax.plot(x, y, linewidth=2, label=f'$J_{{{nu}}}(x)$')
    ax.legend()
    ax.set_title('Bessel Function of the First Kind')
    
    # Visualization omitted (see original code)

## 5.2 Wave Equation in Cylindrical Coordinates

Solutions to wave equations in cylindrical coordinate systems are represented by Bessel functions. 

**📐 Theorem: Wave Equation in Cylindrical Coordinates**  
$$\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial u}{\partial r}\right) + \frac{1}{r^2}\frac{\partial^2 u}{\partial \theta^2} + \frac{\partial^2 u}{\partial z^2} = \frac{1}{c^2}\frac{\partial^2 u}{\partial t^2}$$ **Separation of variables solution:** $$u(r,\theta,z,t) = J_m(k_r r) e^{im\theta} e^{ik_z z} e^{-i\omega t}$$ 

## 5.3 Legendre Polynomials

In boundary value problems in spherical coordinate systems, Legendre polynomials play an important role. 

**📐 Definition: Legendre Differential Equation**  
$$\frac{d}{dx}\left[(1-x^2)\frac{dy}{dx}\right] + n(n+1)y = 0$$ **Rodrigues' formula:** $$P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n}[(x^2-1)^n]$$ 

## 5.4 Spherical Harmonics

For three-dimensional problems in spherical coordinate systems, we use spherical harmonics constructed from associated Legendre functions. 

**📐 Definition: Spherical Harmonics**  
$$Y_l^m(\theta, \phi) = \sqrt{\frac{(2l+1)}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}$$ where $P_l^m$ is associated Legendre function 

**📝 Applications to Quantum Mechanics:**

  * Angular part of hydrogen atom wavefunction
  * Eigenstates of angular momentum
  * $l$: azimuthal quantum number, $m$: magnetic quantum number ($-l \leq m \leq l$)

## 5.5 Hermite Polynomials

Learn Hermite polynomials that appear as quantum mechanical solutions of harmonic oscillator. 

**📐 Definition: Hermite Differential Equation**  
$$\frac{d^2y}{dx^2} - 2x\frac{dy}{dx} + 2ny = 0$$ **Rodrigues' formula:** $$H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2}$$ 

**🔬 Application Example:** Energy eigenvalues of harmonic oscillator  
$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, \ldots$$ Wavefunction: $$\psi_n(x) = \frac{1}{\sqrt{2^n n!}} \left(\frac{1}{\pi}\right)^{1/4} e^{-x^2/2} H_n(x)$$ 

## 5.6 Laguerre Polynomials

Learn Laguerre polynomials that appear in radial wavefunctions of hydrogen atom. 

**📐 Definition: Generalized Laguerre Polynomials**  
$$L_n^{\alpha}(x) = \frac{e^x x^{-\alpha}}{n!} \frac{d^n}{dx^n}(e^{-x} x^{n+\alpha})$$ 

**🔬 Application Example:** Energy levels of hydrogen atom  
$$E_n = -\frac{13.6 \text{ eV}}{n^2}, \quad n = 1, 2, 3, \ldots$$ Radial wavefunction includes $L_{n-l-1}^{2l+1}$ 

## 5.7 Summary and Relationships of Special Functions

**📝 Correspondence between Coordinate Systems and Special Functions:**

  * Cartesian coordinates: Trigonometric functions (sin, cos)
  * Cylindrical coordinates: Bessel functions ($J_n$, $Y_n$)
  * Spherical coordinates: Legendre polynomials ($P_n$), Spherical harmonics ($Y_l^m$)

**📝 Quantum Mechanics and Special Functions:**

  * Harmonic oscillator: Hermite polynomials ($H_n$)
  * Hydrogen atom (radial): Laguerre polynomials ($L_n^{\alpha}$)
  * Hydrogen atom (angular): Spherical harmonics ($Y_l^m$)
  * Angular momentum: Spherical harmonics ($Y_l^m$)

## 5.8 Applications to Materials Science (1): Heat Conduction in Cylindrical Samples

Solve heat conduction equation in cylindrical coordinate system with Bessel functions. 

**🔬 Physical Significance:**  
Solution to heat conduction equation: $$T(r,t) = T_{\text{surface}} + \sum_{n} A_n J_0(\lambda_n r) e^{-\alpha \lambda_n^2 t}$$ Characteristic time: $\tau = R^2/\alpha$ (temperature decays to about 37%) 

## 5.9 Applications to Materials Science (2): Diffusion Problems in Spherical Particles

Solutions to diffusion equations in spherical coordinate systems and applications to drug release systems. 

**🔬 Application Examples:**

  * Drug release capsules (DDS: Drug Delivery System)
  * Reactant diffusion from catalyst particles
  * Dissolution of nanoparticles

## 📝 Chapter Exercises

**✏️ Exercises**

  1. Numerically find the first 3 positive zeros of $J_0(x)$ and calculate eigenfrequencies of circular membrane.
  2. Verify orthogonality of Legendre polynomials $P_2(x)$ and $P_3(x)$ by numerical integration.
  3. Visualize spherical harmonic $Y_2^1(\theta, \phi)$ and confirm its nodal lines.
  4. Confirm number of nodes in wavefunction $\psi_3(x)$ of harmonic oscillator and find positions of classical turning points.

## Summary

  * Special functions are solutions to differential equations according to coordinate systems and boundary conditions
  * Bessel functions: Wave and diffusion problems in cylindrical coordinates
  * Legendre polynomials: Laplace equation in spherical coordinates
  * Spherical harmonics: Angular dependence of 3D spherically symmetric problems
  * Hermite polynomials: Harmonic oscillator (quantum mechanics)
  * Laguerre polynomials: Radial wavefunction of hydrogen atom
  * Wide applications in materials science (heat conduction, diffusion) and quantum mechanics

[← Chapter 4: Fourier Transform](<chapter-4.html>) [To Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
