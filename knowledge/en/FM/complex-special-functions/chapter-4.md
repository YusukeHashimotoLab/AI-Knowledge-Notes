---
title: "Chapter 4: Fourier Transform and Laplace Transform"
chapter_title: "Chapter 4: Fourier Transform and Laplace Transform"
subtitle: Fourier and Laplace Transforms
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/complex-special-functions/chapter-4.html>) | Last sync: 2025-11-16

[Fundamentals Mathematics Dojo](<../index.html>) > [Complex Functions and Special Functions](<index.html>) > Chapter 4 

## 4.1 Fourier Series

Periodic functions can be represented by series of trigonometric functions (Fourier series). 

**📐 Definition: Fourier Series Expansion**  
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos\frac{n\pi x}{L} + b_n \sin\frac{n\pi x}{L} \right)$$ **Fourier coefficients:** $$a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\frac{n\pi x}{L} dx$$ $$b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\frac{n\pi x}{L} dx$$ 

### 💻 Code Example 1: Fourier Series Expansion

Python Implementation: Fourier Series Approximation of Square Wave
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import integrate
    
    def fourier_coefficients(f, L, n_max):
        """Calculate Fourier coefficients"""
        a0 = (1/L) * integrate.quad(f, -L, L)[0]
    
        a_n = []
        b_n = []
    
        for n in range(1, n_max + 1):
            # a_n
            integrand_a = lambda x: f(x) * np.cos(n * np.pi * x / L)
            a_n.append((1/L) * integrate.quad(integrand_a, -L, L)[0])
    
            # b_n
            integrand_b = lambda x: f(x) * np.sin(n * np.pi * x / L)
            b_n.append((1/L) * integrate.quad(integrand_b, -L, L)[0])
    
        return a0, np.array(a_n), np.array(b_n)
    
    # Test function: square wave
    L = np.pi
    def square_wave(x):
        return np.where(np.abs(x) < L/2, 1.0, 0.0)
    
    # Visualization omitted (see original code)

## 4.2 Fourier Transform

For non-periodic functions, we use the Fourier transform, which is a continuous version of Fourier series. 

**📐 Definition: Fourier Transform**  
$$F(\omega) = \mathcal{F}[f(t)] = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt$$ **Inverse Fourier transform:** $$f(t) = \mathcal{F}^{-1}[F(\omega)] = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} d\omega$$ 

## 4.3 Convolution Theorem

Fourier transform converts convolution operation into simple multiplication. 

**📐 Theorem: Convolution Theorem**  
$$\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]$$ where convolution $(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau$ 

## 4.4 Laplace Transform

Laplace transform is a generalization of one-sided Fourier transform and is useful for solving differential equations. 

**📐 Definition: Laplace Transform**  
$$F(s) = \mathcal{L}[f(t)] = \int_0^{\infty} f(t) e^{-st} dt$$ **Main properties:**

  * Differentiation: $\mathcal{L}[f'(t)] = sF(s) - f(0)$
  * Integration: $\mathcal{L}\left[\int_0^t f(\tau)d\tau\right] = \frac{F(s)}{s}$
  * Convolution: $\mathcal{L}[f * g] = F(s) \cdot G(s)$

## 4.5 Inverse Laplace Transform and Differential Equations

Using Laplace transform, differential equations can be transformed into algebraic equations. 

**🔬 Application Example:** Solving differential equations  
Differential equation: $y'' + 4y' + 3y = e^{-t}$, $y(0) = 0$, $y'(0) = 0$  
  
By Laplace transform:  
$(s^2 + 4s + 3)Y(s) = \frac{1}{s+1}$  
  
Solution: $Y(s) = \frac{1}{(s+1)^2(s+3)}$ 

## 4.6 Properties of Fourier Transform

Fourier transform has various useful properties. 

**📝 Main properties:**

  * Time shift: $f(t-t_0) \rightarrow e^{-i\omega t_0}F(\omega)$
  * Scaling: $f(at) \rightarrow \frac{1}{|a|}F(\omega/a)$
  * Differentiation: $f'(t) \rightarrow i\omega F(\omega)$

## 4.7 Window Functions and Spectral Leakage

When analyzing finite-length signals with FFT, window functions are used to suppress spectral leakage. 

**📐 Theorem: Characteristics of Window Functions**  

  * **Rectangular:** Minimum main lobe width, large side lobes
  * **Hann:** Well-balanced, general purpose
  * **Blackman:** Minimum side lobes, large main lobe width

## 4.8 Applications to Materials Science: X-ray Diffraction Pattern Analysis

In crystal structure analysis, atomic arrangement in real space corresponds to reciprocal space (diffraction pattern) by Fourier transform. 

**🔬 Physical Significance:**

  * Periodic structure in real space → Discrete Bragg peaks in reciprocal space
  * Large lattice constant $a$ → Small Bragg peak spacing
  * Large crystal size → Sharp Bragg peaks

## 📝 Chapter Exercises

**✏️ Exercises**

  1. Find Fourier series expansion of square wave up to 10th order and observe Gibbs phenomenon.
  2. Calculate Fourier transform of Gaussian function $f(t) = e^{-t^2/(2\sigma^2)}$ and confirm self-duality.
  3. Use convolution theorem to find frequency response of cascade connection of two lowpass filters.
  4. Solve differential equation $y'' + 2y' + 2y = \sin(t)$, $y(0)=0$, $y'(0)=1$ using Laplace transform.

## Summary

  * Fourier series represents periodic functions as sum of trigonometric functions
  * Fourier transform mutually converts between time and frequency domains
  * Laplace transform converts differential equations into algebraic equations
  * Convolution theorem makes signal processing efficient
  * Wide applications in materials science (X-ray diffraction) and engineering (control theory)

[← Chapter 3: Laurent Expansion](<chapter-3.html>) [Chapter 5: Special Functions →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
