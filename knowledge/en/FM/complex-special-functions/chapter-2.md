---
title: "Chapter 2: Analytic Functions and Complex Calculus"
chapter_title: "Chapter 2: Analytic Functions and Complex Calculus"
subtitle: Analytic Functions and Complex Calculus
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/complex-special-functions/chapter-2.html>) | Last sync: 2025-11-16

[Fundamentals Mathematics Dojo](<../index.html>) > [Complex Functions and Special Functions](<index.html>) > Chapter 2 

## 2.1 Complex Differentiation and Cauchy-Riemann Equations

A complex function $f(z)$ is differentiable at point $z_0$ if the limit $\lim_{h \to 0} \frac{f(z_0+h) - f(z_0)}{h}$ exists independently of how $h$ approaches zero. 

**📐 Theorem: Cauchy-Riemann Equations**  
Necessary and sufficient condition for $f(z) = u(x,y) + iv(x,y)$ to be analytic: $$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$ **Calculation of complex derivative:** $$f'(z) = \frac{\partial u}{\partial x} + i\frac{\partial v}{\partial x} = \frac{\partial v}{\partial y} - i\frac{\partial u}{\partial y}$$ 

### 💻 Code Example 1: Verification of Cauchy-Riemann Equations

Python Implementation: Verification of Cauchy-Riemann Equations
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.misc import derivative
    
    def f_analytic(z):
        """Analytic function: f(z) = z^2"""
        return z**2
    
    def f_not_analytic(z):
        """Non-analytic function: f(z) = z̄ (complex conjugate)"""
        return np.conj(z)
    
    # Separate real and imaginary parts
    def extract_uv(f, x, y):
        z = x + 1j*y
        w = f(z)
        return w.real, w.imag
    
    # Numerical calculation of partial derivatives
    def check_cauchy_riemann(f, x0, y0, h=1e-5):
        u, v = extract_uv(f, x0, y0)
    
        # ∂u/∂x
        u_xp, _ = extract_uv(f, x0+h, y0)
        u_xm, _ = extract_uv(f, x0-h, y0)
        du_dx = (u_xp - u_xm) / (2*h)
    
        # ∂u/∂y
        u_yp, _ = extract_uv(f, x0, y0+h)
        u_ym, _ = extract_uv(f, x0, y0-h)
        du_dy = (u_yp - u_ym) / (2*h)
    
        # ∂v/∂x
        _, v_xp = extract_uv(f, x0+h, y0)
        _, v_xm = extract_uv(f, x0-h, y0)
        dv_dx = (v_xp - v_xm) / (2*h)
    
        # ∂v/∂y
        _, v_yp = extract_uv(f, x0, y0+h)
        _, v_ym = extract_uv(f, x0, y0-h)
        dv_dy = (v_yp - v_ym) / (2*h)
    
        return du_dx, du_dy, dv_dx, dv_dy
    
    # Test point
    x0, y0 = 1.5, 2.0
    
    print("=== Analytic function: f(z) = z^2 ===")
    du_dx, du_dy, dv_dx, dv_dy = check_cauchy_riemann(f_analytic, x0, y0)
    print(f"∂u/∂x = {du_dx:.6f}")
    print(f"∂v/∂y = {dv_dy:.6f}")
    print(f"∂u/∂x - ∂v/∂y = {du_dx - dv_dy:.6e} (should be ~0)")
    print(f"\n∂u/∂y = {du_dy:.6f}")
    print(f"-∂v/∂x = {-dv_dx:.6f}")
    print(f"∂u/∂y - (-∂v/∂x) = {du_dy - (-dv_dx):.6e} (should be ~0)")
    
    print("\n\n=== Non-analytic function: f(z) = z̄ ===")
    du_dx, du_dy, dv_dx, dv_dy = check_cauchy_riemann(f_not_analytic, x0, y0)
    print(f"∂u/∂x = {du_dx:.6f}")
    print(f"∂v/∂y = {dv_dy:.6f}")
    print(f"∂u/∂x - ∂v/∂y = {du_dx - dv_dy:.6f} (NOT ~0)")
    print(f"\n∂u/∂y = {du_dy:.6f}")
    print(f"-∂v/∂x = {-dv_dx:.6f}")
    print(f"∂u/∂y - (-∂v/∂x) = {du_dy - (-dv_dx):.6f} (NOT ~0)")
    
    # Visualization omitted (see original code)

**📌 Note:** For analytic functions, the contour lines of real and imaginary parts are orthogonal (geometric meaning of Cauchy-Riemann equations). 

## 2.2 Examples and Properties of Analytic Functions

Many complex functions are analytic, but functions involving complex conjugate or absolute value are not analytic. 

**📐 Theorem: Analytic Functions**  
**Examples of analytic functions:**

  * Polynomials: $z^n$, $a_n z^n + \cdots + a_1 z + a_0$
  * Exponential function: $e^z$
  * Trigonometric functions: $\sin z$, $\cos z$
  * Rational functions: $\frac{P(z)}{Q(z)}$ (in region where $Q(z) \neq 0$)

**Examples of non-analytic functions:**

  * Complex conjugate: $\bar{z}$
  * Real/Imaginary part: $\mathrm{Re}(z)$, $\mathrm{Im}(z)$
  * Absolute value: $|z|$

## 2.3 Calculation of Complex Integrals

Complex integrals are defined as path integrals: $\int_C f(z) dz = \int_a^b f(z(t)) z'(t) dt$ 

**📐 Theorem: Complex Line Integral**  
$$\int_C f(z) dz = \int_a^b f(z(t)) \frac{dz}{dt} dt$$ where $z(t)$ is parametric representation of path $C$ $(a \leq t \leq b)$ 

## 2.4 Cauchy's Integral Theorem

The integral of an analytic function along a closed curve is zero. This is known as Cauchy's integral theorem. 

**📐 Theorem: Cauchy's Integral Theorem**  
$$\oint_C f(z) dz = 0$$ where $f(z)$ is analytic inside closed curve $C$ 

## 2.5 Cauchy's Integral Formula

The value of an analytic function can be obtained from values on the surrounding closed curve. 

**📐 Theorem: Cauchy's Integral Formula**  
**Cauchy's integral formula:** $$f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0} dz$$ **Formula for derivatives:** $$f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^{n+1}} dz$$ 

## 2.6 Conformal Mapping and Harmonic Functions

Analytic functions are angle-preserving mappings (conformal mappings). Also, the real and imaginary parts of analytic functions are harmonic functions (solutions to Laplace's equation). 

**📐 Theorem: Conformal Mapping and Harmonic Functions**  
**Conformal mapping:** Analytic function $w = f(z)$ preserves angles  
**Harmonic functions:** If $f(z) = u + iv$ is analytic, then $$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$ $$\nabla^2 v = \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} = 0$$ 

## 📝 Chapter Exercises

**✏️ Exercises**

  1. For $f(z) = z^3$, verify the Cauchy-Riemann equations at point $z_0 = 1+i$.
  2. Calculate $\oint_C z^n dz$ where $C$ is unit circle centered at origin and $n$ is an integer.
  3. For $f(z) = \frac{1}{z-2}$, calculate integral $\oint_C f(z) dz$ along unit circle and discuss relation to Cauchy's integral theorem.
  4. Find what curve the line $x=1$ is mapped to by conformal mapping $w = z^2$.

## 🔗 References

  * Ahlfors, L. V. (1979). _Complex Analysis_. McGraw-Hill.
  * Stein, E. M., & Shakarchi, R. (2003). _Complex Analysis_. Princeton University Press.

[← To Chapter 1](<chapter-1.html>) [To Chapter 3 →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
