---
title: "Chapter 1: Complex Numbers and Complex Plane"
chapter_title: "Chapter 1: Complex Numbers and Complex Plane"
subtitle: Complex Numbers and Complex Plane
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/complex-special-functions/chapter-1.html>) | Last sync: 2025-11-16

[Fundamentals Mathematics Dojo](<../index.html>) > [Complex Functions and Special Functions](<index.html>) > Chapter 1 

## 1.1 Basic Operations of Complex Numbers

Complex numbers are expressed in the form \\(z = x + iy\\) and have a real part \\(x\\) and an imaginary part \\(y\\). In Python, they can be handled with `complex` type or NumPy. 

**📐 Definition: Complex Numbers**  
Definition of complex numbers: \\[z = x + iy, \quad i = \sqrt{-1}\\] Basic operations: 

  * Addition: \\((x_1 + iy_1) + (x_2 + iy_2) = (x_1 + x_2) + i(y_1 + y_2)\\)
  * Multiplication: \\((x_1 + iy_1)(x_2 + iy_2) = (x_1x_2 - y_1y_2) + i(x_1y_2 + x_2y_1)\\)
  * Conjugate: \\(\bar{z} = x - iy\\)
  * Absolute value: \\(|z| = \sqrt{x^2 + y^2}\\)

### 💻 Code Example 1: Basic Operations of Complex Numbers

Python Implementation: Basic Operations of Complex Numbers
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 💻 Code Example 1: Basic Operations of Complex Numbers
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define complex numbers
    z1 = 3 + 4j
    z2 = 1 - 2j
    
    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
    print(f"z1 + z2 = {z1 + z2}")
    print(f"z1 * z2 = {z1 * z2}")
    print(f"z1 / z2 = {z1 / z2}")
    
    # Complex conjugate and absolute value
    print(f"\nConjugate: z1.conjugate() = {z1.conjugate()}")
    print(f"Absolute value: |z1| = {np.abs(z1)}")
    print(f"Argument: arg(z1) = {np.angle(z1)} rad = {np.degrees(np.angle(z1)):.2f}°")
    
    # Visualization on complex plane
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Draw complex numbers as vectors
    def plot_complex(z, label, color):
        ax.arrow(0, 0, z.real, z.imag, head_width=0.3, head_length=0.2,
                 fc=color, ec=color, linewidth=2, label=label)
        ax.plot(z.real, z.imag, 'o', color=color, markersize=8)
        ax.text(z.real + 0.3, z.imag + 0.3, label, fontsize=12, color=color)
    
    plot_complex(z1, 'z1', 'blue')
    plot_complex(z2, 'z2', 'red')
    plot_complex(z1 + z2, 'z1+z2', 'green')
    
    ax.set_xlabel('Real part (Re)', fontsize=12)
    ax.set_ylabel('Imaginary part (Im)', fontsize=12)
    ax.set_title('Vector representation on complex plane', fontsize=14)
    ax.legend()
    ax.axis('equal')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-3, 5)
    plt.tight_layout()
    plt.show()

## 1.2 Polar Form and Euler's Formula

Complex numbers can also be expressed in polar form \\(z = r e^{i\theta}\\), which is based on Euler's formula \\(e^{i\theta} = \cos\theta + i\sin\theta\\). 

**📐 Theorem: Euler's Formula**  
Polar form representation: \\[z = r e^{i\theta} = r(\cos\theta + i\sin\theta)\\] where \\(r = |z|\\) (absolute value), \\(\theta = \arg(z)\\) (argument)  
Special case: \\[e^{i\pi} + 1 = 0 \quad \text{(Euler's identity)}\\] 

## Summary

  * Complex numbers consist of real and imaginary parts and can be represented as vectors on the complex plane
  * Polar form representation allows understanding multiplication and division of complex numbers as rotation and scaling
  * Euler's formula is an important relation connecting complex numbers and trigonometric functions
  * Complex numbers are used to describe various physical phenomena such as AC circuit analysis and quantum mechanics

[← Series Top](<index.html>) [Chapter 2: Analytic Functions and Complex Calculus →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
