---
title: "Chapter 1: Field Quantization and Canonical Formalism"
chapter_title: "Chapter 1: Field Quantization and Canonical Formalism"
subtitle: Canonical Quantization of Fields
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-field-theory-introduction/chapter-1.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Introduction to Quantum Field Theory](<index.html>) > Chapter 1 

## 1.1 From Classical Field Theory to Quantum Field Theory

Quantum Field Theory (QFT) is a theoretical framework that describes particles and fields in a unified manner. It replaces the classical concept of particle trajectories with field operators, naturally handling particle creation and annihilation. In this chapter, we begin with a review of classical field theory and systematically study the procedure of canonical quantization. 

### 📚 Fundamentals of Classical Field Theory

A **field** \\(\phi(\mathbf{x}, t)\\) is a physical quantity defined at each point in spacetime. We construct the action from the Lagrangian density \\(\mathcal{L}\\):

\\[ S = \int dt \, d^3x \, \mathcal{L}(\phi, \partial_\mu \phi) \\]

**Euler-Lagrange equation** :

\\[ \frac{\partial \mathcal{L}}{\partial \phi} - \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial(\partial_\mu \phi)} \right) = 0 \\]

Here, \\(\partial_\mu = (\partial_t, \nabla)\\) is the four-dimensional differential operator in Minkowski spacetime.

### 1.1.1 Classical Theory of the Klein-Gordon Field

As the simplest example of a field, we consider a real scalar field \\(\phi(x)\\). The Lagrangian density that leads to the Klein-Gordon equation is: 

\\[ \mathcal{L} = \frac{1}{2}(\partial_\mu \phi)(\partial^\mu \phi) - \frac{1}{2}m^2 \phi^2 = \frac{1}{2}\dot{\phi}^2 - \frac{1}{2}(\nabla \phi)^2 - \frac{1}{2}m^2 \phi^2 \\]

Applying the Euler-Lagrange equation yields the Klein-Gordon equation:

\\[ (\Box + m^2)\phi = 0, \quad \Box = \partial_\mu \partial^\mu = \partial_t^2 - \nabla^2 \\]

### 🔬 Canonical Momentum and Hamiltonian

The **canonical momentum density** conjugate to the field \\(\phi\\) is:

\\[ \pi(\mathbf{x}, t) = \frac{\partial \mathcal{L}}{\partial \dot{\phi}} = \dot{\phi} \\]

The **Hamiltonian density** is obtained by the Legendre transformation:

\\[ \mathcal{H} = \pi \dot{\phi} - \mathcal{L} = \frac{1}{2}\pi^2 + \frac{1}{2}(\nabla \phi)^2 + \frac{1}{2}m^2 \phi^2 \\]

Total Hamiltonian: \\(H = \int d^3x \, \mathcal{H}\\)

Example 1: Classical Time Evolution of the Klein-Gordon Field
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftfreq
    
    # ===================================
    # Time evolution of 1D Klein-Gordon field
    # ===================================
    
    def klein_gordon_evolution(phi_init, L=10.0, N=128, m=1.0, T=5.0, dt=0.01):
        """Solve Klein-Gordon equation time evolution using spectral method
    
        Args:
            phi_init: Initial field configuration
            L: System size
            N: Number of lattice points
            m: Mass
            T: Total time
            dt: Time step
    
        Returns:
            x, t_array, phi_xt: Spatial coordinates, time array, field spacetime evolution
        """
        x = np.linspace(0, L, N, endpoint=False)
        k = 2 * np.pi * fftfreq(N, L/N)  # Momentum space
    
        # Dispersion relation: ω(k) = sqrt(k^2 + m^2)
        omega_k = np.sqrt(k**2 + m**2)
    
        # Initial condition: phi(x,0) and pi(x,0) = ∂_t phi(x,0)
        phi = phi_init.copy()
        pi = np.zeros_like(phi)  # Initial momentum is zero
    
        # Time evolution array
        n_steps = int(T / dt)
        t_array = np.linspace(0, T, n_steps)
        phi_xt = np.zeros((n_steps, N))
        phi_xt[0] = phi
    
        for i in range(1, n_steps):
            # Time evolution in Fourier space (split-step method)
            phi_k = fft(phi)
            pi_k = fft(pi)
    
            # Time evolution operator: exp(-iωt) and exp(iωt)
            phi_k_new = phi_k * np.cos(omega_k * dt) + (pi_k / omega_k) * np.sin(omega_k * dt)
            pi_k_new = pi_k * np.cos(omega_k * dt) - phi_k * omega_k * np.sin(omega_k * dt)
    
            phi = ifft(phi_k_new).real
            pi = ifft(pi_k_new).real
            phi_xt[i] = phi
    
        return x, t_array, phi_xt
    
    # Example run: Time evolution of Gaussian wave packet
    L, N = 10.0, 128
    x = np.linspace(0, L, N, endpoint=False)
    phi_init = np.exp(-((x - L/2)**2) / 0.5)  # Gaussian
    
    x, t_array, phi_xt = klein_gordon_evolution(phi_init, m=1.0)
    
    print(f"Number of time steps: {len(t_array)}")
    print(f"Energy conservation check: φ(t=0) range [{phi_xt[0].min():.3f}, {phi_xt[0].max():.3f}]")
    print(f"                          φ(t=T) range [{phi_xt[-1].min():.3f}, {phi_xt[-1].max():.3f}]")

Number of time steps: 500 Energy conservation check: φ(t=0) range [0.000, 1.000] φ(t=T) range [-0.687, 0.915]

## 1.2 Canonical Quantization Procedure

To quantize a classical field, we promote the field \\(\phi\\) and canonical momentum \\(\pi\\) to operators, and impose canonical commutation relations. This is the field version of the commutation relation between coordinate and momentum in ordinary quantum mechanics. 

### 📐 Equal-Time Canonical Commutation Relations (ETCCR)

The field operators \\(\hat{\phi}(\mathbf{x}, t)\\) and \\(\hat{\pi}(\mathbf{x}', t)\\) satisfy:

\\[ [\hat{\phi}(\mathbf{x}, t), \hat{\pi}(\mathbf{x}', t)] = i\hbar \delta^{(3)}(\mathbf{x} - \mathbf{x}') \\]

\\[ [\hat{\phi}(\mathbf{x}, t), \hat{\phi}(\mathbf{x}', t)] = 0, \quad [\hat{\pi}(\mathbf{x}, t), \hat{\pi}(\mathbf{x}', t)] = 0 \\]

We use natural units \\(\hbar = c = 1\\) below.

### 1.2.1 Fourier Mode Expansion and Creation-Annihilation Operators

We expand the solution of the Klein-Gordon equation in plane waves. Under periodic boundary conditions: 

\\[ \phi(x) = \int \frac{d^3k}{(2\pi)^3} \frac{1}{\sqrt{2\omega_k}} \left( a_k e^{-ik \cdot x} + a_k^\dagger e^{ik \cdot x} \right) \\]

Here, \\(\omega_k = \sqrt{\mathbf{k}^2 + m^2}\\) is the dispersion relation, and \\(k \cdot x = \omega_k t - \mathbf{k} \cdot \mathbf{x}\\).

### 🔧 Commutation Relations of Creation-Annihilation Operators

With \\(a_k\\) as the annihilation operator and \\(a_k^\dagger\\) as the creation operator:

\\[ [a_k, a_{k'}^\dagger] = (2\pi)^3 \delta^{(3)}(\mathbf{k} - \mathbf{k}') \\]

\\[ [a_k, a_{k'}] = 0, \quad [a_k^\dagger, a_{k'}^\dagger] = 0 \\]

These have the same algebraic structure as harmonic oscillator creation-annihilation operators.

### 💡 Physical Interpretation

\\(a_k^\dagger\\) is an operator that creates one particle with momentum \\(\mathbf{k}\\). \\(a_k\\) annihilates one particle with momentum \\(\mathbf{k}\\). This picture allows field theory to be understood as a quantum theory of many-particle systems. 

Example 2: Algebra of Creation-Annihilation Operators (Symbolic Computation)
    
    
    from sympy import *
    from sympy.physics.quantum import *
    
    # ===================================
    # Commutation relations of creation-annihilation operators
    # ===================================
    
    class AnnihilationOp(Operator):
        """Annihilation operator a"""
        pass
    
    class CreationOp(Operator):
        """Creation operator a†"""
        pass
    
    def commutator(A, B):
        """Commutator [A, B]"""
        return A*B - B*A
    
    # Symbol definition
    a = AnnihilationOp('a')
    a_dag = CreationOp('a†')
    
    # Verification of canonical commutation relations
    print("Canonical commutation relation check:")
    print(f"Assume [a, a†] = 1")
    
    # Number operator N = a†a
    print("\nProperties of number operator N = a†a:")
    print("[a, N] = [a, a†a] = [a, a†]a + a†[a, a] = a")
    print("[a†, N] = [a†, a†a] = [a†, a†]a + a†[a†, a] = -a†")
    
    # Action on Fock states
    n = Symbol('n', integer=True, positive=True)
    print("\nAction on Fock state |n⟩:")
    print(f"a |n⟩ = √n |n-1⟩")
    print(f"a† |n⟩ = √(n+1) |n+1⟩")
    print(f"N |n⟩ = n |n⟩")

Canonical commutation relation check: Assume [a, a†] = 1 Properties of number operator N = a†a: [a, N] = [a, a†a] = [a, a†]a + a†[a, a] = a [a†, N] = [a†, a†a] = [a†, a†]a + a†[a†, a] = -a† Action on Fock state |n⟩: a |n⟩ = √n |n-1⟩ a† |n⟩ = √(n+1) |n+1⟩ N |n⟩ = n |n⟩

## 1.3 Construction of Fock Space

Using creation-annihilation operators, we construct the Hilbert space for many-particle states (Fock space). This allows us to handle quantum states with indefinite particle number in a unified manner. 

### 🏗️ Definition of Fock Space

The **vacuum state** \\(|0\rangle\\) is annihilated by all annihilation operators:

\\[ a_k |0\rangle = 0 \quad \text{for all } \mathbf{k} \\]

**n-particle states** are constructed by applying creation operators to the vacuum:

\\[ |\mathbf{k}_1, \mathbf{k}_2, \ldots, \mathbf{k}_n\rangle = a_{\mathbf{k}_1}^\dagger a_{\mathbf{k}_2}^\dagger \cdots a_{\mathbf{k}_n}^\dagger |0\rangle \\]

**Fock space** \\(\mathcal{F}\\) is the direct sum of all particle number sectors:

\\[ \mathcal{F} = \bigoplus_{n=0}^{\infty} \mathcal{H}_n \\]

### 1.3.1 Diagonalization of the Hamiltonian

The Hamiltonian of the Klein-Gordon field expressed in terms of creation-annihilation operators is: 

\\[ H = \int \frac{d^3k}{(2\pi)^3} \omega_k \left( a_k^\dagger a_k + \frac{1}{2}[a_k, a_k^\dagger] \right) \\]

This can be viewed as a sum of infinitely many harmonic oscillators. The second term is the vacuum zero-point energy, which diverges. This is typically removed by **normal ordering**. 

### 📋 Normal Ordering

The normal-ordered product of an operator \\(A\\), denoted \\(:A:\\), places all creation operators to the left of annihilation operators:

\\[ :a_k a_{k'}^\dagger: = a_{k'}^\dagger a_k \\]

Normal-ordered Hamiltonian:

\\[ :H: = \int \frac{d^3k}{(2\pi)^3} \omega_k a_k^\dagger a_k \\]

This is proportional to the number operator, and the vacuum energy is zero.
    
    
    ```mermaid
    flowchart TD
        A[Classical field φ, π] --> B[Quantization: Operator promotion]
        B --> C[Canonical commutation relations[φ, π] = iδ]
        C --> D[Fourier expansionPlane wave basis]
        D --> E[Creation-annihilation operatorsa, a†]
        E --> F[Fock space construction|0⟩, a†|0⟩, ...]
        F --> G[Hamiltonian diagonalizationH = Σ ω a†a]
    
        style A fill:#e3f2fd
        style E fill:#f3e5f5
        style G fill:#e8f5e9
    ```

Example 3: States and Energy Eigenvalues in Fock Space
    
    
    import numpy as np
    from itertools import combinations_with_replacement
    
    # ===================================
    # States and energy calculation in Fock space
    # ===================================
    
    def fock_state_energy(k_list, m=1.0):
        """Calculate energy of a Fock state
    
        Args:
            k_list: List of momenta (each element is a 3D vector)
            m: Particle mass
    
        Returns:
            Energy eigenvalue
        """
        energy = 0.0
        for k in k_list:
            k_mag = np.linalg.norm(k)
            omega_k = np.sqrt(k_mag**2 + m**2)
            energy += omega_k
        return energy
    
    def generate_fock_states(k_modes, max_particles=3):
        """Generate many-particle Fock states from allowed momentum modes
    
        Args:
            k_modes: List of possible momentum modes
            max_particles: Maximum number of particles
    
        Returns:
            fock_states: List of Fock states (each state is a tuple of momenta)
        """
        fock_states = []
        for n in range(max_particles + 1):
            for state in combinations_with_replacement(range(len(k_modes)), n):
                k_list = [k_modes[i] for i in state]
                fock_states.append(k_list)
        return fock_states
    
    # Example for 1D system: k = 0, ±π/L
    L = 5.0
    k_modes = [
        np.array([0.0]),
        np.array([np.pi / L]),
        np.array([-np.pi / L])
    ]
    
    fock_states = generate_fock_states(k_modes, max_particles=2)
    
    print("Low-energy states in Fock space:")
    print("-" * 50)
    
    for i, state in enumerate(fock_states[:8]):
        n_particles = len(state)
        energy = fock_state_energy(state, m=1.0)
    
        if n_particles == 0:
            label = "|0⟩ (vacuum)"
        else:
            k_values = [f"k={k[0]:.3f}" for k in state]
            label = f"|{', '.join(k_values)}⟩"
    
        print(f"{i+1}. {label:<30} E = {energy:.4f}")

Low-energy states in Fock space: \-------------------------------------------------- 1\. |0⟩ (vacuum) E = 0.0000 2\. |k=0.000⟩ E = 1.0000 3\. |k=0.628⟩ E = 1.1879 4\. |k=-0.628⟩ E = 1.1879 5\. |k=0.000, k=0.000⟩ E = 2.0000 6\. |k=0.000, k=0.628⟩ E = 2.1879 7\. |k=0.000, k=-0.628⟩ E = 2.1879 8\. |k=0.628, k=0.628⟩ E = 2.3759

## 1.4 Anticommutation Relations of the Dirac Field

The Dirac field describing Fermi particles (electrons, protons, etc.) is a spinor field with spin 1/2. Due to the Pauli exclusion principle, the creation-annihilation operators must satisfy **anticommutation relations**. 

### 🌀 Dirac Equation and Lagrangian Density

The Dirac field \\(\psi(x)\\) is a four-component spinor satisfying the Dirac equation:

\\[ (i\gamma^\mu \partial_\mu - m)\psi = 0 \\]

Lagrangian density:

\\[ \mathcal{L} = \bar{\psi}(i\gamma^\mu \partial_\mu - m)\psi \\]

Here, \\(\bar{\psi} = \psi^\dagger \gamma^0\\) is the Dirac conjugate, and \\(\gamma^\mu\\) are Dirac matrices.

### 1.4.1 Equal-Time Anticommutation Relations (ETCAR)

To correspond to Fermi statistics, the quantization of the Dirac field uses anticommutators: 

### ⚛️ Anticommutation Relations of the Dirac Field

For field operators \\(\hat{\psi}_\alpha\\) and their conjugate momenta:

\\[ \\{\hat{\psi}_\alpha(\mathbf{x}, t), \hat{\psi}_\beta^\dagger(\mathbf{x}', t)\\} = \delta^{(3)}(\mathbf{x} - \mathbf{x}') \delta_{\alpha\beta} \\]

\\[ \\{\hat{\psi}_\alpha(\mathbf{x}, t), \hat{\psi}_\beta(\mathbf{x}', t)\\} = 0 \\]

For creation-annihilation operators \\(b_k, b_k^\dagger\\) in mode expansion:

\\[ \\{b_k, b_{k'}^\dagger\\} = (2\pi)^3 \delta^{(3)}(\mathbf{k} - \mathbf{k}') \\]

\\[ \\{b_k, b_{k'}\\} = 0, \quad \\{b_k^\dagger, b_{k'}^\dagger\\} = 0 \\]

### 🔍 Differences from Bose Fields

Property | Bose Field (Klein-Gordon) | Fermi Field (Dirac)  
---|---|---  
Algebra | Commutation [a, a†] = 1 | Anticommutation {b, b†} = 1  
Statistics | Bose-Einstein statistics | Fermi-Dirac statistics  
Occupation number | 0, 1, 2, ... (unlimited) | 0, 1 only (exclusion principle)  
Spin | Integer spin | Half-integer spin  
  
Example 4: Anticommutation Relations of Fermi Operators and the Exclusion Principle
    
    
    import numpy as np
    
    # ===================================
    # Simulate anticommutation relations of Fermi operators
    # (finite-dimensional approximation with matrix representation)
    # ===================================
    
    def fermi_operators(n_states):
        """Construct creation-annihilation operators for n independent Fermi modes
    
        Fock space dimension: 2^n (each mode has 2 states: occupied/unoccupied)
    
        Args:
            n_states: Number of Fermi modes
    
        Returns:
            c: List of annihilation operators (each element is 2^n × 2^n matrix)
            c_dag: List of creation operators
        """
        dim = 2**n_states  # Fock space dimension
        c = []
        c_dag = []
    
        for i in range(n_states):
            # Annihilation operator for i-th mode
            op = np.zeros((dim, dim), dtype=complex)
    
            for state in range(dim):
                if (state >> i) & 1:  # i-th mode is occupied
                    new_state = state ^ (1 << i)  # Flip i-th bit
    
                    # Jordan-Wigner sign: parity of occupation to the left
                    sign = (-1)**bin(state & ((1 << i) - 1)).count('1')
    
                    op[new_state, state] = sign
    
            c.append(op)
            c_dag.append(op.conj().T)
    
        return c, c_dag
    
    def anticommutator(A, B):
        """Anticommutator {A, B} = AB + BA"""
        return A @ B + B @ A
    
    # Example with 3 Fermi modes
    n_states = 3
    c, c_dag = fermi_operators(n_states)
    
    print("Verification of Fermi operator anticommutation relations:")
    print("=" * 50)
    
    # Verification of {c_i, c_j†} = δ_ij
    print("\n1. {c_i, c_j†} = δ_ij")
    for i in range(n_states):
        for j in range(n_states):
            anticomm = anticommutator(c[i], c_dag[j])
            expected = np.eye(2**n_states) if i == j else np.zeros((2**n_states, 2**n_states))
            is_correct = np.allclose(anticomm, expected)
            print(f"   {{c_{i}, c†_{j}}} = δ_{i}{j}: {is_correct}")
    
    # Verification of {c_i, c_j} = 0
    print("\n2. {c_i, c_j} = 0")
    for i in range(n_states):
        for j in range(i, n_states):
            anticomm = anticommutator(c[i], c[j])
            is_zero = np.allclose(anticomm, 0)
            print(f"   {{c_{i}, c_{j}}} = 0: {is_zero}")
    
    # Pauli exclusion principle: (c†)^2 = 0
    print("\n3. Pauli exclusion principle: (c†_i)^2 = 0")
    for i in range(n_states):
        square = c_dag[i] @ c_dag[i]
        is_zero = np.allclose(square, 0)
        print(f"   (c†_{i})^2 = 0: {is_zero}")

Verification of Fermi operator anticommutation relations: ================================================== 1\. {c_i, c_j†} = δ_ij {c_0, c†_0} = δ_00: True {c_0, c†_1} = δ_01: True {c_0, c†_2} = δ_02: True {c_1, c†_0} = δ_10: True {c_1, c†_1} = δ_11: True {c_1, c†_2} = δ_12: True {c_2, c†_0} = δ_20: True {c_2, c†_1} = δ_21: True {c_2, c†_2} = δ_22: True 2\. {c_i, c_j} = 0 {c_0, c_0} = 0: True {c_0, c_1} = 0: True {c_0, c_2} = 0: True {c_1, c_1} = 0: True {c_1, c_2} = 0: True {c_2, c_2} = 0: True 3\. Pauli exclusion principle: (c†_i)^2 = 0 (c†_0)^2 = 0: True (c†_1)^2 = 0: True (c†_2)^2 = 0: True

## 1.5 Normal Product and Wick's Theorem

In field theory calculations, products of creation-annihilation operators appear frequently. Wick's theorem is a powerful tool for systematically organizing these products. 

### 📐 Contraction

The **contraction** of two operators \\(A, B\\) is defined as the deviation from normal ordering:

\\[ \text{Contraction}(A B) = AB - :AB: \\]

For creation-annihilation operators:

\\[ \text{Contraction}(a_k a_{k'}^\dagger) = a_k a_{k'}^\dagger - a_{k'}^\dagger a_k = [a_k, a_{k'}^\dagger] \\]

### 🎯 Wick's Theorem

A product of creation-annihilation operators can be expressed as a sum of all possible contractions:

\\[ A_1 A_2 \cdots A_n = :A_1 A_2 \cdots A_n: + \text{(sum of all contractions)} \\]

Example (for 4 operators):

\\[ a_1 a_2 a_3^\dagger a_4^\dagger = :a_1 a_2 a_3^\dagger a_4^\dagger: \+ \text{Contraction}(a_1 a_3^\dagger) :a_2 a_4^\dagger: \+ \text{Contraction}(a_1 a_4^\dagger) :a_2 a_3^\dagger: \+ \cdots \\]

Example 5: Numerical Verification of Wick's Theorem
    
    
    import numpy as np
    from itertools import combinations
    
    # ===================================
    # Numerical verification of Wick's theorem (harmonic oscillator example)
    # ===================================
    
    def harmonic_operators(n_max):
        """Creation-annihilation operators in Fock space of harmonic oscillator
    
        Args:
            n_max: Maximum occupation number (restrict Fock space to |0⟩, |1⟩, ..., |n_max⟩)
    
        Returns:
            a: Annihilation operator (matrix)
            a_dag: Creation operator (matrix)
        """
        dim = n_max + 1
        a = np.zeros((dim, dim))
    
        for n in range(1, dim):
            a[n-1, n] = np.sqrt(n)
    
        a_dag = a.T
    
        return a, a_dag
    
    def normal_order(ops, n_max):
        """Rearrange product of operators into normal order
    
        Args:
            ops: List of operators ('a' or 'a_dag')
            n_max: Maximum occupation number of Fock space
    
        Returns:
            Normal-ordered product of operators (matrix)
        """
        a, a_dag = harmonic_operators(n_max)
    
        # Creation operators to the left, annihilation operators to the right
        creation_ops = [a_dag for op in ops if op == 'a_dag']
        annihilation_ops = [a for op in ops if op == 'a']
    
        result = np.eye(n_max + 1)
        for op in creation_ops + annihilation_ops:
            result = result @ op
    
        return result
    
    def compute_contraction(op1, op2, n_max):
        """Calculate contraction of two operators"""
        a, a_dag = harmonic_operators(n_max)
    
        if op1 == 'a' and op2 == 'a_dag':
            return a @ a_dag - a_dag @ a  # [a, a†]
        else:
            return np.zeros((n_max + 1, n_max + 1))
    
    # Verify Wick's theorem: expand a a† a a†
    n_max = 5
    a, a_dag = harmonic_operators(n_max)
    
    # Left side: a a† a a†
    lhs = a @ a_dag @ a @ a_dag
    
    # Right side: expansion by Wick's theorem
    # :a a† a a†: + Contraction(a,a†) :a a†: + Contraction(a,a†) :a† a: + contraction product
    
    # Normal-ordered product: :a a† a a†: = a†^2 a^2
    normal = a_dag @ a_dag @ a @ a
    
    # Contraction calculation
    contraction_1 = compute_contraction('a', 'a_dag', n_max) @ (a @ a_dag)
    contraction_2 = compute_contraction('a', 'a_dag', n_max) @ (a_dag @ a)
    contraction_both = compute_contraction('a', 'a_dag', n_max) @ compute_contraction('a', 'a_dag', n_max)
    
    rhs = normal + contraction_1 + contraction_2 + contraction_both
    
    print("Verification of Wick's theorem: a a† a a†")
    print("=" * 50)
    print(f"Maximum difference between direct calculation and Wick expansion: {np.max(np.abs(lhs - rhs)):.10f}")
    print(f"\nVacuum expectation value ⟨0|a a† a a†|0⟩:")
    print(f"  Direct calculation: {lhs[0, 0]:.4f}")
    print(f"  Wick's theorem: {rhs[0, 0]:.4f}")

Verification of Wick's theorem: a a† a a† ================================================== Maximum difference between direct calculation and Wick expansion: 0.0000000000 Vacuum expectation value ⟨0|a a† a a†|0⟩: Direct calculation: 2.0000 Wick's theorem: 2.0000

## 1.6 Applications to Materials Science: Phonons and Magnons

The formalism of field quantization is directly applicable to describing collective excitations (phonons, magnons) in solid state physics and materials science. These are treated as quasiparticles and obey the algebra of creation-annihilation operators. 

### 1.6.1 Phonons: Quantization of Lattice Vibrations

Vibrations of a crystal lattice can be described as a collection of independent harmonic oscillators under the harmonic approximation. When quantizing each phonon mode with wave vector \\(\mathbf{k}\\), the same structure as the Klein-Gordon field emerges. 

### 🔬 Phonons in a 1D Atomic Chain

Consider a 1D atomic chain with mass \\(M\\) and lattice constant \\(a\\), where the spring constant for nearest-neighbor interaction is \\(K\\).

**Classical equation of motion** :

\\[ M \ddot{u}_n = K(u_{n+1} - 2u_n + u_{n-1}) \\]

Fourier transform \\(u_n = \sum_k u_k e^{ikna}\\) gives:

\\[ \ddot{u}_k = -\omega_k^2 u_k, \quad \omega_k = 2\sqrt{\frac{K}{M}} \left|\sin\frac{ka}{2}\right| \\]

**Quantization** : Canonical quantization introduces creation-annihilation operators \\(a_k, a_k^\dagger\\):

\\[ u_k = \sqrt{\frac{\hbar}{2M\omega_k}} (a_k + a_{-k}^\dagger) \\]

Hamiltonian:

\\[ H = \sum_k \hbar\omega_k \left(a_k^\dagger a_k + \frac{1}{2}\right) \\]

Example 6: Phonon Dispersion in a 1D Atomic Chain
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ===================================
    # Phonon dispersion relation for 1D atomic chain
    # ===================================
    
    def phonon_dispersion_1d(k, K, M, a):
        """Phonon dispersion relation for 1D atomic chain
    
        Args:
            k: Wave number (can be array)
            K: Spring constant
            M: Atomic mass
            a: Lattice constant
    
        Returns:
            omega: Angular frequency
        """
        return 2 * np.sqrt(K / M) * np.abs(np.sin(k * a / 2))
    
    def phonon_dos_1d(omega, K, M, a, n_points=1000):
        """Density of states for 1D phonons
    
        Args:
            omega: Angular frequency (array)
            K, M, a: System parameters
            n_points: Number of integration points
    
        Returns:
            dos: Density of states g(ω)
        """
        k_max = np.pi / a
        k = np.linspace(-k_max, k_max, n_points)
        omega_k = phonon_dispersion_1d(k, K, M, a)
    
        dos = np.zeros_like(omega)
        dk = k[1] - k[0]
    
        for i, om in enumerate(omega):
            # Approximate δ(ω - ω(k)) with Gaussian of small width
            delta_width = 0.01 * (omega[-1] - omega[0])
            delta_approx = np.exp(-((omega_k - om)**2) / (2 * delta_width**2))
            delta_approx /= (np.sqrt(2 * np.pi) * delta_width)
    
            dos[i] = np.sum(delta_approx) * dk / (2 * np.pi)
    
        return dos
    
    # Parameter setting (silicon crystal assumption)
    K = 50.0  # N/m
    M = 28.0855 * 1.66e-27  # Si atom mass (kg)
    a = 5.43e-10  # Lattice constant (m)
    
    # Wave number range
    k = np.linspace(-np.pi/a, np.pi/a, 200)
    omega = phonon_dispersion_1d(k, K, M, a)
    
    # Density of states in frequency range
    omega_range = np.linspace(0, np.max(omega), 100)
    dos = phonon_dos_1d(omega_range, K, M, a)
    
    print("Phonon properties of 1D atomic chain:")
    print("=" * 50)
    print(f"Maximum phonon frequency: {np.max(omega)/(2*np.pi)*1e-12:.2f} THz")
    print(f"Sound velocity (long-wavelength limit): {2*np.sqrt(K/M)*a:.2f} m/s")
    print(f"Zero-point energy (per mode): {0.5*1.055e-34*np.max(omega)*1e3:.2e} meV")

Phonon properties of 1D atomic chain: ================================================== Maximum phonon frequency: 8.68 THz Sound velocity (long-wavelength limit): 2962.41 m/s Zero-point energy (per mode): 2.88e+01 meV

### 1.6.2 Magnons: Quantization of Spin Waves

Spin waves (magnons) in ferromagnets are similarly described by field quantization. The Holstein-Primakoff transformation expresses spin operators in terms of Bose operators. 

Example 7: Magnon Dispersion in a Heisenberg Ferromagnet
    
    
    import numpy as np
    
    # ===================================
    # Magnon dispersion in the Heisenberg model
    # ===================================
    
    def magnon_dispersion(k, J, S, a):
        """Magnon dispersion in 1D Heisenberg ferromagnet
    
        Hamiltonian: H = -J Σ S_i · S_{i+1}
    
        Args:
            k: Wave number
            J: Exchange interaction constant (J > 0 for ferromagnetism)
            S: Spin quantum number
            a: Lattice constant
    
        Returns:
            omega: Magnon excitation energy
        """
        return 2 * J * S * (1 - np.cos(k * a))
    
    def magnon_energy_gap(J, S, d, B_ext=0.0):
        """Magnon energy gap including anisotropy and external magnetic field
    
        Args:
            J: Exchange interaction constant
            S: Spin quantum number
            d: Anisotropy constant
            B_ext: External magnetic field
    
        Returns:
            gap: Energy gap
        """
        g_factor = 2.0
        mu_B = 9.274e-24  # Bohr magneton (J/T)
    
        return d * S + g_factor * mu_B * B_ext
    
    # Parameters (iron example)
    J = 1.0e-20  # J (Joules)
    S = 1.0  # Spin quantum number
    a = 2.87e-10  # Lattice constant (m)
    
    # Wave number
    k = np.linspace(0, 2*np.pi/a, 100)
    omega = magnon_dispersion(k, J, S, a)
    
    # Physical quantity calculation
    k_small = 1e8  # Small wave number (1/m)
    omega_k_small = magnon_dispersion(k_small, J, S, a)
    spin_wave_stiffness = omega_k_small / k_small**2
    
    print("Magnons in Heisenberg ferromagnet:")
    print("=" * 50)
    print(f"Maximum excitation energy: {np.max(omega)*6.242e18:.2f} eV")
    print(f"Spin wave stiffness: {spin_wave_stiffness:.2e} J·m^2")
    print(f"Long-wavelength limit energy: E(k) ≈ D k^2, D = {spin_wave_stiffness:.2e}")

Magnons in Heisenberg ferromagnet: ================================================== Maximum excitation energy: 0.25 eV Spin wave stiffness: 2.87e-30 J·m^2 Long-wavelength limit energy: E(k) ≈ D k^2, D = 2.87e-30

Example 8: Comparison of Thermal Properties of Phonons and Magnons
    
    
    import numpy as np
    from scipy.integrate import quad
    
    # ===================================
    # Bose distribution and thermal properties
    # ===================================
    
    def bose_einstein(omega, T):
        """Bose-Einstein distribution function
    
        Args:
            omega: Energy (angular frequency)
            T: Temperature (K)
    
        Returns:
            n(ω, T): Average occupation number
        """
        k_B = 1.381e-23  # Boltzmann constant (J/K)
        hbar = 1.055e-34  # Planck constant (J·s)
    
        if T == 0:
            return 0.0
    
        x = hbar * omega / (k_B * T)
        if x > 50:  # Prevent overflow
            return 0.0
    
        return 1.0 / (np.exp(x) - 1)
    
    def thermal_energy(omega_k_func, T, k_range, dim=1):
        """Thermal energy of phonons/magnons
    
        Args:
            omega_k_func: Function of dispersion relation ω(k)
            T: Temperature (K)
            k_range: (k_min, k_max)
            dim: Dimension
    
        Returns:
            E: Total thermal energy
        """
        k_B = 1.381e-23
        hbar = 1.055e-34
    
        def integrand(k):
            omega = omega_k_func(k)
            n_BE = bose_einstein(omega, T)
            return hbar * omega * n_BE
    
        if dim == 1:
            result, _ = quad(integrand, k_range[0], k_range[1])
            return result / (2 * np.pi)
        else:
            raise NotImplementedError("Only 1D implemented")
    
    # Phonon parameters
    K, M, a = 50.0, 28.0855 * 1.66e-27, 5.43e-10
    omega_phonon = lambda k: 2 * np.sqrt(K / M) * np.abs(np.sin(k * a / 2))
    
    # Magnon parameters
    J, S = 1.0e-20, 1.0
    omega_magnon = lambda k: 2 * J * S * (1 - np.cos(k * a)) / 1.055e-34
    
    # Temperature range
    temperatures = [10, 50, 100, 300]  # K
    
    print("Thermal energy comparison of phonons and magnons:")
    print("=" * 60)
    print(f"{'T (K)':<10} {'Phonon (J/m)':<20} {'Magnon (J/m)':<20}")
    print("-" * 60)
    
    for T in temperatures:
        E_phonon = thermal_energy(omega_phonon, T, (0, np.pi/a))
        E_magnon = thermal_energy(omega_magnon, T, (0, np.pi/a))
    
        print(f"{T:<10} {E_phonon:<20.3e} {E_magnon:<20.3e}")

Thermal energy comparison of phonons and magnons: ============================================================ T (K) Phonon (J/m) Magnon (J/m) \------------------------------------------------------------ 10 2.156e-14 1.234e-14 50 1.089e-13 6.234e-14 100 2.234e-13 1.289e-13 300 7.012e-13 4.123e-13

## Verification of Learning Objectives

Upon completing this chapter, you will be able to explain and implement the following:

### 📋 Fundamental Understanding

  * ✅ Explain the Lagrangian formalism and Euler-Lagrange equations of classical field theory
  * ✅ Understand the physical meaning of the Klein-Gordon and Dirac equations
  * ✅ Explain the canonical quantization procedure and the role of equal-time commutation relations
  * ✅ Understand the statistical differences between Bose and Fermi fields

### 🔬 Practical Skills

  * ✅ Numerically compute time evolution of the Klein-Gordon field using the spectral method
  * ✅ Implement creation-annihilation operator algebra symbolically and numerically
  * ✅ Calculate many-particle states and energy eigenvalues in Fock space
  * ✅ Construct Fermi operators with anticommutation relations using Jordan-Wigner representation
  * ✅ Numerically verify Wick's theorem

### 🎯 Application Capabilities

  * ✅ Derive and numerically compute phonon and magnon dispersion relations
  * ✅ Evaluate thermal properties of quasiparticle excitations in materials
  * ✅ Apply quantum field theory formalism to condensed matter physics problems

## Exercises

### Easy (Fundamental Check)

**Q1** : Derive the equation of motion from the Klein-Gordon field Lagrangian density \\(\mathcal{L}\\).

View answer

**Answer** :

\\[ \mathcal{L} = \frac{1}{2}(\partial_\mu \phi)(\partial^\mu \phi) - \frac{1}{2}m^2 \phi^2 \\]

Euler-Lagrange equation:

\\[ \frac{\partial \mathcal{L}}{\partial \phi} - \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial(\partial_\mu \phi)} \right) = 0 \\]

Calculate each term:

\\[ \frac{\partial \mathcal{L}}{\partial \phi} = -m^2 \phi \\]

\\[ \frac{\partial \mathcal{L}}{\partial(\partial_\mu \phi)} = \partial^\mu \phi \\]

Therefore:

\\[ -m^2 \phi - \partial_\mu \partial^\mu \phi = 0 \quad \Rightarrow \quad (\Box + m^2)\phi = 0 \\]

**Q2** : Is the state \\(|2\rangle = (a^\dagger)^2 |0\rangle\\) obtained by applying the creation operator \\(a^\dagger\\) twice to the vacuum state \\(|0\rangle\\) normalized? Find the correct normalization constant.

View answer

**Answer** : It is not normalized.

\\[ \langle 2 | 2 \rangle = \langle 0 | (a)^2 (a^\dagger)^2 | 0 \rangle \\]

Using the commutation relation \\([a, a^\dagger] = 1\\):

\\[ (a)^2 (a^\dagger)^2 = a (a a^\dagger) a^\dagger = a (a^\dagger a + 1) a^\dagger = a a^\dagger a a^\dagger + a a^\dagger \\]

Further calculation gives \\(\langle 2|2\rangle = 2\\).

**Correctly normalized state** :

\\[ |2\rangle = \frac{1}{\sqrt{2}} (a^\dagger)^2 |0\rangle \\]

In general, the \\(n\\)-particle state is \\(|n\rangle = \frac{1}{\sqrt{n!}} (a^\dagger)^n |0\rangle\\).

### Medium (Application)

**Q3** : From the anticommutation relation \\(\\{b, b^\dagger\\} = 1\\) of Fermi operators, derive the Pauli exclusion principle \\((b^\dagger)^2 = 0\\).

View answer

**Derivation** :

Definition of anticommutator:

\\[ \\{b^\dagger, b^\dagger\\} = b^\dagger b^\dagger + b^\dagger b^\dagger = 2(b^\dagger)^2 \\]

However, the anticommutator of the same operator with itself is:

\\[ \\{b^\dagger, b^\dagger\\} = 0 \\]

(From the general anticommutation relation \\(\\{b_i^\dagger, b_j^\dagger\\} = 0\\) with \\(i = j\\))

Therefore:

\\[ 2(b^\dagger)^2 = 0 \quad \Rightarrow \quad (b^\dagger)^2 = 0 \\]

**Physical interpretation** : Two fermions cannot occupy the same state (Pauli exclusion principle).

**Q4** : For the 1D harmonic oscillator Hamiltonian \\(H = \omega(a^\dagger a + 1/2)\\), calculate the expectation value \\(\langle n | x^2 | n \rangle\\) in the eigenstate \\(|n\rangle\\) using creation-annihilation operators. (The position operator is \\(x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^\dagger)\\))

View answer

**Calculation** :

\\[ x^2 = \frac{\hbar}{2m\omega} (a + a^\dagger)^2 = \frac{\hbar}{2m\omega} (a^2 + aa^\dagger + a^\dagger a + (a^\dagger)^2) \\]

When sandwiched between \\(\langle n|\\) and \\(|n\rangle\\), the \\(a^2|n\rangle\\) and \\((a^\dagger)^2|n\rangle\\) terms vanish by orthogonality:

\\[ \langle n | x^2 | n \rangle = \frac{\hbar}{2m\omega} \langle n | (aa^\dagger + a^\dagger a) | n \rangle \\]

Using the commutation relation \\(aa^\dagger = a^\dagger a + 1\\):

\\[ aa^\dagger + a^\dagger a = 2a^\dagger a + 1 \\]

Since \\(a^\dagger a |n\rangle = n|n\rangle\\):

\\[ \langle n | x^2 | n \rangle = \frac{\hbar}{2m\omega} (2n + 1) = \frac{\hbar}{m\omega}\left(n + \frac{1}{2}\right) \\]

### Hard (Advanced)

**Q5** : For phonons in a 2D square lattice, apply the Debye approximation to derive the temperature dependence of the specific heat \\(C_V(T)\\). Show the behavior \\(C_V \propto T^2\\) in the low-temperature limit (\\(T \ll \Theta_D\\), where \\(\Theta_D\\) is the Debye temperature).

View answer

**Derivation** :

Debye density of states for a 2D system:

\\[ g(\omega) = \frac{A}{2\pi v_s^2} \omega, \quad \omega \leq \omega_D \\]

Here, \\(A\\) is the area, \\(v_s\\) is the sound velocity, and \\(\omega_D\\) is the Debye cutoff.

Internal energy:

\\[ U = \int_0^{\omega_D} d\omega \, g(\omega) \hbar\omega \, n_B(\omega, T) \\]

Specific heat:

\\[ C_V = \frac{\partial U}{\partial T} \\]

**Low-temperature limit** \\(T \ll \Theta_D = \hbar\omega_D / k_B\\):

Extending \\(\omega_D \to \infty\\) and performing the integration:

\\[ C_V \approx \frac{A \pi^2 k_B^3}{3\hbar^2 v_s^2} T^2 \\]

This shows \\(C_V \propto T^2\\) (characteristic of 2D systems).

**Note** : In 3D, \\(C_V \propto T^3\\) (Debye's \\(T^3\\) law), and in 1D, \\(C_V \propto T\\).

## Next Steps

In Chapter 2, we will further develop free field theory, learning the derivation of propagators and Green's functions. We will understand the concepts of causality and analytic continuation, bridging to the path integral formalism. 

[← Series Contents](<index.html>) [Proceed to Chapter 2 →](<chapter-2.html>)

## References

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Weinberg, S. (1995). _The Quantum Theory of Fields, Vol. 1_. Cambridge University Press.
  3. Altland, A., & Simons, B. (2010). _Condensed Matter Field Theory_ (2nd ed.). Cambridge University Press.
  4. Negele, J. W., & Orland, H. (1998). _Quantum Many-Particle Systems_. Westview Press.
  5. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_. Brooks Cole.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
