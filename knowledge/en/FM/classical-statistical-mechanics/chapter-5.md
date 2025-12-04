---
title: "Chapter 5: Statistical Mechanics Simulations"
chapter_title: "Chapter 5: Statistical Mechanics Simulations"
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/classical-statistical-mechanics/chapter-5.html>) | Last sync: 2025-11-16

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Classical Statistical Mechanics](<index.html>) > Chapter 5 

## 🎯 Learning Objectives

  * Understand the principles of the Monte Carlo method and importance sampling
  * Implement the Metropolis algorithm
  * Execute simulations of the 2D Ising model
  * Understand the concepts of ergodicity and equilibration
  * Evaluate statistical errors of physical quantities
  * Learn the basics of the Molecular Dynamics method (Verlet algorithm)
  * Understand Lennard-Jones potential and radial distribution function
  * Practice applications to materials science (adsorption, magnetism)

## 📖 Fundamentals of the Monte Carlo Method

### Monte Carlo Method

Generate samples \\(\\{x_i\\}\\) following a probability distribution \\(P(x)\\) and calculate expectation values:

\\[ \langle f \rangle = \int f(x) P(x) dx \approx \frac{1}{N} \sum_{i=1}^N f(x_i) \\]

**Importance Sampling** :

Sample from the canonical distribution \\(P(E) = e^{-\beta E} / Z\\) to calculate thermodynamic quantities.

**Markov Chain Monte Carlo (MCMC)** : This method defines a transition probability \\(W_{i \to j}\\) from state \\(i\\) to state \\(j\\). The detailed balance condition \\(P_i W_{i \to j} = P_j W_{j \to i}\\) ensures that the chain converges to the equilibrium distribution.

### Metropolis Algorithm

Sampling method for the canonical ensemble:

  1. Propose a new state \\(j\\) randomly from current state \\(i\\)
  2. Calculate energy difference \\(\Delta E = E_j - E_i\\)
  3. Transition probability \\(W_{i \to j} = \min(1, e^{-\beta \Delta E})\\)
  4. Accept state \\(j\\) with probability \\(W\\), reject with probability \\(1 - W\\)

**Characteristics** : When \\(\Delta E < 0\\), the move is always accepted since it lowers the energy. When \\(\Delta E > 0\\), the move is accepted with probability \\(e^{-\beta \Delta E}\\). This acceptance criterion satisfies the detailed balance condition.

## 💻 Example 5.1: Monte Carlo Simulation of the 2D Ising Model

### 2D Ising Model

Spin system on an \\(L \times L\\) lattice:

\\[ H = -J \sum_{\langle i,j \rangle} s_i s_j \\]

Periodic boundary conditions are applied. Sample the canonical distribution using the Metropolis algorithm.

**Physical Quantities Measured** : The **magnetization** is \\(m = \frac{1}{N}\sum_i s_i\\), the **energy** is \\(E = -J \sum_{\langle i,j \rangle} s_i s_j\\), the **specific heat** is \\(C = \beta^2 (\langle E^2 \rangle - \langle E \rangle^2)\\), and the **magnetic susceptibility** is \\(\chi = \beta N (\langle m^2 \rangle - \langle m \rangle^2)\\).

Python Implementation: 2D Ising Model
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    class Ising2D:
        def __init__(self, L, T, J=1.0):
            """
            2D Ising Model
            L: Lattice size (L x L)
            T: Temperature
            J: Interaction strength
            """
            self.L = L
            self.N = L * L
            self.T = T
            self.J = J
            self.beta = 1.0 / T if T > 0 else np.inf
    
            # Random initial configuration
            self.spins = np.random.choice([-1, 1], size=(L, L))
    
            # Statistics
            self.energy_history = []
            self.magnetization_history = []
    
        def energy(self):
            """Total energy of the system"""
            E = 0
            for i in range(self.L):
                for j in range(self.L):
                    # Interaction with nearest neighbors (periodic boundary conditions)
                    S = self.spins[i, j]
                    neighbors = (
                        self.spins[(i+1) % self.L, j] +
                        self.spins[i, (j+1) % self.L]
                    )
                    E += -self.J * S * neighbors
            return E
    
        def delta_energy(self, i, j):
            """Energy change when flipping spin (i,j)"""
            S = self.spins[i, j]
            neighbors = (
                self.spins[(i+1) % self.L, j] +
                self.spins[(i-1) % self.L, j] +
                self.spins[i, (j+1) % self.L] +
                self.spins[i, (j-1) % self.L]
            )
            return 2 * self.J * S * neighbors
    
        def magnetization(self):
            """Magnetization"""
            return np.abs(np.sum(self.spins)) / self.N
    
        def metropolis_step(self):
            """One Metropolis step (attempt each spin once)"""
            for _ in range(self.N):
                # Randomly select a spin
                i, j = np.random.randint(0, self.L, size=2)
    
                # Energy change
                dE = self.delta_energy(i, j)
    
                # Metropolis criterion
                if dE < 0 or np.random.random() < np.exp(-self.beta * dE):
                    self.spins[i, j] *= -1  # Flip spin
    
        def simulate(self, steps, equilibration=1000):
            """Execute simulation"""
            # Equilibration
            for _ in range(equilibration):
                self.metropolis_step()
    
            # Measurement
            for step in range(steps):
                self.metropolis_step()
    
                if step % 10 == 0:  # Record every 10 steps
                    self.energy_history.append(self.energy())
                    self.magnetization_history.append(self.magnetization())
    
        def statistics(self):
            """Calculate statistics"""
            E_mean = np.mean(self.energy_history)
            E_std = np.std(self.energy_history)
            M_mean = np.mean(self.magnetization_history)
            M_std = np.std(self.magnetization_history)
    
            # Specific heat and susceptibility
            E_array = np.array(self.energy_history)
            M_array = np.array(self.magnetization_history)
    
            C = self.beta**2 * (np.mean(E_array**2) - np.mean(E_array)**2) / self.N
            chi = self.beta * self.N * (np.mean(M_array**2) - np.mean(M_array)**2)
    
            return {
                'E_mean': E_mean / self.N,
                'E_std': E_std / self.N,
                'M_mean': M_mean,
                'M_std': M_std,
                'C': C,
                'chi': chi
            }
    
    # Simulate at different temperatures
    L = 20
    T_range = np.linspace(1.0, 4.0, 16)
    J = 1.0
    T_c_onsager = 2 * J / np.log(1 + np.sqrt(2))  # Onsager analytical solution
    
    results = []
    
    for T in T_range:
        print(f"Temperature T = {T:.2f}...")
        ising = Ising2D(L, T, J)
        ising.simulate(steps=5000, equilibration=1000)
        stats = ising.statistics()
        stats['T'] = T
        results.append(stats)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy
    ax1 = axes[0, 0]
    T_vals = [r['T'] for r in results]
    E_vals = [r['E_mean'] for r in results]
    E_err = [r['E_std'] for r in results]
    ax1.errorbar(T_vals, E_vals, yerr=E_err, fmt='o-', linewidth=2, markersize=6)
    ax1.axvline(T_c_onsager, color='r', linestyle='--', linewidth=2, label=f'T_c = {T_c_onsager:.2f}')
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel('Energy per spin ⟨E⟩/N')
    ax1.set_title('Temperature Dependence of Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Magnetization
    ax2 = axes[0, 1]
    M_vals = [r['M_mean'] for r in results]
    M_err = [r['M_std'] for r in results]
    ax2.errorbar(T_vals, M_vals, yerr=M_err, fmt='o-', linewidth=2, markersize=6, color='blue')
    ax2.axvline(T_c_onsager, color='r', linestyle='--', linewidth=2, label=f'T_c = {T_c_onsager:.2f}')
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('Magnetization ⟨|m|⟩')
    ax2.set_title('Temperature Dependence of Magnetization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Specific heat
    ax3 = axes[1, 0]
    C_vals = [r['C'] for r in results]
    ax3.plot(T_vals, C_vals, 'o-', linewidth=2, markersize=6, color='green')
    ax3.axvline(T_c_onsager, color='r', linestyle='--', linewidth=2, label=f'T_c = {T_c_onsager:.2f}')
    ax3.set_xlabel('Temperature T')
    ax3.set_ylabel('Specific heat C')
    ax3.set_title('Specific Heat (Peak at T_c)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Susceptibility
    ax4 = axes[1, 1]
    chi_vals = [r['chi'] for r in results]
    ax4.plot(T_vals, chi_vals, 'o-', linewidth=2, markersize=6, color='purple')
    ax4.axvline(T_c_onsager, color='r', linestyle='--', linewidth=2, label=f'T_c = {T_c_onsager:.2f}')
    ax4.set_xlabel('Temperature T')
    ax4.set_ylabel('Susceptibility χ')
    ax4.set_title('Susceptibility (Peak at T_c)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_ising_mc_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Numerical results
    print("\n=== 2D Ising Model Monte Carlo Simulation ===\n")
    print(f"Lattice size: {L} × {L}")
    print(f"Onsager critical temperature: T_c = {T_c_onsager:.4f}\n")
    
    print("Temperature dependence:")
    for r in results:
        print(f"T = {r['T']:.2f}: E/N = {r['E_mean']:.3f}, M = {r['M_mean']:.3f}, C = {r['C']:.3f}, χ = {r['chi']:.2f}")
    

## 💻 Example 5.2: Autocorrelation and Ergodicity

### Autocorrelation Function

Autocorrelation of physical quantity \\(A(t)\\):

\\[ C(t) = \langle A(t_0) A(t_0 + t) \rangle - \langle A \rangle^2 \\]

Correlation time \\(\tau\\): Decay constant where \\(C(t) \sim e^{-t/\tau}\\).

**Statistical Error** :

\\[ \sigma_{\bar{A}} = \frac{\sigma_A}{\sqrt{N_{\text{eff}}}}, \quad N_{\text{eff}} = \frac{N}{2\tau + 1} \\]

For correlated data, the effective sample size decreases.

Python Implementation: Autocorrelation and Statistical Errors
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def autocorrelation(data):
        """Calculate autocorrelation function"""
        n = len(data)
        mean = np.mean(data)
        variance = np.var(data)
    
        # Center to zero mean
        data_centered = data - mean
    
        # Autocorrelation
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[n-1:] / (variance * np.arange(n, 0, -1))
    
        return autocorr
    
    def correlation_time(autocorr):
        """Estimate correlation time (exponential decay fit)"""
        # Find first zero crossing or point where sufficiently small
        for i, val in enumerate(autocorr):
            if val < np.exp(-1) or val < 0:
                return i
        return len(autocorr) // 2
    
    # Autocorrelation in 2D Ising model
    L = 20
    temperatures = [1.5, T_c_onsager, 3.0]
    colors = ['blue', 'red', 'green']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Autocorrelation at different temperatures
    ax1 = axes[0, 0]
    
    for T, color in zip(temperatures, colors):
        ising = Ising2D(L, T, J=1.0)
        ising.simulate(steps=10000, equilibration=1000)
    
        # Magnetization autocorrelation
        autocorr = autocorrelation(np.array(ising.magnetization_history))
        tau = correlation_time(autocorr)
    
        ax1.plot(autocorr[:100], color=color, linewidth=2,
                 label=f'T = {T:.2f}, τ ≈ {tau}')
    
    ax1.set_xlabel('Time lag (MC steps)')
    ax1.set_ylabel('Autocorrelation C(t)')
    ax1.set_title('Magnetization Autocorrelation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Verification of ergodicity (time average vs ensemble average)
    ax2 = axes[0, 1]
    T_test = 2.5
    num_runs = 20
    
    time_averages = []
    for run in range(num_runs):
        ising = Ising2D(L, T_test, J=1.0)
        ising.simulate(steps=5000, equilibration=1000)
        time_avg = np.mean(ising.magnetization_history)
        time_averages.append(time_avg)
    
    ax2.hist(time_averages, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(time_averages), color='r', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(time_averages):.3f}')
    ax2.set_xlabel('Time-averaged magnetization')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Ergodicity Verification (T = {T_test}, {num_runs} runs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Statistical error evaluation
    ax3 = axes[1, 0]
    ising_err = Ising2D(L, T_c_onsager, J=1.0)
    ising_err.simulate(steps=10000, equilibration=1000)
    
    M_data = np.array(ising_err.magnetization_history)
    autocorr_err = autocorrelation(M_data)
    tau_err = correlation_time(autocorr_err)
    
    # Block averaging method
    block_sizes = np.logspace(0, 3, 20, dtype=int)
    block_errors = []
    
    for block_size in block_sizes:
        if block_size > len(M_data) // 2:
            break
    
        n_blocks = len(M_data) // block_size
        blocks = M_data[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = np.mean(blocks, axis=1)
        block_error = np.std(block_means) / np.sqrt(n_blocks)
        block_errors.append(block_error)
    
    ax3.semilogx(block_sizes[:len(block_errors)], block_errors, 'o-', linewidth=2, markersize=6)
    ax3.axhline(block_errors[-1], color='r', linestyle='--', linewidth=2,
                label=f'Converged error ≈ {block_errors[-1]:.4f}')
    ax3.set_xlabel('Block size')
    ax3.set_ylabel('Standard error')
    ax3.set_title('Statistical Error Evaluation by Block Averaging')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Equilibration process
    ax4 = axes[1, 1]
    ising_eq = Ising2D(L, 2.0, J=1.0)
    
    # Start from fully ordered state
    ising_eq.spins = np.ones((L, L))
    
    equilibration_M = []
    for step in range(2000):
        ising_eq.metropolis_step()
        if step % 10 == 0:
            equilibration_M.append(ising_eq.magnetization())
    
    ax4.plot(np.arange(len(equilibration_M)) * 10, equilibration_M, 'b-', linewidth=2)
    ax4.axhline(np.mean(equilibration_M[-50:]), color='r', linestyle='--', linewidth=2,
                label='Equilibrium value')
    ax4.set_xlabel('MC steps')
    ax4.set_ylabel('Magnetization')
    ax4.set_title('Equilibration Process (Order→Disorder)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_mc_autocorrelation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Autocorrelation and Statistical Errors ===\n")
    print(f"Correlation time τ ≈ {tau_err} MC steps")
    print(f"Effective sample size N_eff = N / (2τ + 1) = {len(M_data)} / {2*tau_err + 1} ≈ {len(M_data) / (2*tau_err + 1):.0f}")
    print(f"\nStatistical error (block averaging): σ ≈ {block_errors[-1]:.4f}")
    

## 💻 Example 5.3: Fundamentals of Molecular Dynamics

### Molecular Dynamics Method

Numerical integration of Newton's equations of motion:

\\[ m \frac{d^2 \mathbf{r}_i}{dt^2} = \mathbf{F}_i = -\nabla_i U(\\{\mathbf{r}_j\\}) \\]

**Velocity Verlet Method** :

  1. \\(\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t) \Delta t + \frac{1}{2}\mathbf{a}(t) \Delta t^2\\)
  2. \\(\mathbf{a}(t + \Delta t) = \mathbf{F}(\mathbf{r}(t + \Delta t)) / m\\)
  3. \\(\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \frac{1}{2}[\mathbf{a}(t) + \mathbf{a}(t + \Delta t)] \Delta t\\)

**Lennard-Jones Potential** :

\\[ U(r) = 4\varepsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right] \\]

Here, \\(\varepsilon\\) is the depth of the potential well and \\(\sigma\\) is the zero-crossing distance.

Python Implementation: Lennard-Jones Molecular Dynamics
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LennardJonesMD:
        def __init__(self, N, L, T, dt=0.001, epsilon=1.0, sigma=1.0):
            """
            Lennard-Jones Molecular Dynamics
            N: Number of particles
            L: Box size
            T: Temperature
            dt: Time step
            """
            self.N = N
            self.L = L
            self.T = T
            self.dt = dt
            self.epsilon = epsilon
            self.sigma = sigma
    
            # Initial configuration (FCC lattice)
            n = int(np.ceil(N**(1/3)))
            positions = []
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        if len(positions) < N:
                            positions.append([i, j, k])
    
            self.positions = np.array(positions, dtype=float) * (L / n)
    
            # Initial velocities (Maxwell distribution)
            self.velocities = np.random.randn(N, 3) * np.sqrt(T)
            # Zero center-of-mass momentum
            self.velocities -= np.mean(self.velocities, axis=0)
    
            # History
            self.energy_history = []
            self.temperature_history = []
    
        def apply_pbc(self, r):
            """Periodic boundary conditions"""
            return r - self.L * np.floor(r / self.L)
    
        def lennard_jones(self, r):
            """Lennard-Jones potential and force"""
            r_norm = np.linalg.norm(r)
            if r_norm < 0.1 * self.sigma:
                r_norm = 0.1 * self.sigma  # Avoid divergence
    
            sr6 = (self.sigma / r_norm)**6
            potential = 4 * self.epsilon * (sr6**2 - sr6)
            force_magnitude = 24 * self.epsilon * (2 * sr6**2 - sr6) / r_norm
            force = force_magnitude * r / r_norm
    
            return potential, force
    
        def compute_forces(self):
            """Forces and potential energy between all particles"""
            forces = np.zeros((self.N, 3))
            potential = 0
    
            for i in range(self.N):
                for j in range(i+1, self.N):
                    r_ij = self.positions[i] - self.positions[j]
                    r_ij = self.apply_pbc(r_ij)
    
                    U_ij, F_ij = self.lennard_jones(r_ij)
    
                    forces[i] += F_ij
                    forces[j] -= F_ij
                    potential += U_ij
    
            return forces, potential
    
        def velocity_verlet_step(self):
            """Velocity Verlet algorithm"""
            # Current forces
            forces, potential = self.compute_forces()
    
            # Position update
            self.positions += self.velocities * self.dt + 0.5 * forces * self.dt**2
            self.positions = self.apply_pbc(self.positions)
    
            # New forces
            new_forces, new_potential = self.compute_forces()
    
            # Velocity update
            self.velocities += 0.5 * (forces + new_forces) * self.dt
    
            # Energy and temperature
            kinetic = 0.5 * np.sum(self.velocities**2)
            total_energy = kinetic + new_potential
            temperature = 2 * kinetic / (3 * self.N)
    
            return total_energy, temperature
    
        def simulate(self, steps):
            """Execute simulation"""
            for step in range(steps):
                energy, temp = self.velocity_verlet_step()
    
                if step % 10 == 0:
                    self.energy_history.append(energy)
                    self.temperature_history.append(temp)
    
        def radial_distribution(self, bins=100):
            """Radial distribution function g(r)"""
            r_max = self.L / 2
            hist, bin_edges = np.histogram([], bins=bins, range=(0, r_max))
    
            for i in range(self.N):
                for j in range(i+1, self.N):
                    r_ij = self.positions[i] - self.positions[j]
                    r_ij = self.apply_pbc(r_ij)
                    r = np.linalg.norm(r_ij)
    
                    if r < r_max:
                        bin_idx = int(r / r_max * bins)
                        if bin_idx < bins:
                            hist[bin_idx] += 1
    
            # Normalization
            r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
            dr = r_bins[1] - r_bins[0]
            shell_volume = 4 * np.pi * r_bins**2 * dr
            density = self.N / self.L**3
    
            g_r = hist / (shell_volume * density * self.N)
    
            return r_bins, g_r
    
    # Lennard-Jones MD simulation
    N = 64
    L = 5.0
    T = 1.0
    dt = 0.001
    steps = 5000
    
    md = LennardJonesMD(N, L, T, dt)
    md.simulate(steps)
    
    # Radial distribution function
    r_bins, g_r = md.radial_distribution(bins=50)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy conservation
    ax1 = axes[0, 0]
    time = np.arange(len(md.energy_history)) * 10 * dt
    ax1.plot(time, md.energy_history, 'b-', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Total Energy')
    ax1.set_title('Energy Conservation (Verlet Method)')
    ax1.grid(True, alpha=0.3)
    
    # Temperature fluctuations
    ax2 = axes[0, 1]
    ax2.plot(time, md.temperature_history, 'r-', linewidth=1)
    ax2.axhline(T, color='k', linestyle='--', linewidth=2, label=f'T_target = {T}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Time Evolution of Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Radial distribution function
    ax3 = axes[1, 0]
    ax3.plot(r_bins, g_r, 'g-', linewidth=2)
    ax3.set_xlabel('r / σ')
    ax3.set_ylabel('g(r)')
    ax3.set_title('Radial Distribution Function')
    ax3.grid(True, alpha=0.3)
    
    # Particle configuration (2D projection)
    ax4 = axes[1, 1]
    ax4.scatter(md.positions[:, 0], md.positions[:, 1], s=100, alpha=0.6, c='blue', edgecolors='black')
    ax4.set_xlim([0, L])
    ax4.set_ylim([0, L])
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Particle Configuration (XY Plane)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_lennard_jones_md.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Lennard-Jones Molecular Dynamics ===\n")
    print(f"Number of particles: {N}")
    print(f"Box size: {L:.2f}")
    print(f"Time step: {dt}")
    print(f"Total steps: {steps}")
    print(f"\nAverage energy: {np.mean(md.energy_history):.4f}")
    print(f"Average temperature: {np.mean(md.temperature_history):.4f}")
    print(f"Temperature standard deviation: {np.std(md.temperature_history):.4f}")
    

## 💻 Example 5.4: Materials Science Application - Adsorption Simulation

### Grand Canonical Monte Carlo (GCMC)

Monte Carlo method in the grand canonical ensemble: **Particle insertion** is accepted with probability \\(\min(1, \frac{V}{N+1} e^{\beta\mu} e^{-\beta \Delta E})\\), **particle deletion** is accepted with probability \\(\min(1, \frac{N}{V} e^{-\beta\mu} e^{-\beta \Delta E})\\), and **particle movement** follows the standard Metropolis criterion.

Can simulate gas adsorption onto materials.

Python Implementation: GCMC Adsorption Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class GCMCAdsorption:
        def __init__(self, L, mu, T, epsilon_wall=-1.0, sigma=1.0):
            """
            Grand Canonical Monte Carlo Adsorption Simulation
            L: Box size
            mu: Chemical potential
            T: Temperature
            epsilon_wall: Interaction with wall
            """
            self.L = L
            self.mu = mu
            self.T = T
            self.beta = 1.0 / T
            self.epsilon_wall = epsilon_wall
            self.sigma = sigma
    
            self.particles = []  # List of particle positions
            self.N_history = []
    
        def wall_energy(self, z):
            """Interaction energy with wall (z direction)"""
            # Simple Lennard-Jones type wall potential
            if z < self.sigma:
                z = self.sigma
    
            E = self.epsilon_wall * ((self.sigma / z)**12 - (self.sigma / z)**6)
            return E
    
        def particle_energy(self, pos):
            """Particle energy (wall interaction only)"""
            z = pos[2]
            return self.wall_energy(z)
    
        def insert_particle(self):
            """Particle insertion attempt"""
            # Random position
            new_pos = np.random.rand(3) * self.L
    
            # Energy calculation
            E_new = self.particle_energy(new_pos)
    
            # Acceptance probability
            N = len(self.particles)
            acceptance = min(1, (self.L**3 / (N + 1)) * np.exp(self.beta * (self.mu - E_new)))
    
            if np.random.random() < acceptance:
                self.particles.append(new_pos)
                return True
            return False
    
        def delete_particle(self):
            """Particle deletion attempt"""
            if len(self.particles) == 0:
                return False
    
            # Randomly select particle
            idx = np.random.randint(len(self.particles))
            pos = self.particles[idx]
    
            # Energy calculation
            E_old = self.particle_energy(pos)
    
            # Acceptance probability
            N = len(self.particles)
            acceptance = min(1, (N / self.L**3) * np.exp(-self.beta * (self.mu - E_old)))
    
            if np.random.random() < acceptance:
                del self.particles[idx]
                return True
            return False
    
        def move_particle(self):
            """Particle movement attempt"""
            if len(self.particles) == 0:
                return False
    
            idx = np.random.randint(len(self.particles))
            old_pos = self.particles[idx].copy()
    
            # Random displacement
            displacement = (np.random.rand(3) - 0.5) * 0.5
            new_pos = old_pos + displacement
    
            # Periodic boundary conditions
            new_pos = new_pos % self.L
    
            # Energy difference
            dE = self.particle_energy(new_pos) - self.particle_energy(old_pos)
    
            # Metropolis criterion
            if dE < 0 or np.random.random() < np.exp(-self.beta * dE):
                self.particles[idx] = new_pos
                return True
            return False
    
        def gcmc_step(self):
            """One GCMC step"""
            # Randomly select operation
            operation = np.random.choice(['insert', 'delete', 'move'], p=[0.33, 0.33, 0.34])
    
            if operation == 'insert':
                self.insert_particle()
            elif operation == 'delete':
                self.delete_particle()
            else:
                self.move_particle()
    
        def simulate(self, steps, record_interval=10):
            """Execute simulation"""
            for step in range(steps):
                self.gcmc_step()
    
                if step % record_interval == 0:
                    self.N_history.append(len(self.particles))
    
        def density_profile(self, bins=50):
            """Density profile (z direction)"""
            if len(self.particles) == 0:
                return np.linspace(0, self.L, bins), np.zeros(bins)
    
            z_coords = [p[2] for p in self.particles]
            hist, bin_edges = np.histogram(z_coords, bins=bins, range=(0, self.L))
            z_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
            # Density (particles/volume)
            bin_volume = self.L**2 * (self.L / bins)
            density = hist / bin_volume
    
            return z_bins, density
    
    # GCMC at different chemical potentials
    L = 10.0
    T = 1.0
    epsilon_wall = -2.0
    mu_values = [-5.0, -4.0, -3.0, -2.0]
    
    results_gcmc = []
    
    for mu in mu_values:
        print(f"Chemical potential μ = {mu:.2f}...")
        gcmc = GCMCAdsorption(L, mu, T, epsilon_wall)
        gcmc.simulate(steps=10000, record_interval=10)
    
        z_bins, density = gcmc.density_profile(bins=50)
    
        results_gcmc.append({
            'mu': mu,
            'N_avg': np.mean(gcmc.N_history[-100:]),
            'N_history': gcmc.N_history,
            'z_bins': z_bins,
            'density': density
        })
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time evolution of particle number
    ax1 = axes[0, 0]
    colors = ['blue', 'green', 'orange', 'red']
    for r, color in zip(results_gcmc, colors):
        ax1.plot(r['N_history'], color=color, linewidth=1, alpha=0.7,
                 label=f"μ = {r['mu']:.1f}")
    ax1.set_xlabel('MC steps')
    ax1.set_ylabel('Number of particles N')
    ax1.set_title('Time Evolution of Particle Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Adsorption isotherm
    ax2 = axes[0, 1]
    mu_vals = [r['mu'] for r in results_gcmc]
    N_avg_vals = [r['N_avg'] for r in results_gcmc]
    ax2.plot(mu_vals, N_avg_vals, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Chemical potential μ')
    ax2.set_ylabel('Average number of particles ⟨N⟩')
    ax2.set_title('Adsorption Isotherm (GCMC)')
    ax2.grid(True, alpha=0.3)
    
    # Density profile
    ax3 = axes[1, 0]
    for r, color in zip(results_gcmc, colors):
        ax3.plot(r['z_bins'], r['density'], color=color, linewidth=2,
                 label=f"μ = {r['mu']:.1f}")
    ax3.set_xlabel('z (distance from wall)')
    ax3.set_ylabel('Density ρ(z)')
    ax3.set_title('Density Profile (Adsorption Near Wall)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Particle number distribution
    ax4 = axes[1, 1]
    for r, color in zip(results_gcmc, colors):
        ax4.hist(r['N_history'][-500:], bins=20, alpha=0.5, color=color,
                 label=f"μ = {r['mu']:.1f}, ⟨N⟩ = {r['N_avg']:.1f}")
    ax4.set_xlabel('Number of particles N')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Particle Number Distribution (Fluctuations)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_gcmc_adsorption.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== GCMC Adsorption Simulation ===\n")
    print("Results:")
    for r in results_gcmc:
        print(f"μ = {r['mu']:.2f}: ⟨N⟩ = {r['N_avg']:.2f}")
    

## 📚 Summary

1\. The **Monte Carlo method** is a technique to calculate equilibrium properties by sampling from probability distributions.

2\. The **Metropolis algorithm** satisfies the detailed balance condition and samples the canonical distribution.

3\. **2D Ising model** MC simulations enable observation of phase transitions and critical phenomena.

4\. **Autocorrelation** and correlation time are important for statistical error evaluation, determining the number of independent samples.

5\. **Ergodicity** ensures equivalence of time averages and ensemble averages.

6\. The **molecular dynamics method** numerically integrates Newton's equations to calculate dynamic properties.

7\. The **Velocity Verlet method** is a time integration scheme with excellent energy conservation.

8\. The **Lennard-Jones potential** is the standard model for real gases and liquids.

9\. **GCMC** is effective for simulating adsorption and chemical reaction systems with variable particle numbers.

10\. Statistical mechanics simulations are essential tools for structure and property prediction in materials science.

### 💡 Exercise Problems

  1. **Wolff algorithm** : Implement a cluster algorithm that avoids critical slowing down and compare efficiency near the critical point.
  2. **3D Ising model** : Simulate the 3D Ising model and determine the critical temperature and critical exponents.
  3. **Andersen thermostat** : Implement the Andersen thermostat in Lennard-Jones MD and verify temperature control effectiveness.
  4. **Radial distribution function and structure factor** : Calculate the static structure factor S(q) from g(r) and analyze liquid structure.
  5. **Multi-component adsorption** : Simulate competitive adsorption of two gas species using GCMC and investigate selective adsorption.
  6. **Magnetic materials** : Simulate the Heisenberg spin model (classical 3-component spins) and calculate magnetization curves.

## 🎓 Series Completion

Congratulations! You have completed the **Introduction to Classical Statistical Mechanics** series. So far, you have learned the three statistical ensembles of statistical mechanics (microcanonical, canonical, grand canonical), derivation of thermodynamic quantities from partition functions, phase transition theory, and fundamentals of computational statistical mechanics. 

**Next Steps** : For further study, consider exploring **Quantum Statistical Mechanics** for details of Fermi-Dirac and Bose-Einstein statistics and superconductivity theory (BCS theory), **Non-equilibrium Statistical Mechanics** covering the Boltzmann equation, linear response theory, and the fluctuation-dissipation theorem, **Advanced Computational Statistical Mechanics** including path integral Monte Carlo, quantum Monte Carlo, and first-principles molecular dynamics, and **Materials Science Applications** such as phase diagram calculations, combination with first-principles calculations, and machine learning potentials. 

Statistical mechanics is the theoretical foundation of materials science. Please apply the concepts and simulation methods learned here to your actual research! 

[← Chapter 4: Interacting Systems and Phase Transitions](<chapter-4.html>) [Return to Series Index →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
