---
title: "Chapter 6: Thermodynamics and Equations of State"
chapter_title: "Chapter 6: Thermodynamics and Equations of State"
subtitle: Cubic Equations of State, Critical Phenomena, Phase Equilibrium, and Fugacity
reading_time: 30-35 minutes
difficulty: Advanced
code_examples: 8
---

Chapters 1-5 described supercritical fluids qualitatively. This chapter supplies the quantitative machinery: the equations of state that predict density and compressibility, the scaling laws that govern the anomalies near the critical point, the phase-equilibrium criteria that decide what dissolves in what, and the fugacity relations that turn all of this into numbers. Eight worked Python examples let you reproduce every result yourself.

## Learning Objectives

After completing this chapter, you will be able to:

  * Explain why the ideal gas law fails catastrophically near the critical point, and quantify the deviation with the compressibility factor $Z$
  * Interpret the physical meaning of the van der Waals parameters $a$ and $b$, and derive the critical constants and $Z_c = 3/8$ from them
  * Compute densities and compressibility factors with the Peng-Robinson equation of state, including the role of the acentric factor $\omega$
  * Describe critical opalescence and the divergence of $\kappa_T$ and $C_P$ in terms of universal critical exponents
  * Explain why the speed of sound passes through a minimum at the critical point, and what this means for process control
  * Classify binary phase diagrams (Type I, II, III) and model solid solubility with the Chrastil equation
  * Calculate fugacity coefficients from a cubic EOS and apply the phase-equilibrium criterion $\phi_i^L x_i = \phi_i^V y_i$
  * Apply van der Waals mixing rules with a binary interaction parameter $k_{ij}$ to multicomponent systems

* * *

## 6.1 Equations of State for Supercritical Fluids

### Limitations of the Ideal Gas Law

The ideal gas law

$$ PV = nRT $$ 

rests on three assumptions: molecules are point particles of negligible volume, they exert no forces on one another, and all collisions are elastic. Each of these fails in the conditions where supercritical fluids are actually used:

  1. **High pressure** : intermolecular distances shrink until attractive (van der Waals) forces are no longer negligible.
  2. **High density** : the excluded volume of the molecules themselves becomes a significant fraction of the total volume.
  3. **Near the critical point** : the distinction between liquid and gas disappears and density fluctuations become enormous.

Property | Ideal Gas Prediction | SCF Reality  
---|---|---  
Compressibility | Constant | Diverges at $T_c$  
Density dependence | Linear in $P$ | Strongly nonlinear  
Phase transition | Not predicted | Sharp VLE boundary  
Solvent power | Proportional to $P$ | Enhanced by molecular clustering  
  
The deviation from ideality is quantified by the **compressibility factor** :

$$ Z = \frac{PV}{nRT} = \frac{PM}{\rho RT} $$ 

where $M$ is the molar mass. For an ideal gas $Z = 1$; near the critical point $Z$ ranges from roughly 0.2 to 1.2.

### The van der Waals Equation

van der Waals (1873) corrected the ideal gas law for molecular size and attraction:

$$ \left(P + \frac{a}{V_m^2}\right)(V_m - b) = RT $$ 

where $V_m$ is the molar volume (m³/mol), $a$ the attraction parameter (Pa·m⁶·mol⁻²) and $b$ the excluded-volume parameter (m³/mol).

#### Physical meaning of the two corrections

  * **Pressure correction $a/V_m^2$** : attractive forces create an internal pressure, so the measured external pressure is lower than the ideal value. The $1/V_m^2$ dependence follows from pairwise interactions (the number of pairs scales as $N^2$). Larger $a$ means stronger attraction — polar or large molecules.
  * **Volume correction $-b$** : each molecule excludes a volume around itself, reducing the space available to the others. For hard spheres of radius $r$, $b \approx 4 \times \frac{4}{3}\pi r^3$, i.e. about four times the molecular volume.

Solving for pressure gives the form used for plotting isotherms:

$$ P = \frac{RT}{V_m - b} - \frac{a}{V_m^2} $$ 

The shape of the isotherm depends on temperature:

  * **$T > T_c$ (supercritical)**: monotonically decreasing, no phase transition
  * **$T = T_c$ (critical isotherm)** : a horizontal inflection point
  * **$T < T_c$ (two-phase)**: an S-shaped loop whose unphysical part is replaced by a horizontal tie line via the Maxwell equal-area construction

**Critical point conditions.** At the critical point the isotherm has a horizontal inflection:

$$ \left(\frac{\partial P}{\partial V_m}\right)_{T_c} = 0, \quad \left(\frac{\partial^2 P}{\partial V_m^2}\right)_{T_c} = 0 $$ 

Solving the two conditions simultaneously yields the critical constants

$$ T_c = \frac{8a}{27Rb}, \quad P_c = \frac{a}{27b^2}, \quad V_{m,c} = 3b $$ 

which can be inverted to obtain $a$ and $b$ from tabulated critical data:

$$ a = \frac{27R^2T_c^2}{64P_c}, \qquad b = \frac{RT_c}{8P_c} $$ 

**Universal van der Waals behaviour.** Substituting the critical constants back into the definition of $Z$ gives a value independent of the substance:

$$ Z_c = \frac{P_c V_{m,c}}{RT_c} = \frac{3}{8} = 0.375 $$ 

Real fluids have $Z_c$ between about 0.23 and 0.31 (CO₂: 0.274), so van der Waals systematically overestimates $Z_c$. This single number is the clearest statement of the equation's limits: it is excellent for teaching, unreliable for design.

### The Peng-Robinson Equation

The Peng-Robinson equation (1976) is the workhorse cubic EOS for hydrocarbons, CO₂ and supercritical process design:

$$ P = \frac{RT}{V_m - b} - \frac{a\alpha(T)}{V_m^2 + 2bV_m - b^2} $$ 

with

$$ a = 0.45724\frac{R^2T_c^2}{P_c} $$ $$ b = 0.07780\frac{RT_c}{P_c} $$ $$ \alpha(T) = \left[1 + \kappa\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2 $$ $$ \kappa = 0.37464 + 1.54226\omega - 0.26992\omega^2 $$ 

The **acentric factor** $\omega$ measures how far a molecule departs from a spherically symmetric, simple fluid:

$$ \omega = -\log_{10}\left(\frac{P^{sat}(T_r = 0.7)}{P_c}\right) - 1 $$  Substance | $\omega$ | Character  
---|---|---  
Ar, Kr, Xe | 0.00 - 0.01 | Spherical noble gases (simple fluids)  
CH₄ | 0.011 | Nearly spherical  
CO₂ | 0.225 | Linear molecule with quadrupole moment  
n-Hexane | 0.301 | Flexible chain  
H₂O | 0.344 | Polar, hydrogen bonded  
Long-chain alkanes | > 0.5 | Strongly non-spherical  
  
**Cubic form.** For computation the equation is recast in terms of $Z$:

$$ Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0 $$ 

where

$$ A = \frac{a\alpha P}{R^2T^2}, \quad B = \frac{bP}{RT} $$ 

Inside the two-phase envelope the cubic has three real positive roots: the largest is the vapour phase, the smallest is the liquid phase, and the middle root is thermodynamically unstable and physically meaningless. In the single-phase supercritical region only one real positive root survives.

**Heavy molecules.** The $\kappa$ correlation above was fitted for $\omega \le 0.49$. For heavier compounds Peng and Robinson later published an extended form:

$$ \kappa = 0.379642 + 1.48503\omega - 0.164423\omega^2 + 0.016666\omega^3 \quad (\omega > 0.49) $$ 

Using the light-molecule correlation for a heavy solute is a common and silent source of error in solubility calculations.

### Comparing Equations of State

EOS | Accuracy | Computational Cost | Best For  
---|---|---|---  
Ideal gas | Poor near $T_c$ | Minimal | $T \gg T_c$, $P < 1$ MPa  
van der Waals | Qualitative only | Low | Teaching, conceptual understanding  
Peng-Robinson | Good (5-10% in density) | Moderate | Hydrocarbons, CO₂, process design  
Soave-Redlich-Kwong | Good (similar to PR) | Moderate | Petroleum industry  
SAFT / PC-SAFT | Excellent (1-3%) | High | Complex molecules, polymers, associating fluids  
  
**Typical density prediction errors at $T = 1.05T_c$, $P = 1.5P_c$** — the region where most scCO₂ extraction runs:

  * van der Waals: 15-25%
  * Peng-Robinson: 5-10%
  * PC-SAFT: 1-3%

Because solubility depends on density raised to a power of roughly 4-10 (see the Chrastil equation in Section 6.4), a 20% density error becomes an order-of-magnitude solubility error. Choosing the EOS is therefore not a cosmetic decision.

* * *

## 6.2 Critical Phenomena

### Critical Opalescence

Approach the critical point and a transparent fluid turns milky white. This is **critical opalescence** , and it is the most direct visual evidence that something singular is happening.

  1. **Density fluctuations** : near $T_c$ the local density fluctuates by large amounts over long distances, because the free-energy cost of a fluctuation approaches zero.
  2. **Correlation length** : the spatial extent $\xi$ of correlated fluctuations diverges as $$ \xi \sim |T - T_c|^{-\nu} $$ with $\nu \approx 0.63$, the correlation-length critical exponent.
  3. **Light scattering** : when $\xi$ becomes comparable to the wavelength of visible light ($\lambda \sim 400$-$700$ nm), Rayleigh scattering intensifies dramatically.

The scattered intensity grows as

$$ I \sim \frac{\xi^6}{\lambda^4} \sim |T - T_c|^{-6\nu} \approx |T - T_c|^{-3.8} $$ 

which is why a supercritical fluid looks perfectly clear a few kelvin away from $T_c$ and opaque within a fraction of a kelvin of it.

### Divergence of the Response Functions

**Isothermal compressibility:**

$$ \kappa_T = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T = \frac{1}{\rho}\left(\frac{\partial \rho}{\partial P}\right)_T \sim |T - T_c|^{-\gamma} $$ 

with $\gamma \approx 1.24$. **Isobaric heat capacity:**

$$ C_P = T\left(\frac{\partial S}{\partial T}\right)_P \sim |T - T_c|^{-\alpha} $$ 

with $\alpha \approx 0.11$.

#### What the divergences mean in the laboratory

  * Diverging $\kappa_T$: a tiny pressure change produces a large volume (density) change — the origin of the "piston effect" and of density oscillations in flow loops.
  * Diverging $C_P$: a tiny temperature change absorbs a large amount of heat, so thermal response is sluggish and temperature control overshoots easily.

Both are the reason supercritical processes are usually operated at a deliberate distance from $T_c$ rather than on top of it.

### Universality and Critical Exponents

Remarkably, CO₂, water and xenon — molecules with nothing structural in common — share the same critical exponents. This is **universality**.

Exponent | Quantity described | 3D Ising value  
---|---|---  
$\alpha$ | Heat capacity | 0.110  
$\beta$ | Order parameter (density difference) | 0.326  
$\gamma$ | Susceptibility (compressibility) | 1.237  
$\delta$ | Critical isotherm | 4.789  
$\nu$ | Correlation length | 0.630  
  
The **order parameter** for a fluid is the density difference between the coexisting phases:

$$ \Delta\rho = \rho_L - \rho_V \sim |T - T_c|^{\beta} $$ 

and along the **critical isotherm** ($T = T_c$):

$$ |P - P_c| \sim |\rho - \rho_c|^{\delta} $$ 

The exponents are not independent; they obey **scaling relations** :

$$ \alpha + 2\beta + \gamma = 2 \quad \text{(Rushbrooke)} $$ $$ \gamma = \beta(\delta - 1) \quad \text{(Widom)} $$ 

### Renormalization Group Theory

**Renormalization group** theory (Wilson, 1971) explains universality. Its central idea is that at the critical point the system becomes invariant under changes of length scale, so short-range details are progressively washed out. What remains relevant is only:

  * the spatial dimensionality ($d = 3$ for fluids),
  * the dimensionality of the order parameter ($n = 1$ for a scalar density),
  * the symmetry of the interactions.

Molecular mass, bond angles and chemical identity become _irrelevant_ in the technical sense. The practical payoff is large: measure the critical behaviour of one fluid carefully and you have it for all of them. This is also the theoretical foundation of the principle of corresponding states, which lets a single reduced-property chart serve many substances.

* * *

## 6.3 Thermodynamic Properties near the Critical Point

### Enthalpy and Entropy

Departures from ideal behaviour are collected in residual integrals:

$$ H(T, P) = H^{ideal}(T) + \int_0^P \left[V - T\left(\frac{\partial V}{\partial T}\right)_P\right] dP $$ $$ S(T, P) = S^{ideal}(T, P) - \int_0^P \left(\frac{\partial V}{\partial T}\right)_P dP $$ 

Near $T_c$ these integrals become large and strongly nonlinear, because the thermal expansion coefficient $(\partial V/\partial T)_P$ itself diverges.

At the critical point the two coexisting phases become identical, so the latent heat and the entropy of vaporization both vanish:

$$ \Delta H_{vap} = H_V - H_L \sim |T - T_c|^{\beta} \to 0, \qquad \Delta S_{vap} = S_V - S_L \to 0 $$ 

The Clausius-Clapeyron equation then becomes indeterminate,

$$ \frac{dP}{dT} = \frac{\Delta S_{vap}}{\Delta V_{vap}} \to \frac{0}{0} $$ 

which is the thermodynamic statement that the vapour-pressure curve simply _stops_ at the critical point rather than continuing into the supercritical region.

**Practical implication** : heat exchangers in supercritical processes must be sized for very large enthalpy changes across small temperature spans.

### Heat Capacity Anomalies

The isobaric heat capacity

$$ C_P = \left(\frac{\partial H}{\partial T}\right)_P $$ 

shows a sharp peak at the critical point. For CO₂ at $P \approx P_c$:

Temperature | Condition | $C_P$ [J/(mol·K)]  
---|---|---  
295 K | ~10 K below $T_c$ (liquid) | ≈ 180  
304.1 K | At $T_c$ | → ∞ (theoretically divergent)  
313 K | ~10 K above $T_c$ | ≈ 70  
  
Physically, heat added near $T_c$ goes into rearranging molecular structure — breaking and forming clusters — rather than into raising the temperature. The constant-volume heat capacity $C_V$ also shows an anomaly, but a much weaker one.

The heat-capacity ratio follows from standard thermodynamics:

$$ \frac{C_P}{C_V} = \frac{\kappa_T}{\kappa_S} = 1 + \frac{TV\alpha_P^2}{\kappa_T C_V} $$ 

where $\alpha_P = (1/V)(\partial V/\partial T)_P$. Both capacities diverge at the critical point, but their ratio stays finite and approaches 1.

### Speed of Sound

The speed of sound is an adiabatic derivative:

$$ c = \sqrt{\left(\frac{\partial P}{\partial \rho}\right)_S} = \sqrt{\frac{1}{\rho\kappa_S}} $$ 

and for a real fluid can be written as

$$ c = \sqrt{\frac{C_P}{C_V}\left(\frac{\partial P}{\partial \rho}\right)_T} $$ 

Near the critical point $(\partial P/\partial \rho)_T \to 0$ because $\kappa_T$ diverges, while $C_P/C_V \to 1$ and the adiabatic compressibility $\kappa_S$ stays finite. The net result is a **minimum** in the speed of sound at $T_c$. For CO₂ at $P_c$:

  * $T = 290$ K (liquid): $c \approx 900$ m/s
  * $T = 304$ K ($T_c$): $c \approx 180$ m/s — the minimum
  * $T = 320$ K (supercritical): $c \approx 250$ m/s

**Process design implication** : ultrasonic level and flow instrumentation, which converts a transit time into a distance using an assumed sound speed, becomes unreliable near the critical point.

### Consequences for Process Design
    
    
    ```mermaid
    graph TD
        A[Critical Point Thermodynamics]
        A --> B[Process Challenges]
        A --> C[Process Opportunities]
    
        B --> B1[Temperature control difficulty]
        B --> B2[Pressure drop issues]
        B --> B3[Heat transfer limitations]
        B --> B4[Density oscillations]
    
        C --> C1[Tunable solvent power]
        C --> C2[Enhanced mass transfer]
        C --> C3[Rapid phase separation]
        C --> C4[Selective extraction]
    
        style A fill:#e0f7fa
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

#### Design guidance

  * Operate at a deliberate distance from the critical point: $T_r = 1.05$-$1.20$ and $P_r = 1.1$-$2.0$ retain most of the tunability while avoiding the worst of the divergences.
  * Provide generous control margins on both temperature and pressure.
  * Evaluate transient response with dynamic simulation — steady-state calculations hide the density oscillations that dominate start-up and shutdown.

* * *

## 6.4 Phase Equilibrium in SCF Systems

### Vapour-Liquid Equilibrium

In the subcritical region two phases coexist along the vapour-pressure curve. Equilibrium requires equality of chemical potential, or equivalently of fugacity:

$$ \mu_L(T, P) = \mu_V(T, P) \quad \Longleftrightarrow \quad f_L(T, P) = f_V(T, P) $$ 

The slope of the vapour-pressure curve is given by the **Clausius-Clapeyron equation** :

$$ \frac{dP^{sat}}{dT} = \frac{\Delta H_{vap}}{T \Delta V} = \frac{\Delta H_{vap}}{T(V_V - V_L)} $$ 

Assuming an ideal vapour and a temperature-independent latent heat gives the familiar integrated form $\ln P = -\Delta H_{vap}/(RT) + C$. For engineering work the empirical **Antoine equation** is preferred:

$$ \log_{10} P^{sat} = A - \frac{B}{C + T} $$ 

where $A$, $B$, $C$ are substance-specific constants tabulated for a stated temperature range and set of units. As $T \to T_c$, $\Delta H_{vap} \to 0$ and $V_V - V_L \to 0$, and the curve terminates at the critical point.

### Binary Phase Diagrams Containing an SCF

For a binary system (SCF solvent plus solute) the phase behaviour lives on a $P$-$T$-$x$ surface. Two features organize the picture:

  * the **critical locus** , the curve connecting the critical point of pure component 1 to that of pure component 2, which need not be monotonic;
  * the **three-phase line** , along which solid, liquid and vapour coexist, extending from the triple point of the pure component.

The classical van Konynenburg-Scott classification distinguishes several types; three matter for supercritical practice:

Type | Example system | Characteristic behaviour  
---|---|---  
Type I | CO₂ + light alkanes | Continuous critical curve between the two pure critical points; no azeotrope, no liquid-liquid immiscibility  
Type II | CO₂ + heavier alkanes | Critical curve with a temperature maximum and pressure minimum; a three-phase liquid-liquid-vapour region can appear  
Type III | CO₂ + heavy hydrocarbons, polymers | Critical curve runs away to high pressure; liquid-liquid immiscibility at moderate temperature  
  
Type III behaviour is not a nuisance but a tool: the strong pressure sensitivity it implies is exactly what makes highly selective supercritical fractionation possible.

### Solubility Modelling: the Chrastil Equation

Chrastil (1982) proposed a semi-empirical correlation for the solubility of solids in supercritical fluids:

$$ \ln S = k \ln \rho + \frac{a}{T} + b $$ 

where $S$ is the solubility, $\rho$ the SCF density (kg/m³), $k$ the association number, $a$ a constant related to the heats of solvation and vaporization (K), and $b$ a constant. Note that $S$ is reported either per unit volume of SCF (kg/m³) or per unit mass of SCF (kg/kg) depending on the source; the fitted $b$ absorbs the difference, so always check the units before reusing published parameters.

#### Interpreting the association number $k$

  * $k \approx 2$-6 for small molecules (caffeine, nicotine)
  * $k \approx 10$-20 for large molecules (triglycerides, fatty acids)

Physically, $k$ is the number of SCF molecules that solvate one solute molecule — the size of the solvation cluster. It also fixes the pressure sensitivity of the process, because

$$ \frac{\partial \ln S}{\partial \ln \rho}\bigg|_T = k, \qquad \frac{\partial \ln S}{\partial (1/T)}\bigg|_\rho = a $$ 

At constant temperature, solubility rises steeply with density (and hence pressure). At constant density it usually falls with temperature, because solvation is exothermic ($a < 0$).

**Never reuse published Chrastil parameters without a dimensional check.** Because $b$ is a bare additive constant inside a logarithm, it silently absorbs every unit convention in the correlation — kg/m³ versus g/L for $\rho$, kg/kg versus kg/m³ versus g/L for $S$. A parameter set quoted without its units is unusable, and substituting it into the wrong convention can be wrong by many orders of magnitude. Always reproduce one tabulated data point from the paper before trusting a fitted parameter set, and refit from raw data whenever you can.

**Crossover behaviour.** At high pressure the temperature dependence can reverse, because two effects compete: raising $T$ at fixed pressure lowers the density (reducing solubility) but raises the solute's vapour pressure (increasing solubility). The pressure at which the isotherms cross is a real and reproducible feature, and mistaking it for experimental scatter is a classic error in solubility datasets.

### Retrograde Condensation

**Retrograde condensation** is the counterintuitive appearance of a liquid phase when temperature is _raised_ at constant pressure (or when pressure is lowered isothermally in a gas-condensate reservoir).

  1. Start in the single-phase supercritical region above the critical pressure.
  2. Raise the temperature isobarically.
  3. Cross the phase boundary — liquid droplets appear.
  4. Continue heating — the droplets redissolve and vanish.

**Molecular explanation** : at the lower temperature the SCF density, and therefore the solvent power, is high and everything stays dissolved. As temperature rises the density falls faster than the solute's vapour pressure rises, and the solute precipitates. It occurs when $T > T_c$, $P \approx P_c$, and the mixture critical curve has a negative slope.

**Where it matters** : natural-gas processing (condensate recovery from methane), supercritical antisolvent precipitation (SAS), and CO₂ enhanced oil recovery.

* * *

## 6.5 Thermodynamic Calculations

### Fugacity and the Fugacity Coefficient

**Fugacity** $f$ is the "effective pressure" that makes the ideal-gas expression for chemical potential exact:

$$ d\mu = RT\, d\ln f, \qquad \mu = \mu^\circ(T) + RT\ln\frac{f}{f^\circ} $$ 

The **fugacity coefficient** measures the departure from ideality:

$$ \phi = \frac{f}{P} $$ 

with $\phi = 1$ for an ideal gas. It is obtained from any equation of state by integration:

$$ \ln \phi = \int_{\infty}^{V_m} \left[\frac{P}{RT} - \frac{1}{V_m}\right] dV_m - \ln Z $$ 

Performing the integral for the Peng-Robinson equation gives a closed-form expression — one of the main reasons cubic equations remain popular:

$$ \ln \phi = (Z - 1) - \ln(Z - B) - \frac{A}{2\sqrt{2}B} \ln\left(\frac{Z + (1+\sqrt{2})B}{Z + (1-\sqrt{2})B}\right) $$ 

with $A$ and $B$ as defined in Section 6.1. The equilibrium condition between a liquid and a vapour phase then reads

$$ \phi_i^L x_i = \phi_i^V y_i $$ 

where $x_i$ and $y_i$ are the mole fractions in the liquid and vapour phases.

### Chemical Potential in the SCF Phase

For component $i$ in a mixture,

$$ \mu_i = \left(\frac{\partial G}{\partial n_i}\right)_{T,P,n_{j\neq i}} = \mu_i^0(T) + RT \ln\left(\frac{f_i}{f_i^0}\right), \qquad f_i = \phi_i(T, P, \\{x\\})\, x_i P $$ 

At liquid-like densities it is often more convenient to use an activity coefficient:

$$ f_i = \gamma_i(T, P, \\{x\\})\, x_i f_i^{pure}(T, P) $$ 

Equating the chemical potential of a solute in the solid and in the supercritical phase, $\mu_i^{solid} = \mu_i^{SCF}$, gives the solubility directly, and at infinite dilution

$$ \ln \gamma_i^\infty = \ln \phi_i^\infty - \ln \phi_i^{sat} + \frac{v_i^L(P - P_i^{sat})}{RT} $$ 

so that a solubility can be predicted from pure-component properties alone — the Poynting correction is the last term.

### Mixing Rules for Multicomponent Systems

Cubic EOS parameters for a mixture are built from pure-component parameters using **van der Waals one-fluid mixing rules** :

$$ a_m = \sum_i \sum_j x_i x_j a_{ij}, \qquad b_m = \sum_i x_i b_i $$ $$ a_{ij} = \sqrt{a_i a_j}\,(1 - k_{ij}) $$ 

The **binary interaction parameter** $k_{ij}$ corrects the geometric-mean assumption for unlike pairs. It is fitted to experimental data, satisfies $k_{ii} = 0$, and is typically small but far from negligible:

Pair | $k_{ij}$  
---|---  
CO₂ - ethanol | 0.10  
CO₂ - n-hexane | 0.13  
CO₂ - water | 0.19  
  
Positive $k_{ij}$ indicates weaker-than-geometric-mean attraction between unlike molecules (the usual case for CO₂ with hydrocarbons, $k_{ij} \approx 0.1$-$0.15$); negative values are rare. Advanced rules (Wong-Sandler, MHV2) embed an activity-coefficient model into the mixing rule and do considerably better for strongly polar mixtures.

#### Computational strategy for a two-phase flash

  1. Specify $T$, $P$ and the overall composition $\\{z_i\\}$.
  2. Guess the liquid $\\{x_i\\}$ and vapour $\\{y_i\\}$ compositions.
  3. Solve the cubic EOS for each phase to obtain $\phi_i^L$ and $\phi_i^V$.
  4. Update the compositions from the equilibrium relations.
  5. Iterate until $|\phi_i^L x_i - \phi_i^V y_i| < \epsilon$ for every component.

* * *

## 6.6 Python Code Examples

**Environment.** All eight examples in this section use only NumPy, SciPy and Matplotlib:

`pip install numpy scipy matplotlib`

Nothing here requires a thermophysical property library, so every result is reproducible from the equations printed above. Chapter 7 introduces CoolProp and `thermo` for reference-quality property data.

### Example 1: van der Waals Isotherms

Plot $P$-$V$ isotherms for CO₂ from the van der Waals equation, covering subcritical, critical and supercritical temperatures, and verify $Z_c = 3/8$ numerically.

Code Example 1: van der Waals Isotherms for CO₂
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # van der Waals parameters for CO2
    R = 8.314  # J/(mol·K)
    Tc = 304.1  # K
    Pc = 7.38e6  # Pa
    
    # Calculate a and b from critical constants
    a = 27 * R**2 * Tc**2 / (64 * Pc)  # Pa·m^6/mol^2
    b = R * Tc / (8 * Pc)  # m^3/mol
    
    # Molar volume range (avoid singularity at Vm = b)
    Vm = np.linspace(1.5*b, 50*b, 1000)
    
    # Temperature array: subcritical, critical, supercritical
    temperatures = [280, 304.1, 320, 350]
    colors = ['blue', 'red', 'green', 'purple']
    
    plt.figure(figsize=(10, 6))
    
    for T, color in zip(temperatures, colors):
        # van der Waals pressure
        P = R * T / (Vm - b) - a / Vm**2
    
        # Convert to MPa for plotting
        P_MPa = P / 1e6
    
        label = f'T = {T} K'
        if T == Tc:
            label += ' (critical)'
    
        plt.plot(Vm * 1e6, P_MPa, color=color, linewidth=2, label=label)
    
    # Mark critical point
    Vm_c = 3 * b
    P_c_vdw = R * Tc / (Vm_c - b) - a / Vm_c**2
    plt.plot(Vm_c * 1e6, P_c_vdw / 1e6, 'ro', markersize=10, label='Critical Point')
    
    plt.xlabel('Molar Volume (cm³/mol)', fontsize=12)
    plt.ylabel('Pressure (MPa)', fontsize=12)
    plt.title('van der Waals Isotherms for CO₂', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(0, 500)
    plt.ylim(0, 15)
    plt.tight_layout()
    plt.show()
    
    print("van der Waals parameters for CO₂:")
    print(f"a = {a:.4e} Pa·m⁶/mol²")
    print(f"b = {b:.4e} m³/mol")
    print(f"Critical compressibility: Zc = {Pc * Vm_c / (R * Tc):.3f}")
    

van der Waals parameters for CO₂: a = 3.6541e-01 Pa·m⁶/mol² b = 4.2823e-05 m³/mol Critical compressibility: Zc = 0.375

**What to look for:** subcritical isotherms ($T < T_c$) show the unphysical S-shaped loop that the Maxwell construction replaces with a horizontal tie line; the critical isotherm has a horizontal inflection point; supercritical isotherms are monotonic. The printed $Z_c = 0.375$ is exactly $3/8$.

### Example 2: Critical Point Determination

Locate the critical point as the simultaneous zero of the first and second derivatives of the isotherm, and check the analytical result numerically.

Code Example 2: Critical Point from the Inflection Conditions
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def van_der_waals_P(Vm, T, a, b, R=8.314):
        """van der Waals pressure."""
        return R * T / (Vm - b) - a / Vm**2
    
    def dP_dVm(Vm, T, a, b, R=8.314):
        """First derivative of pressure w.r.t. molar volume."""
        return -R * T / (Vm - b)**2 + 2 * a / Vm**3
    
    def d2P_dVm2(Vm, T, a, b, R=8.314):
        """Second derivative of pressure w.r.t. molar volume."""
        return 2 * R * T / (Vm - b)**3 - 6 * a / Vm**4
    
    def find_critical_point(a, b, R=8.314):
        """
        Find critical point from van der Waals parameters.
    
        Returns:
        --------
        Tc, Pc, Vm_c : floats
            Critical temperature, pressure, and molar volume
        """
        # Analytical solution for van der Waals
        Tc = 8 * a / (27 * R * b)
        Vm_c = 3 * b
        Pc = a / (27 * b**2)
    
        return Tc, Pc, Vm_c
    
    def verify_critical_conditions(Vm_c, Tc, a, b):
        """
        Verify that the first and second derivatives vanish at the critical point.
    
        In SI units these derivatives carry enormous scale factors
        (dP/dVm is of order Pc/Vm_c ~ 1e11 Pa per m³/mol), so an absolute
        tolerance is meaningless. Compare dimensionless residuals instead.
        """
        dP = dP_dVm(Vm_c, Tc, a, b)
        d2P = d2P_dVm2(Vm_c, Tc, a, b)
    
        Pc = a / (27 * b**2)
        res1 = dP * Vm_c / Pc
        res2 = d2P * Vm_c**2 / Pc
    
        print("Verification at critical point:")
        print(f"  (∂P/∂Vm)_Tc   · Vm_c/Pc   = {res1:.2e} (should be ≈ 0)")
        print(f"  (∂²P/∂Vm²)_Tc · Vm_c²/Pc  = {res2:.2e} (should be ≈ 0)")
    
        return np.abs(res1) < 1e-9 and np.abs(res2) < 1e-9
    
    # Example: Find critical point for CO2
    R = 8.314
    a = 0.3658  # Pa·m^6/mol^2 (fitted to experimental data)
    b = 4.267e-5  # m^3/mol
    
    Tc, Pc, Vm_c = find_critical_point(a, b, R)
    
    print("Critical Point Determination for CO₂")
    print("=" * 50)
    print("van der Waals parameters:")
    print(f"  a = {a:.4f} Pa·m⁶/mol²")
    print(f"  b = {b:.2e} m³/mol")
    print("\nCalculated critical point:")
    print(f"  Tc = {Tc:.2f} K")
    print(f"  Pc = {Pc/1e6:.2f} MPa")
    print(f"  Vm,c = {Vm_c*1e6:.2f} cm³/mol")
    print(f"  Zc = {Pc * Vm_c / (R * Tc):.4f}")
    print("\nExperimental values:")
    print("  Tc = 304.1 K")
    print("  Pc = 7.38 MPa")
    print("  Zc ≈ 0.274")
    
    # Verify conditions
    is_critical = verify_critical_conditions(Vm_c, Tc, a, b)
    print(f"\nCritical point conditions satisfied: {is_critical}")
    
    # Plot isotherms around the critical temperature
    Vm_range = np.linspace(1.5*b, 10*b, 500)
    temps = [Tc - 5, Tc, Tc + 5]
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(10, 6))
    
    for T, color in zip(temps, colors):
        P_values = [van_der_waals_P(Vm, T, a, b, R) for Vm in Vm_range]
        label = f'T = {T:.1f} K'
        if T == Tc:
            label += ' (Critical)'
        plt.plot(Vm_range*1e6, np.array(P_values)/1e6, color=color, linewidth=2, label=label)
    
    # Mark critical point
    Pc_calc = van_der_waals_P(Vm_c, Tc, a, b, R)
    plt.plot(Vm_c*1e6, Pc_calc/1e6, 'ko', markersize=10, label='Critical Point')
    
    # Add tangent line at critical point (should be horizontal)
    tangent_Vm = np.linspace(0.9*Vm_c, 1.1*Vm_c, 50)
    tangent_P = np.ones_like(tangent_Vm) * Pc_calc / 1e6
    plt.plot(tangent_Vm*1e6, tangent_P, 'k--', linewidth=1.5, alpha=0.7,
             label='Tangent (horizontal)')
    
    plt.xlabel('Molar Volume (cm³/mol)', fontsize=12)
    plt.ylabel('Pressure (MPa)', fontsize=12)
    plt.title('Critical Point: Inflection in van der Waals Isotherm',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(0, 400)
    plt.ylim(0, 12)
    plt.tight_layout()
    plt.show()
    

**Key insights.** The critical point is defined by $(\partial P/\partial V)_T = 0$ _and_ $(\partial^2 P/\partial V^2)_T = 0$. van der Waals reproduces $T_c$ and $P_c$ acceptably from fitted $a$ and $b$, but returns $Z_c = 0.375$ against the measured 0.274 — a 37% error in the critical molar volume. Quantitative work needs a better EOS.

### Example 3: Peng-Robinson Density Calculation

A reusable Peng-Robinson class: solve the cubic for $Z$, select the vapour or liquid root, and convert to mass density.

Code Example 3: Peng-Robinson EOS Solver
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Peng-Robinson EOS for pure component
    class PengRobinson:
        def __init__(self, Tc, Pc, omega):
            """
            Initialize PR EOS parameters.
    
            Parameters:
            -----------
            Tc : float
                Critical temperature (K)
            Pc : float
                Critical pressure (Pa)
            omega : float
                Acentric factor
            """
            self.R = 8.314  # J/(mol·K)
            self.Tc = Tc
            self.Pc = Pc
            self.omega = omega
    
            # Calculate a and b
            self.a = 0.45724 * self.R**2 * Tc**2 / Pc
            self.b = 0.07780 * self.R * Tc / Pc
    
            # Kappa parameter
            self.kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    
        def alpha(self, T):
            """Temperature-dependent attraction parameter correction."""
            Tr = T / self.Tc
            return (1 + self.kappa * (1 - np.sqrt(Tr)))**2
    
        def solve_Z(self, T, P):
            """
            Solve cubic equation for compressibility factor Z.
    
            Returns:
            --------
            Z_positive : array
                Real positive roots of the cubic equation
            """
            A = self.a * self.alpha(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
    
            # Cubic equation: Z^3 + p*Z^2 + q*Z + r = 0
            p = -(1 - B)
            q = A - 3*B**2 - 2*B
            r = -(A*B - B**2 - B**3)
    
            # Solve cubic
            coeffs = [1, p, q, r]
            Z_roots = np.roots(coeffs)
    
            # Return only real positive roots
            Z_real = Z_roots[np.isreal(Z_roots)].real
            Z_positive = Z_real[Z_real > 0]
    
            return Z_positive
    
        def density(self, T, P, phase='vapor'):
            """
            Calculate molar density.
    
            Parameters:
            -----------
            phase : str
                'vapor' (largest Z) or 'liquid' (smallest Z)
    
            Returns:
            --------
            rho : float
                Molar density (mol/m³)
            Z : float
                Selected compressibility factor
            """
            Z_values = self.solve_Z(T, P)
    
            if len(Z_values) == 0:
                raise ValueError("No real positive roots found")
    
            if phase == 'vapor':
                Z = np.max(Z_values)
            elif phase == 'liquid':
                Z = np.min(Z_values)
            else:
                raise ValueError("Phase must be 'vapor' or 'liquid'")
    
            Vm = Z * self.R * T / P  # Molar volume (m³/mol)
            rho = 1 / Vm  # Molar density (mol/m³)
    
            return rho, Z
    
    # CO2 properties
    co2 = PengRobinson(Tc=304.1, Pc=7.38e6, omega=0.225)
    
    # Test at various conditions
    test_conditions = [
        (310, 8e6, 'Supercritical'),
        (350, 15e6, 'High P, High T'),
        (280, 5e6, 'Subcritical (liquid)'),
        (280, 5e6, 'Subcritical (vapor)')
    ]
    
    print("Peng-Robinson EOS Results for CO₂:")
    print("=" * 70)
    
    for T, P, description in test_conditions:
        try:
            if 'liquid' in description:
                rho, Z = co2.density(T, P, phase='liquid')
            else:
                rho, Z = co2.density(T, P, phase='vapor')
    
            # Convert to kg/m³ (MW of CO2 = 44.01 g/mol)
            rho_kg = rho * 44.01 / 1000
    
            print(f"\n{description}:")
            print(f"  T = {T} K, P = {P/1e6:.1f} MPa")
            print(f"  Density = {rho_kg:.1f} kg/m³")
            print(f"  Compressibility factor Z = {Z:.3f}")
    
        except Exception as e:
            print(f"\n{description}: Error - {e}")
    
    # Plot density vs pressure at constant temperature
    T_iso = 313  # K (supercritical)
    P_range = np.linspace(7.5e6, 30e6, 50)
    densities = []
    
    for P in P_range:
        rho, _ = co2.density(T_iso, P, phase='vapor')
        densities.append(rho * 44.01 / 1000)  # Convert to kg/m³
    
    plt.figure(figsize=(8, 5))
    plt.plot(P_range / 1e6, densities, 'b-', linewidth=2)
    plt.xlabel('Pressure (MPa)', fontsize=12)
    plt.ylabel('Density (kg/m³)', fontsize=12)
    plt.title(f'CO₂ Density vs Pressure at T = {T_iso} K (Peng-Robinson)',
              fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

Peng-Robinson EOS Results for CO₂: ====================================================================== Supercritical: T = 310 K, P = 8.0 MPa Density = 330.0 kg/m³ Compressibility factor Z = 0.414 High P, High T: T = 350 K, P = 15.0 MPa Density = 428.6 kg/m³ Compressibility factor Z = 0.529 Subcritical (liquid): T = 280 K, P = 5.0 MPa Density = 868.7 kg/m³ Compressibility factor Z = 0.109 Subcritical (vapor): T = 280 K, P = 5.0 MPa Density = 203.3 kg/m³ Compressibility factor Z = 0.465

**What to look for:** at 280 K and 5 MPa the cubic has three real roots, and the liquid root (869 kg/m³) and vapour root (203 kg/m³) are both returned from the same call — the two-phase region in a nutshell. The whole liquid/vapour distinction in a cubic EOS reduces to picking the smallest or the largest root. Above $T_c$ only one root survives, and the density-pressure plot shows the extreme nonlinearity just above $P_c$ that makes scCO₂ tunable. Expect 5-10% deviation from reference data, growing to 10-20% within a few kelvin of $T_c$; Chapter 7 quantifies this against CoolProp.

### Example 4: Generalized Compressibility Chart

Reduced coordinates ($T_r$, $P_r$) collapse the behaviour of many fluids onto one chart. This example regenerates the classical compressibility chart from the Peng-Robinson equation instead of reading it off a figure.

Code Example 4: Compressibility Factor in Reduced Coordinates
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def PR_compressibility(Tr, Pr, omega):
        """
        Calculate compressibility factor using Peng-Robinson EOS
        in reduced coordinates.
    
        Parameters:
        -----------
        Tr : float
            Reduced temperature T/Tc
        Pr : float
            Reduced pressure P/Pc
        omega : float
            Acentric factor
    
        Returns:
        --------
        Z : float
            Compressibility factor (largest real root)
        """
        # PR parameters
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
    
        # Reduced parameters
        a_r = 0.45724 * alpha / Tr**2
        b_r = 0.07780 / Tr
    
        A = a_r * Pr
        B = b_r * Pr
    
        # Solve cubic equation
        coeffs = [1, -(1-B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
        roots = np.roots(coeffs)
    
        # Take largest real root (vapor-like phase)
        real_roots = roots[np.isreal(roots)].real
        Z = np.max(real_roots) if len(real_roots) > 0 else 1.0
    
        return Z
    
    # Create meshgrid of reduced properties
    Tr_range = np.linspace(0.7, 2.0, 100)
    Pr_range = np.linspace(0.1, 5.0, 100)
    Tr_grid, Pr_grid = np.meshgrid(Tr_range, Pr_range)
    
    # Calculate Z for CO2 (omega = 0.225)
    Z_grid = np.zeros_like(Tr_grid)
    
    for i in range(len(Pr_range)):
        for j in range(len(Tr_range)):
            Z_grid[i, j] = PR_compressibility(Tr_grid[i, j], Pr_grid[i, j], omega=0.225)
    
    # Plot contour map
    plt.figure(figsize=(12, 8))
    
    contour = plt.contourf(Tr_grid, Pr_grid, Z_grid, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Compressibility Factor Z')
    
    # Add contour lines
    contour_lines = plt.contour(Tr_grid, Pr_grid, Z_grid, levels=10, colors='white',
                                 linewidths=0.5, alpha=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Mark critical point
    plt.plot(1.0, 1.0, 'r*', markersize=20, label='Critical Point (Tr=1, Pr=1)')
    
    plt.xlabel('Reduced Temperature Tr = T/Tc', fontsize=13)
    plt.ylabel('Reduced Pressure Pr = P/Pc', fontsize=13)
    plt.title('Generalized Compressibility Chart (Peng-Robinson, ω=0.225)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.2, color='white')
    plt.xlim(0.7, 2.0)
    plt.ylim(0.1, 5.0)
    plt.tight_layout()
    plt.show()
    
    # Plot Z vs Pr for different isotherms
    plt.figure(figsize=(10, 6))
    
    Tr_values = [0.9, 1.0, 1.05, 1.1, 1.2, 1.5]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(Tr_values)))
    
    for Tr, color in zip(Tr_values, colors):
        Z_values = [PR_compressibility(Tr, Pr, 0.225) for Pr in Pr_range]
        label = f'Tr = {Tr:.2f}'
        if Tr == 1.0:
            label += ' (Critical)'
        plt.plot(Pr_range, Z_values, color=color, linewidth=2.5, label=label)
    
    # Ideal gas line
    plt.plot(Pr_range, np.ones_like(Pr_range), 'k--', linewidth=1.5,
             label='Ideal Gas (Z=1)')
    
    plt.xlabel('Reduced Pressure Pr = P/Pc', fontsize=12)
    plt.ylabel('Compressibility Factor Z', fontsize=12)
    plt.title('Compressibility Factor vs Reduced Pressure for CO₂',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(alpha=0.3)
    plt.xlim(0, 5)
    plt.ylim(0, 1.5)
    plt.tight_layout()
    plt.show()
    
    # Print some specific values
    print("\nCompressibility Factor at Selected Conditions:")
    print("=" * 60)
    test_cases = [
        (1.0, 1.0, "Critical point"),
        (1.05, 1.5, "Supercritical (typical extraction)"),
        (1.2, 2.0, "Supercritical (high density)"),
        (0.9, 0.5, "Subcritical vapor"),
    ]
    
    for Tr, Pr, description in test_cases:
        Z = PR_compressibility(Tr, Pr, 0.225)
        print(f"{description:35s}: Tr={Tr:.2f}, Pr={Pr:.2f} → Z={Z:.3f}")
    

Compressibility Factor at Selected Conditions: ============================================================ Critical point : Tr=1.00, Pr=1.00 → Z=0.321 Supercritical (typical extraction) : Tr=1.05, Pr=1.50 → Z=0.345 Supercritical (high density) : Tr=1.20, Pr=2.00 → Z=0.615 Subcritical vapor : Tr=0.90, Pr=0.50 → Z=0.664

**What to look for:** $Z$ falls as $P_r$ rises at fixed $T_r$ (attraction dominates), returns towards 1 at low $P_r$ (ideal-gas limit), and reaches its lowest values near $T_r = 1$, $P_r \sim 1$-$2$ — precisely the region used for extraction. The value at $T_r = P_r = 1$ is the Peng-Robinson critical compressibility, 0.307 analytically; the numerical root returns 0.321 because the cubic is extremely flat there.

### Example 5: Critical Exponents and the Order Parameter

The power law $\rho_L - \rho_V \sim |T - T_c|^{\beta}$ is best verified on log-log axes, where it becomes a straight line of slope $\beta$.

Code Example 5: Density Difference and the Exponent β
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def density_difference(T, Tc, rho_c, beta=0.326):
        """
        Vapour-liquid density difference near the critical point.
    
        rho_L - rho_V ~ |T - Tc|^beta
        """
        epsilon = np.abs((T - Tc) / Tc)
        # Amplitude B0 is substance-specific; 2*rho_c is a reasonable scale
        B0 = 2.0 * rho_c
        return B0 * epsilon**beta
    
    # Parameters for CO2
    Tc = 304.1  # K
    rho_c = 467.6  # kg/m³
    beta = 0.326  # critical exponent
    
    # Temperature range below Tc
    T = np.linspace(280, Tc - 0.01, 100)
    
    # Density difference
    delta_rho = density_difference(T, Tc, rho_c, beta)
    
    # Reduced temperature distance
    epsilon = (Tc - T) / Tc
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear plot
    ax1.plot(T, delta_rho, 'b-', linewidth=2)
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Density difference ρ_L − ρ_V (kg/m³)', fontsize=12)
    ax1.set_title('Vapour-liquid density difference near Tc', fontsize=13)
    ax1.axvline(Tc, color='red', linestyle='--', label=f'Tc = {Tc} K')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Log-log plot: a power law becomes a straight line of slope beta
    ax2.loglog(epsilon, delta_rho, 'bo-', linewidth=2, markersize=3, label='Data')
    fit_line = delta_rho[0] * (epsilon / epsilon[0])**beta
    ax2.loglog(epsilon, fit_line, 'r--', linewidth=2, label=f'Power law (β={beta})')
    ax2.set_xlabel('Reduced distance ε = (Tc − T) / Tc', fontsize=12)
    ax2.set_ylabel('Density difference (kg/m³)', fontsize=12)
    ax2.set_title('Power-law verification (log-log)', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # Recover the exponent from the synthetic data by linear regression
    slope, intercept = np.polyfit(np.log(epsilon), np.log(delta_rho), 1)
    print(f"Assumed critical exponent β = {beta}")
    print(f"Recovered slope from log-log fit = {slope:.3f}")
    

Assumed critical exponent β = 0.326 Recovered slope from log-log fit = 0.326

**What to look for:** the linear plot shows the coexistence curve flattening as $T \to T_c$; the log-log plot turns that into a straight line whose slope _is_ the exponent. This is the standard way experimental coexistence data are analysed — and the same procedure applied to real CO₂ data returns 0.32-0.33, not the mean-field value 0.5 that van der Waals predicts.

### Example 6: Fugacity Coefficient from Peng-Robinson

The fugacity coefficient is what converts an equation of state into a phase-equilibrium calculation. Here it is evaluated in closed form for CO₂ over a range of pressures.

Code Example 6: Fugacity Coefficient of CO₂
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    def fugacity_coefficient_PR(T, P, Tc, Pc, omega):
        """
        Fugacity coefficient from the Peng-Robinson equation of state.
    
        Returns
        -------
        phi : float
            Fugacity coefficient f/P
        Z : float
            Compressibility factor
        """
        R = 8.314  # J/(mol·K)
    
        # PR parameters
        a = 0.45724 * R**2 * Tc**2 / Pc
        b = 0.07780 * R * Tc / Pc
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
        alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2
    
        A = a * alpha * P / (R**2 * T**2)
        B = b * P / (R * T)
    
        # Compressibility factor from the cubic form
        def equation(Z):
            return Z**3 - (1-B)*Z**2 + (A - 3*B**2 - 2*B)*Z - (A*B - B**2 - B**3)
    
        Z = fsolve(equation, 1.0)[0]
    
        # Closed-form fugacity coefficient
        sqrt2 = np.sqrt(2)
        ln_phi = ((Z - 1) - np.log(Z - B)
                  - A/(2*sqrt2*B) * np.log((Z + (1+sqrt2)*B) / (Z + (1-sqrt2)*B)))
    
        return np.exp(ln_phi), Z
    
    # CO2 parameters
    Tc = 304.1  # K
    Pc = 7.38e6  # Pa
    omega = 0.225
    
    temperatures = [300, 320, 350]  # K
    pressures = np.linspace(0.1e6, 20e6, 100)  # Pa
    
    plt.figure(figsize=(10, 6))
    for T in temperatures:
        phi_values = []
        for P in pressures:
            phi, Z = fugacity_coefficient_PR(T, P, Tc, Pc, omega)
            phi_values.append(phi)
    
        label = f'{T} K'
        label += ' (subcritical)' if T < Tc else ' (supercritical)'
        plt.plot(pressures/1e6, phi_values, linewidth=2, label=label)
    
    plt.axhline(1.0, color='gray', linestyle='--', label='Ideal gas (φ=1)')
    plt.axvline(Pc/1e6, color='red', linestyle='--', alpha=0.5, label='Critical pressure')
    plt.xlabel('Pressure (MPa)', fontsize=12)
    plt.ylabel('Fugacity coefficient φ = f/P', fontsize=12)
    plt.title('Fugacity Coefficient of CO₂ (Peng-Robinson)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Value at a single condition
    T_target = 320  # K
    P_target = 10e6  # Pa
    phi, Z = fugacity_coefficient_PR(T_target, P_target, Tc, Pc, omega)
    f = phi * P_target
    print(f"\nT = {T_target} K, P = {P_target/1e6:.1f} MPa:")
    print(f"Compressibility factor Z = {Z:.3f}")
    print(f"Fugacity coefficient φ = {phi:.3f}")
    print(f"Fugacity f = {f/1e6:.2f} MPa")
    print(f"Departure from ideality: {abs(1-phi)*100:.1f}%")
    

T = 320 K, P = 10.0 MPa: Compressibility factor Z = 0.392 Fugacity coefficient φ = 0.604 Fugacity f = 6.04 MPa Departure from ideality: 39.6%

**What to look for:** $\phi \to 1$ as $P \to 0$, then falls steadily below 1 as attraction takes over. At 10 MPa and 320 K the fugacity is only 6.0 MPa — 40% below the pressure. Treating scCO₂ as an ideal gas at process conditions is not a small approximation, and any solubility calculation that does so will be wrong by a similar factor.

### Example 7: Fitting the Chrastil Solubility Model

Because the Chrastil equation is linear in $\ln\rho$ and $1/T$, the three parameters can be recovered by ordinary least squares on log-transformed data — no nonlinear solver, no initial guess.

Code Example 7: Chrastil Model Regression
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # This example reuses the PengRobinson class from Code Example 3 to supply
    # the CO2 densities. Run that block first (or place both in the same script).
    
    def chrastil_solubility(rho, T, k, a, b):
        """
        Chrastil solubility model.
    
        ln(S) = k * ln(rho) + a/T + b
    
        Parameters:
        -----------
        rho : float or array
            SCF density (kg/m³)
        T : float or array
            Temperature (K)
        k, a, b : float
            Chrastil parameters
    
        Returns:
        --------
        S : float or array
            Solubility (kg solute / m³ SCF)
        """
        ln_S = k * np.log(rho) + a / T + b
        return np.exp(ln_S)
    
    # Experimental data: beta-carotene in scCO2 (representative literature values)
    data = {
        'T': np.array([313, 313, 313, 333, 333, 333, 353, 353, 353]),  # K
        'P': np.array([15, 20, 25, 15, 20, 25, 15, 20, 25]) * 1e6,  # Pa
        'S_exp': np.array([0.08, 0.15, 0.22, 0.12, 0.20, 0.30, 0.18, 0.28, 0.40])  # kg/m³
    }
    
    co2 = PengRobinson(Tc=304.1, Pc=7.38e6, omega=0.225)
    M_CO2 = 44.01e-3  # kg/mol
    
    def co2_density(T, P):
        """Mass density of CO2 (kg/m³) from the Peng-Robinson EOS."""
        rho_molar, _ = co2.density(T, P, phase='vapor')
        return rho_molar * M_CO2
    
    data['rho'] = np.array([co2_density(T, P)
                            for T, P in zip(data['T'], data['P'])])
    print("CO₂ densities from Peng-Robinson (kg/m³):")
    print(np.round(data['rho'], 1))
    
    # Linear least squares on the log-transformed model:
    # ln(S) = k * ln(rho) + a * (1/T) + b
    ln_S_exp = np.log(data['S_exp'])
    ln_rho = np.log(data['rho'])
    T_inv = 1 / data['T']
    
    X = np.column_stack([ln_rho, T_inv, np.ones(len(ln_rho))])
    y = ln_S_exp
    
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    k, a, b = params
    
    print("Chrastil Model Fitting for β-Carotene in scCO₂")
    print("=" * 60)
    print("Fitted parameters:")
    print(f"  k (association number) = {k:.3f}")
    print(f"  a (heat term, K) = {a:.1f}")
    print(f"  b (constant) = {b:.3f}")
    
    # Predicted solubilities and goodness of fit
    S_pred = chrastil_solubility(data['rho'], data['T'], k, a, b)
    
    SS_res = np.sum((S_pred - data['S_exp'])**2)
    SS_tot = np.sum((data['S_exp'] - np.mean(data['S_exp']))**2)
    R2 = 1 - SS_res / SS_tot
    RMSE = np.sqrt(np.mean((S_pred - data['S_exp'])**2))
    
    print("\nGoodness of fit:")
    print(f"  R² = {R2:.4f}")
    print(f"  RMSE = {RMSE:.4f} kg/m³")
    
    # Visualise the fit
    plt.figure(figsize=(12, 5))
    
    # Parity plot
    plt.subplot(1, 2, 1)
    plt.scatter(data['S_exp'], S_pred, c=data['T'], cmap='coolwarm', s=100,
                edgecolor='black')
    plt.plot([0, max(data['S_exp'])], [0, max(data['S_exp'])], 'k--',
             label='Perfect fit')
    plt.xlabel('Experimental solubility (kg/m³)', fontsize=12)
    plt.ylabel('Predicted solubility (kg/m³)', fontsize=12)
    plt.title('Parity plot: Chrastil model', fontsize=13, fontweight='bold')
    plt.colorbar(label='Temperature (K)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Solubility vs pressure at each temperature
    plt.subplot(1, 2, 2)
    
    for T_iso in [313, 333, 353]:
        P_smooth = np.linspace(15e6, 25e6, 50)
        rho_smooth = np.array([co2_density(T_iso, P) for P in P_smooth])
        S_smooth = chrastil_solubility(rho_smooth, T_iso, k, a, b)
    
        plt.plot(P_smooth / 1e6, S_smooth, linewidth=2, label=f'T = {T_iso} K (model)')
    
        mask = data['T'] == T_iso
        plt.scatter(data['P'][mask] / 1e6, data['S_exp'][mask], s=100,
                    edgecolor='black')
    
    plt.xlabel('Pressure (MPa)', fontsize=12)
    plt.ylabel('Solubility (kg/m³)', fontsize=12)
    plt.title('β-Carotene solubility in scCO₂', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Predict at a new condition
    T_new, P_new = 323, 22e6  # K, Pa
    rho_new = co2_density(T_new, P_new)
    S_new = chrastil_solubility(rho_new, T_new, k, a, b)
    
    print("\nPrediction at a new condition:")
    print(f"  T = {T_new} K, P = {P_new/1e6:.1f} MPa")
    print(f"  Estimated ρ(CO₂) = {rho_new:.1f} kg/m³")
    print(f"  Predicted solubility = {S_new:.3f} kg/m³")
    

CO₂ densities from Peng-Robinson (kg/m³): [748.7 830.6 886.2 562.2 694.9 774.6 411. 563.4 663.5] Chrastil Model Fitting for β-Carotene in scCO₂ ============================================================ Fitted parameters: k (association number) = 2.281 a (heat term, K) = -4544.9 b (constant) = -2.787 Goodness of fit: R² = 0.8986 RMSE = 0.0298 kg/m³ Prediction at a new condition: T = 323 K, P = 22.0 MPa Estimated ρ(CO₂) = 793.6 kg/m³ Predicted solubility = 0.196 kg/m³

**The fit is only as good as the density model.** Solubility scales as $\rho^k$, so a relative error $\varepsilon$ in density becomes roughly $k\varepsilon$ in solubility — and it propagates straight into the fitted parameters. Peng-Robinson densities (5-10% accurate) are adequate to demonstrate the regression; a reference EOS such as CoolProp (Chapter 7) is what you would use to publish parameters. The illustrative dataset here returns $k \approx 2.3$, at the low end of the range expected for a molecule as large as β-carotene, which is exactly the kind of symptom a density-model error produces.

The negative $a = -4545$ K confirms exothermic solvation: at fixed density, solubility falls as temperature rises.

### Example 8: Mixing Rules for a Binary System

Finally, the mixing rules that extend a pure-component EOS to mixtures, and the sensitivity of the mixture parameters to the binary interaction parameter $k_{12}$.

Code Example 8: van der Waals Mixing Rules for CO₂ + Ethanol
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def mixing_rule(x1, a1, a2, b1, b2, k12=0):
        """
        van der Waals one-fluid mixing rules.
    
        Parameters:
        -----------
        x1 : float
            Mole fraction of component 1
        a1, a2 : float
            Pure-component a parameters
        b1, b2 : float
            Pure-component b parameters
        k12 : float
            Binary interaction parameter
    
        Returns:
        --------
        a_mix, b_mix : float
            Mixture parameters
        """
        x2 = 1 - x1
    
        # Cross parameter
        a12 = np.sqrt(a1 * a2) * (1 - k12)
    
        # Mixing rules
        a_mix = x1**2 * a1 + 2*x1*x2*a12 + x2**2 * a2
        b_mix = x1 * b1 + x2 * b2
    
        return a_mix, b_mix
    
    R = 8.314  # J/(mol·K)
    
    # CO2
    Tc1, Pc1 = 304.1, 7.38e6  # K, Pa
    a1 = 27 * R**2 * Tc1**2 / (64 * Pc1)
    b1 = R * Tc1 / (8 * Pc1)
    
    # Ethanol
    Tc2, Pc2 = 513.9, 6.14e6  # K, Pa
    a2 = 27 * R**2 * Tc2**2 / (64 * Pc2)
    b2 = R * Tc2 / (8 * Pc2)
    
    k12_values = [0.0, 0.05, 0.10, 0.15]
    x_CO2 = np.linspace(0, 1, 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for k12 in k12_values:
        a_mix_values = []
        b_mix_values = []
    
        for x1 in x_CO2:
            a_mix, b_mix = mixing_rule(x1, a1, a2, b1, b2, k12)
            a_mix_values.append(a_mix)
            b_mix_values.append(b_mix)
    
        ax1.plot(x_CO2, a_mix_values, linewidth=2, label=f'k₁₂ = {k12}')
        ax2.plot(x_CO2, np.array(b_mix_values)*1e6, linewidth=2, label=f'k₁₂ = {k12}')
    
    ax1.set_xlabel('CO₂ mole fraction', fontsize=12)
    ax1.set_ylabel('a_mix (Pa·m⁶/mol²)', fontsize=12)
    ax1.set_title('Mixture attraction parameter', fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('CO₂ mole fraction', fontsize=12)
    ax2.set_ylabel('b_mix (cm³/mol)', fontsize=12)
    ax2.set_title('Mixture excluded-volume parameter', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Effect of the interaction parameter at equimolar composition
    print("Effect of k₁₂ at x_CO₂ = 0.5:")
    for k12 in k12_values:
        a_mix, b_mix = mixing_rule(0.5, a1, a2, b1, b2, k12)
        print(f"  k₁₂ = {k12:.2f}: a_mix = {a_mix:.4f} Pa·m⁶/mol², "
              f"b_mix = {b_mix*1e6:.2f} cm³/mol")
    

Effect of k₁₂ at x_CO₂ = 0.5: k₁₂ = 0.00: a_mix = 0.7434 Pa·m⁶/mol², b_mix = 64.90 cm³/mol k₁₂ = 0.05: a_mix = 0.7265 Pa·m⁶/mol², b_mix = 64.90 cm³/mol k₁₂ = 0.10: a_mix = 0.7096 Pa·m⁶/mol², b_mix = 64.90 cm³/mol k₁₂ = 0.15: a_mix = 0.6926 Pa·m⁶/mol², b_mix = 64.90 cm³/mol

**What to look for:** $b_m$ is strictly linear in composition and completely insensitive to $k_{12}$, while $a_m$ bows downward as $k_{12}$ increases. Only the attraction term carries the correction — which is why fitting a single $k_{12}$ per pair is usually enough to reproduce binary VLE data, and why an entrainer such as ethanol changes solvent power out of proportion to its mole fraction.

* * *

## Summary

### Key Takeaways

**1\. Equations of state**

  * The ideal gas law fails catastrophically near the critical point.
  * van der Waals introduces physically meaningful corrections for attraction ($a$) and molecular size ($b$), and predicts the universal but inaccurate $Z_c = 3/8$.
  * Peng-Robinson gives 5-10% density accuracy at moderate cost and is the practical default.
  * The acentric factor $\omega$ carries the molecular non-sphericity.

**2\. Critical phenomena**

  * Critical opalescence follows from a diverging correlation length.
  * $\kappa_T$ and $C_P$ diverge with universal exponents; only dimensionality and symmetry matter.
  * Renormalization group theory explains why chemically unrelated fluids behave identically.

**3\. Properties near $T_c$**

  * Enthalpy, entropy and heat capacity vary sharply over small temperature spans.
  * $\Delta H_{vap} \to 0$: the vapour-pressure curve ends at the critical point.
  * The speed of sound passes through a minimum, defeating ultrasonic instrumentation.

**4\. Phase equilibrium**

  * Binary systems are classified by the shape of the critical locus (Type I, II, III).
  * The Chrastil equation, $\ln S = k \ln \rho + a/T + b$, captures solubility with three parameters.
  * Retrograde condensation is counterintuitive but industrially exploited.

**5\. Thermodynamic calculations**

  * The fugacity coefficient $\phi$ quantifies non-ideality and has a closed form for cubic EOS.
  * Phase equilibrium reduces to $\phi_i^L x_i = \phi_i^V y_i$.
  * Mixing rules with a fitted $k_{ij}$ extend pure-component EOS to mixtures.

**Practical implications**

  * Choose the EOS according to fluid type and the accuracy the decision actually requires.
  * Design with margin for the rapid property variation near the critical point.
  * Use reduced properties for generalized correlations and sanity checks.
  * Never assume ideal mixing in a multicomponent supercritical system.

* * *

**Review Questions**

#### Question 1: Ideal Gas Failure

List three assumptions of the ideal gas law and explain which one fails first as CO₂ is compressed isothermally at 320 K from 1 MPa to 20 MPa.

#### Question 2: van der Waals Critical Constants

Starting from $P = RT/(V_m - b) - a/V_m^2$, derive $V_{m,c} = 3b$ and $T_c = 8a/(27Rb)$, then show that $Z_c = 3/8$. Why is the measured value for CO₂ only 0.274?

#### Question 3: Acentric Factor

Water has $\omega = 0.344$ and methane $\omega = 0.011$. Compute $\kappa$ for each and explain what the difference implies about the temperature dependence of $\alpha(T)$.

#### Question 4: Critical Exponents

Using $\alpha = 0.110$, $\beta = 0.326$ and $\gamma = 1.237$, check the Rushbrooke relation $\alpha + 2\beta + \gamma = 2$. What would mean-field (van der Waals) theory predict for $\beta$, and how does that compare with experiment?

#### Question 5: Process Control

An extraction unit is specified at $T_r = 1.005$ to maximize density tunability. Give three reasons why an experienced engineer would move it to $T_r = 1.05$.

#### Question 6: Chrastil Interpretation

A fit for a triglyceride in scCO₂ returns $k = 14$. What does that value say about the solvation cluster, and what does it imply about the pressure sensitivity of the extraction yield?

#### Question 7: Fugacity

At 320 K and 10 MPa, CO₂ has $\phi \approx 0.60$. State what this means physically, and explain why a solubility model that uses $P$ in place of $f$ will be systematically biased.

#### Question 8: Mixing Rules

Ethanol is added to scCO₂ as an entrainer at 5 mol%. Using $k_{12} = 0.10$, discuss qualitatively how $a_m$ and $b_m$ change and why a small amount of a polar co-solvent has a disproportionate effect on the solubility of polar solutes.

* * *
