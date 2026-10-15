---
title: "Chapter 5: Pumps and Piping Systems"
chapter_title: "Chapter 5: Pumps and Piping Systems"
subtitle: Matching the Machine to the Circuit
reading_time: 20-25 minutes
difficulty: Intermediate
code_examples: 1
exercises: 3
version: 1.0
created_at: 2026-08-23
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/lfGMREF-V-c?start=3065"
    title="Chemical Engineering Fluid Mechanics Ch.5: Pumps and Piping Systems"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

# Chapter 5: Pumps and Piping Systems

This chapter closes the series with the machine that pays the bill. Chapter 4 counted the cost of moving fluid through pipe; here we meet the pump that supplies it, learn to find the flow rate at which a pump and its piping agree, and see two common ways of controlling that flow — one usually cheap to install and wasteful to run, one usually costlier to install and more efficient in variable-flow service.

**Matching the Machine to the Circuit**

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Explain what a pump supplies to the mechanical energy balance: elevation, pressure, and friction
  * ✅ Define **head** in meters and convert it to pressure with $\Delta P = \rho g H$
  * ✅ Distinguish centrifugal from positive-displacement pumps and state when each is chosen
  * ✅ Locate the **operating point** where a pump curve meets a system curve, and predict how throttling and speed changes move it
  * ✅ State the **affinity laws** and use them to explain why variable-speed drives save energy
  * ✅ Screen a suction line for cavitation risk with $\text{NPSH}_a > \text{NPSH}_r$ plus an application-appropriate margin

**Reading Time**: 20-25 minutes **Code Examples**: 1 **Exercises**: 3

* * *

## 5.1 The Pump's Job

[Chapter 2](chapter-2.html) wrote the mechanical energy balance with a term $w_{\text{pump}}$ for the mechanical energy a pump delivers to the fluid, and left it as a promise. [Chapter 4](chapter-4.html) then made that promise expensive: friction in pipes, fittings, and valves consumes mechanical energy continuously, and something must keep replacing it. That something is the pump, and for the tank-to-tank duties treated here its job splits into three parts. (Those endpoints are large vessels whose surface velocities are negligible; where endpoint velocities differ appreciably, the balance carries a velocity-head change as a fourth term.)

It must **lift** the liquid against gravity, from a suction tank to a higher discharge point. It must **push** against pressure, from a vessel at one pressure into a vessel at a higher one. And it must **pay the friction bill** computed in Chapter 4 — a loss that, in turbulent flow with roughly constant friction and loss coefficients, grows about with the square of the flow rate. A pump that only had to lift liquid 10 m might easily be delivering another 15 m worth of energy purely to overcome pipe friction; on a long transfer line, friction can be the dominant term, though a large elevation or vessel-pressure difference can still outweigh it.

Centrifugal-pump performance is not usually quoted as energy in joules or as the resulting rise in pascals — differential pressure is common for positive-displacement equipment, but centrifugal curves are quoted in **head**, symbol $H$, in **meters**:

$$ H = \frac{w_{\text{pump}}}{g} $$

**Head is energy per unit weight of fluid rather than per unit mass**, which is why dividing the hydraulic energy per kilogram by $g$ leaves a length — and note that $w_{\text{pump}}$ here is the energy delivered to the *fluid*, so the shaft work supplied is larger by $1/\eta$. A pump "delivering 30 m of head" would, in the absence of friction, raise *any* suitable liquid 30 m — water, gasoline, or sulfuric acid alike. That approximate fluid-independence is a real engineering advantage of the unit: one number on a nameplate describes the machine rather than the contents, at comparable operating conditions and subject to viscosity and other corrections.

Pressure behaves differently. The pressure rise a pump produces follows from the hydrostatic relation of [Chapter 1](chapter-1.html):

$$ \Delta P = \rho g H $$

taken between sections of equal elevation and equal velocity head, where the total head reduces to the static-pressure rise. So the *same* 30 m of head becomes a different pressure rise in every liquid, in proportion to density. **Head is a property of the pump; pressure is a property of the pump and the fluid together.** Confusing the two is a common error in pump calculations, and Exercise 2 makes the difference concrete.

The power required follows from head and flow. Useful hydraulic power is $\rho g Q H$; the power actually drawn at the shaft is larger by the efficiency $\eta$, typically 0.6–0.8 for a well-selected centrifugal pump:

$$ \dot{W}_{\text{shaft}} = \frac{\rho g Q H}{\eta} $$

## 5.2 Two Families

Almost every pump in a chemical plant belongs to one of two families, distinguished by *how* they add energy.

```mermaid
flowchart TB
    P["Pump: adds mechanical energy to a liquid"] --> C["Centrifugal: a spinning impeller<br/>flings liquid outward"]
    P --> D["Positive displacement: a fixed volume<br/>is trapped and pushed on each stroke"]
    C --> C2["Head falls as flow rises;<br/>flow depends on the system"]
    D --> D2["Flow nearly constant;<br/>pressure depends on the system"]
    C2 --> C3["Single- and multi-stage centrifugals"]
    D2 --> D3["Gear, piston, diaphragm, peristaltic"]
```

A **centrifugal pump** accelerates liquid outward with a rotating impeller and converts that velocity into pressure. It is the dominant workhorse: simple, cheap, no valves, continuous, and — in purpose-designed slurry and solids-handling versions — able to pass suitable solids, which an ordinary clean-liquid impeller cannot. Its characteristic behavior is that **head generally falls as flow rises** over the normal operating range — open the discharge valve wider and the pump gives you more flow at less head.

A **positive-displacement (PD) pump** traps a fixed volume of liquid and pushes it out mechanically — a gear mesh, a piston, a flexing diaphragm. Its defining behavior is the mirror image: **flow is nearly constant regardless of the pressure it must overcome.** That is powerful and dangerous. Many driven PD pumps will keep trying to displace their volume against a closed discharge valve, and the pressure climbs until something yields. Some types do stop — a torque-limited drive can stall or trip, and an air-operated diaphragm pump simply stalls once discharge pressure balances its air supply — but that behavior is a property of the particular machine and drive, never something to rely on as protection. **A PD pump must never be operated against a closed valve without a relief device in the discharge line** — this is standard practice, not a refinement.

| | **Centrifugal** | **Positive displacement** |
|---|---|---|
| **Flow vs. pressure** | Flow falls as head rises | Flow nearly independent of pressure |
| **Best for** | Low-to-moderate viscosity, large flows | Viscous liquids, metering, high pressure |
| **Flow steadiness** | Smooth | Often pulsating (piston, diaphragm) |
| **Closed discharge valve** | Heats up; observe the manufacturer's minimum-flow and shutoff limits | Can rapidly reach destructive overpressure without effective protection |
| **Typical duty** | Transfer, circulation, reflux, cooling water | Dosing, polymer and slurry transfer, homogenizing |

As an initial selection tendency: **viscous and metering duties lean toward positive displacement, and most other services toward centrifugal.** High pressure does not settle the question on its own — multistage centrifugals cover many high-pressure duties — and flow rate, solids, control requirements, reliability, and economics all enter the final choice. Centrifugal pumps are the common default because they suit the majority of plant service.

## 5.3 The Operating Point

A centrifugal pump does not have "a flow rate." It has a **pump curve** — head versus flow, sloping downward — and the piping has a **system curve**: the static head it must overcome plus the friction loss of [Chapter 4](chapter-4.html), expressed as head, rising roughly as $Q^2$ for a turbulent, fixed-configuration system like the one used here. The pump can only deliver a combination that lies on *both* curves, so the plant runs at a feasible, stable intersection — the **operating point** — and the simple monotonic curves drawn here give exactly one.

Two common ways exist to move it. **Throttling** a discharge valve adds friction, steepening the system curve and sliding the point left to lower flow. It is simple and can act quickly, subject to actuator and hydraulic dynamics, and it is what the control valve of [Introduction Chapter 3](../chemical-engineering-introduction/chapter-3.html) does. It is also wasteful: the pump still produces high head, and the excess is destroyed across the valve as heat.

A **variable-speed drive (VSD)** instead moves the *pump* curve down. The scaling is captured by the **affinity laws**, the proportionalities relating a centrifugal pump's performance to its rotational speed $N$:

- Flow scales with speed: $Q \propto N$
- Head scales with speed squared: $H \propto N^2$
- Power scales with speed cubed: $P \propto N^3$

The cube on power is the central motivation for variable-speed pumping in friction-dominated service: a 20% speed reduction removes roughly half the shaft power. The economics also turn on the load profile, the static head, equipment efficiencies, energy prices, and capital cost. These are idealizations — they assume geometric similarity and unchanged efficiency — so treat them as a first estimate, not a guarantee.

The code below finds the operating point by intersecting a quadratic pump curve $H = a s^2 - bQ^2$ with a system curve $H = H_{\text{static}} + kQ^2$, where $a$ is the **shutoff head at full (reference) speed** — at a speed ratio $s$ the shutoff head is $a s^2$ — and $s$ is that speed ratio. This is a **simplified textbook treatment**: it applies the affinity scaling to the ideal quadratic curve rather than to a measured one, which is exactly what a real selection would use.

```python
import math

RHO = 998.0      # kg/m3, water
G = 9.81         # m/s2

A = 40.0         # m, pump shutoff head at full speed
B = 0.0015       # m per (m3/h)^2, pump curve droop
H_STATIC = 12.0  # m, static lift between the two tank levels
K = 0.0012       # m per (m3/h)^2, friction coefficient of the open piping

def operating_point(k, s=1.0):
    """Pump curve H = A*s^2 - B*Q^2 meets system curve H = H_STATIC + k*Q^2."""
    q2 = (A * s**2 - H_STATIC) / (B + k)          # solve the two curves for Q^2
    q = math.sqrt(q2)                             # m3/h
    h = H_STATIC + k * q2                         # m
    p_fluid = RHO * G * (q / 3600.0) * h / 1000.0 # kW delivered to the liquid
    p_valve = RHO * G * (q / 3600.0) * ((k - K) * q2) / 1000.0  # kW burned in the valve
    return q, h, p_fluid, p_valve

cases = [("full speed, valve open", K, 1.0),
         ("throttled (k doubled)", 2 * K, 1.0),
         ("speed reduced to 80%", K, 0.8)]

print(f"{'case':>23} {'Q [m3/h]':>9} {'H [m]':>7} {'P_fluid [kW]':>13} {'P_valve [kW]':>13}")
for label, k, s in cases:
    q, h, pf, pv = operating_point(k, s)
    print(f"{label:>23} {q:9.1f} {h:7.2f} {pf:13.2f} {pv:13.2f}")

#                    case  Q [m3/h]   H [m]  P_fluid [kW]  P_valve [kW]
#  full speed, valve open     101.8   24.44          6.77          0.00
#   throttled (k doubled)      84.7   29.23          6.74          1.99
#    speed reduced to 80%      71.0   18.04          3.48          0.00
```

Read the two control strategies against each other, remembering that the two controlled cases do not deliver the same flow — an energy comparison that decides between them has to be made at the same required duty. Throttling cuts flow by 17%, from 101.8 to 84.7 m³/h — and the power delivered to the liquid barely changes, 6.77 to 6.74 kW, because the pump simply climbs its curve to higher head. Nearly 2.0 kW, some 30% of the pump's output, is now destroyed in the valve. Slowing the pump to 80% speed cuts flow by 30% and fluid power by 49%, without burning anything in a valve. Note also that flow falls further than the affinity law alone suggests (71.0 m³/h, not $0.8 \times 101.8 = 81.5$): the affinity laws describe the pump, while the static head belongs to the system and does not scale at all.

## 5.4 NPSH and Cavitation

Chapter 2 previewed a failure that the energy balance makes easy to reach: accelerate a liquid enough, under conditions where the Bernoulli trade-off applies, and its static pressure falls toward its own vapor pressure, at which point it can boil. In a pump the local pressure minima that do this sit near the **impeller inlet and the blade leading edges** — on a blade's suction surface the minimum can be very localized — and the phenomenon has a name and a number.

**Net Positive Suction Head (NPSH)** is the margin, in meters of head, between the **absolute total** head available at the pump suction — the static-pressure head *plus* the suction velocity head $V_s^2/2g$, referred to the pump's datum — and the liquid's saturation vapor-pressure head $P^{sat}/(\rho g)$ (also absolute) at the operating temperature — the property introduced in [Thermodynamics Chapter 3](../chemical-engineering-thermodynamics/chapter-3.html). Feed a gauge reading into this comparison and one full atmosphere — about 10 m of water head — silently disappears from the margin: [Chapter 1](chapter-1.html)'s gauge-versus-absolute lesson in the most dangerous place it can be forgotten. **NPSH available** is computed from the plant: suction vessel pressure, liquid level, and the friction of the suction line. **NPSH required** is a property of the pump, published on its datasheet. The rule is unforgiving:

$$ \text{NPSH}_a > \text{NPSH}_r $$

with a design margin on top — absolute allowances of 0.5–1 m are seen in some services, but the requirement is set by the pump's suction characteristics, the energy level, and the applicable guidance, and is often expressed as a *ratio* rather than a fixed height. Note also what the published $\text{NPSH}_r$ usually is: NPSH3, the value at which the head has already dropped by 3%, which means some cavitation is present at that point. Satisfying the inequality is therefore not a guarantee of cavitation-free running, and vapor cavities commonly begin forming before the available head falls that far. Fall short of it and cavities form in the low-pressure region and collapse violently downstream where the pressure recovers. That is **cavitation**: gravel-in-the-pump noise, vibration, falling head, and impeller metal pitted away.

The main hydraulic remedies raise the left side or lower the right:

| Fix | Mechanism |
|---|---|
| Raise the suction tank or lower the pump | More static head at the suction |
| Cool the liquid | Lowers $P^{sat}$ steeply |
| Shorten, straighten, or enlarge the suction line | Less friction loss before the impeller |
| Slow the pump | Lowers $\text{NPSH}_r$ |

The classic victims are pumps handling **liquids already at their boiling point**, where the vessel-pressure contribution vanishes entirely and the available NPSH is whatever the liquid elevation above the pump provides, less the suction-line losses: reboiler circulation pumps, condensate pumps, and any pump taking suction from a vessel at saturation. For those services the fix is usually architectural — the vessel is simply installed high above the pump.

## 5.5 Gases: Fans, Blowers, and Compressors

Move a gas instead of a liquid and the job is the same but the fluid is compressible, which changes everything downstream. Machines are graded by pressure rise: **fans** for ventilation-scale pressures, **blowers** for moderate boosts, **compressors** for real ratios. Compressing a gas usually raises its temperature: the compression work goes into the gas faster than a real machine can remove heat, so the process is close to adiabatic, and the First Law of [Thermodynamics Chapter 1](../chemical-engineering-thermodynamics/chapter-1.html) together with the gas's property relations then fixes how much the temperature rises. (With enough heat removal the compression can instead be nearly isothermal.) A compressor therefore delivers hot gas that may need cooling before the next stage, and the work is genuine shaft work, the precious commodity Chapter 2 of this series insisted on. Where the overall pressure ratio or the discharge temperature demands it, the duty is split into **multiple stages with intercooling between them**, the same staged-and-cooled logic that governs interstage cooling in exothermic reactors. The result is less work for the same pressure ratio, and discharge temperatures kept within the limits of the materials, lubricants, and seals.

## 5.6 Series Summary

Five chapters, one sentence each. **Chapter 1** established fluid properties and the pressure that exists before anything moves. **Chapter 2** turned energy conservation into the mechanical energy balance and Bernoulli's equation, the tool that converts pressure, velocity, and elevation into one another. **Chapter 3** introduced the Reynolds number and the laminar-turbulent divide that decides which correlations apply. **Chapter 4** priced the friction in pipes and fittings and showed how sharply diameter controls that cost. **Chapter 5** supplied the machine that pays the bill and matched it to the circuit.

The arc does not end here. A pattern has recurred through this series — a **flux equal to a coefficient times a driving force**, clearest in Newton's law of viscosity, less literal in hydrostatics and in turbulent hydraulic resistance — and that same structure carries directly into **heat transfer** — our [Chemical Engineering Heat Transfer](../chemical-engineering-heat-transfer/index.html) series — where the driving force is a temperature difference, and into **mass transfer** — [Chemical Engineering Mass Transfer and Separation](../chemical-engineering-mass-transfer/index.html) — where, under the usual assumptions, it is a concentration difference. The friction factor you learned here has counterparts there playing analogous roles, though the quantitative transport analogies hold only under restricted conditions.

Where to go next: our *Chemical Engineering Introduction* series for the unit operations these flows serve, *Chemical Engineering Thermodynamics* for the property models behind vapor pressure and compression work, and *Process Informatics Introduction* for the data layer built on top of all of it.

Thank you for learning with us.

## Exercises

1. **Conceptual — choosing a family**: You must specify two pumps. (a) A 250 m³/h cooling-water circulation loop, water at 30 °C. (b) A catalyst additive dosed at 4 L/h into a pressurized reactor, accuracy ±2% required. Choose centrifugal or positive displacement for each, justify it, and name one protective device the second one must have.
   *Hint*: ask which matters more in each service — a large steady flow, or an exact one.
   *Answer*: **(a) Centrifugal.** Large flow, low viscosity, and no accuracy requirement — the kind of service for which centrifugals are commonly the economical, simple, and well-proven choice. (The head is not specified here; assume it is within ordinary single-stage range, and check it before selecting.) (b) **Positive displacement**, specifically a metering (diaphragm or piston) pump. Its flow is set by stroke and speed and is much less sensitive to discharge pressure, which is why it is the right family for holding a dose as reactor pressure varies — though the ±2% itself must be specified and verified over the expected flow and pressure range, since backpressure, compressibility, check-valve behavior, calibration, and suction conditions all bear on it. A centrifugal would deliver whatever the system curve allowed and its flow would swing with reactor pressure. The protective device is a **pressure relief valve in the discharge line** (or an integral relief): a PD pump against a blocked discharge raises pressure until a pipe, seal, or casing fails.

2. **Quantitative — head is not pressure**: A pump delivers **30 m of head**. Compute the pressure rise it produces (a) with water, $\rho = 998$ kg/m³, and (b) with an organic solvent, $\rho = 800$ kg/m³. Take $g = 9.81$ m/s². (c) What does the comparison tell you about reading a pump nameplate?
   *Hint*: $\Delta P = \rho g H$, in Pa when SI units go in; 1 bar = 10⁵ Pa.
   *Answer*: (a) $\Delta P = 998 \times 9.81 \times 30 = \mathbf{293{,}711\ Pa \approx 2.94\ bar}$. (b) $\Delta P = 800 \times 9.81 \times 30 = \mathbf{235{,}440\ Pa \approx 2.35\ bar}$ — about **20% less**, in the density ratio $800/998 = 0.802$. (c) The **head is the same in both cases** because head is a property of the machine, not the liquid; the **pressure rise is not**. So a nameplate head carries over to a new service far better than a pressure figure does — any pressure quoted with it is valid only for the fluid it was quoted for — but the transfer still has to be checked against the new fluid's viscosity, the operating flow and speed, NPSH, entrained gas, and materials compatibility. The same logic bites in reverse: a pump moved from water to a dense brine produces approximately the same head but a higher pressure and draws more power, since $\dot{W} = \rho g Q H/\eta$ also scales with density.

3. **Discussion — the flow is too high**: A centrifugal pump on a transfer line is delivering more flow than the process needs. An operator proposes pinching the discharge valve; an engineer proposes fitting a variable-speed drive. (a) Explain what each does to the curves and to the operating point. (b) Estimate the power saving from a 20% speed reduction and say why the real saving is smaller. (c) A third colleague suggests throttling the *suction* valve instead — what is wrong with that?
   *Hint*: one option changes the system curve, the other changes the pump curve; part (c) is about Section 5.4.
   *Answer*: (a) **Throttling steepens the system curve** by adding friction; the intersection slides left to lower flow but *higher* head, and the surplus head is destroyed as heat across the valve. In the Section 5.3 example, flow fell 17% while the power delivered to the liquid was essentially unchanged, with about 30% of it burned in the valve. **The VSD lowers the pump curve** ($H \propto N^2$) against an unchanged system curve, reaching lower flow at lower head without the additional throttling loss — which is not the same as wasting nothing, since pump, motor, drive, and piping losses all remain. (b) The affinity law $P \propto N^3$ gives $0.8^3 = 0.512$, an idealized saving of nearly **50%**, and the computed case gave 6.77 → 3.48 kW, a 49% drop in fluid power. Do **not** read the second number as confirmation of the first. The affinity laws hold along a path of geometrically similar operating states, which a purely frictional system provides; with **12 m of static head** in this system the operating point takes a different path, and the fluid power $\rho g Q H$ does not follow $N^3$ at all. Note too that the two figures are not even the same kind of quantity: 49% is a *reduction* in power, while 0.512 is the *fraction remaining* — the matching saving would be 48.8%, and comparing a saving with a remainder is the mistake to avoid. What survives is the conclusion: a VSD reaches the lower flow without destroying head in a valve, so in suitable variable-flow service it can save substantially more than throttling — compare the two at the same required flow before claiming a figure. The input-power saving must be computed from the actual operating points and the pump, motor, and drive efficiencies: pump efficiency shifts away from the **best-efficiency point** (the flow of maximum pump efficiency at that speed and configuration), the drive itself has losses of a few percent, and **static head does not scale with speed** — which is also why flow fell to 71.0 rather than the 81.5 m³/h that $Q \propto N$ alone predicts. On a system that is mostly static lift, a VSD saves much less, and can stop delivering entirely once $a s^2$ drops below the static head. (c) **Do not use the suction valve for flow control.** Discharge throttling costs energy; suction throttling reduces the pressure reaching the impeller, cutting $\text{NPSH}_a$ directly and risking **cavitation** — noise, vibration, lost head, and eroded impellers — which is why routine flow control belongs on the discharge side, or on the drive speed.
