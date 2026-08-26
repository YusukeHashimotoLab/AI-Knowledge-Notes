---
title: "Chapter 4: Case Studies — What SDLs Have Actually Done"
chapter_title: "Chapter 4: Case Studies — What SDLs Have Actually Done"
subtitle: Landmark Campaigns, Their Numbers, and Their Controversies
reading_time: 20-25 minutes
difficulty: Intermediate
code_examples: 1
exercises: 3
version: 1.0
created_at: 2026-08-23
---

# Chapter 4: Case Studies — What SDLs Have Actually Done

This chapter examines four landmark autonomous campaigns in detail — the mobile robotic chemist, A-Lab, thin-film/photovoltaic SDLs, and flow-chemistry platforms — including the numbers they reported and the debates they triggered.

**Landmark Campaigns, Their Numbers, and Their Controversies**

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Summarize the design and quantitative results of the Liverpool mobile robotic chemist campaign
  * ✅ Summarize the A-Lab campaign and articulate both the claim and the criticisms
  * ✅ Describe closed-loop successes in thin films, perovskites, and nanoparticles
  * ✅ Explain what flow-chemistry SDLs contribute to organic synthesis
  * ✅ Extract common success factors and failure modes across case studies

**Reading Time**: 20-25 minutes **Code Examples**: 1 **Exercises**: 3

* * *

## 4.1 The Mobile Robotic Chemist (Liverpool, 2020)

**Problem**: find better photocatalyst formulations for hydrogen evolution from water — a 10-variable formulation space (photocatalyst amount, dye and scavenger concentrations, additives, pH-related components) far too large for manual search.

**System**: a commercial mobile robot (KUKA arm on a wheeled base) navigating a *conventional* chemistry lab by laser positioning, operating the same solid dispensers, liquid handlers, sonicator, photolysis station, and gas chromatograph a human would use. Batches of experiments were planned by a Bayesian optimizer.

**Campaign numbers** (Burger et al., *Nature* 2020):

| Metric | Value |
|---|---|
| Duration | 8 days |
| Experiments | 688 |
| Robot working pattern | ~22 hours/day (charging breaks) |
| Search space | 10 variables |
| Result | Photocatalyst mixture ~**6× more active** than baseline |

**Why it matters**: autonomy was achieved *without* redesigning the lab — the robot adapted to human infrastructure. The paper became the template for "retrofit autonomy" and demonstrated genuinely useful chemistry (identifying beneficial and useless additives) rather than a toy objective.

**Limits**: one sample in transit at a time caps throughput; eight days of round-the-clock operation still required humans for stocking and fault recovery.

## 4.2 A-Lab (Berkeley, 2023) — Claim and Controversy

**Problem**: computational databases predict many thousands of plausibly stable inorganic compounds that no one has ever synthesized. Can a robot make them?

**System**: purpose-built solid-state synthesis line — robotic powder dosing, box furnaces served by robot arms, automated XRD of products — with two AI layers: literature-trained models proposing synthesis recipes (precursors, temperatures), and an active-learning loop revising recipes when a synthesis failed. Phase identification used ML-assisted matching of measured XRD patterns against computed references.

**Campaign numbers** (Szymanski et al., *Nature* 2023):

| Metric | Value |
|---|---|
| Duration | 17 days |
| Novel targets attempted | 58 |
| Claimed successes | 41 (71%) |
| Targets sourced from | Materials Project + GNoME predictions |

**The controversy**: shortly after publication, solid-state chemists — most prominently Robert Palgrave, with further analysis co-authored with Leslie Schoop — argued that a substantial fraction of the claimed products were **misidentified**: some XRD fits ignored compositional disorder and actually corresponded to known phases or mixtures, and some "novel" targets were not truly new. The critique did **not** allege fabricated data; it targeted the *automated interpretation* step — exactly the part of the loop with the least human oversight.

**Lessons the field drew**:

1. **Characterization is the weakest link.** Automated Rietveld-style fitting can be confidently wrong; disorder and solid solutions confound database matching.
2. **Autonomous claims need human-grade validation** of at least a sample of results before headline numbers are quoted.
3. Post-publication scrutiny is part of the scientific loop — and SDL papers, with their complete logged datasets, are unusually *auditable*. The debate was possible because A-Lab published its raw patterns.

A-Lab remains a landmark: robotic solid-state synthesis at that scale had never been demonstrated, and the platform continues operating with improved analysis pipelines.

## 4.3 Thin Films, Perovskites, and Nanoparticles

### Ada and Thin-Film Process Optimization

The Ada platform (Chapter 1) demonstrated closed-loop optimization of spray-coated and spin-coated films — e.g., maximizing the hole mobility of solar-cell hole-transport layers by tuning annealing time and dopant concentration — with tens of experiments per campaign. Its architecture (modular stations + central sample-handling robot + BO planner) became a reference design copied widely.

### Perovskite Solar Materials

Multiple groups have run closed loops over halide-perovskite composition and processing:

- Solution-based robots exploring mixed-cation/mixed-halide composition spaces for band gap targeting and stability, with in-line photoluminescence and imaging
- Campaigns explicitly optimizing **stability under illumination/humidity** — a multi-objective problem (efficiency-proxy vs. degradation rate) suited to EHVI-style planners
- Typical outcome: competitive compositions found in **1–2 hundred experiments**, versus thousands in grid-based prior studies

### Nanoparticle Synthesis in Flow

Flow SDLs excel at colloidal nanocrystals (quantum dots, metal nanoparticles): residence time, temperature, and reagent ratios tune size and shape, while in-line UV-Vis/PL gives second-scale feedback. Closed loops routinely dial in **target absorption/emission spectra** — i.e., inverse design ("give me particles that look like *this* spectrum") — within a working day, a task that took weeks manually.

## 4.4 Organic Synthesis: Chemputer and RoboRXN

- **Chemputer (Cronin group, Glasgow)**: a universal batch-chemistry robot executing procedures written in a chemical programming language (XDL). Its contribution is **reproducible executable synthesis** — published procedures that run identically on any Chemputer — more than optimization per se.
- **RoboRXN (IBM)**: cloud-connected flow synthesis driven by ML retrosynthesis models; users submit target molecules and the system plans and executes routes remotely. It demonstrated the **cloud-lab** model: the experimenter never touches the hardware.
- Academic flow platforms (e.g., MIT's) closed the loop on reaction yield optimization across multistep sequences, combining mechanistic models with BO.

These systems broaden the SDL concept from *materials optimization* toward *general chemical execution* — and point at remote, shared autonomous facilities.

## 4.5 Cross-Cutting Lessons

Reading the case studies together:

**Success factors**

- Fast, robust, automatable objective measurement (GC, UV-Vis, PL — minutes per sample)
- Search spaces encoded with chemical constraints (compositions on a simplex, log-scaled concentrations)
- Batch-parallel hardware matched by batch-aware planners
- Complete logging — the campaigns that survived scrutiny are the ones with auditable data

**Recurring failure modes**

- Silent hardware faults poisoning the dataset (undetected failed dispenses)
- Over-trusted automated interpretation (the A-Lab lesson)
- Objectives that drift from the real goal (proxy metrics optimized into meaninglessness)
- Integration debt: months lost to instrument drivers rather than science

```python
# The auditability dividend: every campaign above can be replayed
import pandas as pd
log = pd.read_json("campaign_log.json")          # x, y, raw paths, timestamps
assert log.shape[0] == 688                       # Liverpool-style full record
best = log.loc[log["h2_umol_per_h"].idxmax()]    # headline claim, recomputable
```

## 4.6 Chapter Summary

- Liverpool's mobile chemist: 688 experiments / 8 days / 6× activity gain — retrofit autonomy in a human lab
- A-Lab: 41/58 claimed novel compounds in 17 days — and a phase-identification controversy that made automated characterization the field's acknowledged weak point
- Film, perovskite, and nanoparticle SDLs deliver order-of-magnitude experiment reductions and same-day inverse design in flow
- Chemputer/RoboRXN extend autonomy to executable, remote organic synthesis
- Common thread: measurement speed, encoded chemistry, batch planning, and auditable logs decide success

**Next chapter**: the open problems — reproducibility, standards, data sharing, people — and the global initiatives addressing them.

## Exercises

1. **Analysis**: Compute experiments-per-day for the Liverpool campaign and for A-Lab. Which factors (hardware parallelism, cycle time, sample transport) explain the difference?
2. **Critical reading**: List three concrete checks a human scientist could add to A-Lab's pipeline to catch phase misidentification, and estimate what each costs in throughput.
3. **Transfer**: Pick a materials problem from your own field. Which case-study architecture (mobile retrofit, purpose-built line, flow) fits best, and what is the single hardest step to automate?
