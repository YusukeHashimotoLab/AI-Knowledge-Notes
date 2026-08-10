# Archived: `supercritical-fluids-introduction` (plural slug, Markdown-only)

**Archived**: 2026-08-10
**Moved from**: `knowledge/en/MS/supercritical-fluids-introduction/` and `knowledge/jp/MS/supercritical-fluids-introduction/`
**Moved to**: `claudedocs/archive/supercritical-fluids-md/{en,jp}/`

## Why

Two supercritical-fluid courses coexisted in the repository:

| | published | orphan (this directory) |
|---|---|---|
| slug | `supercritical-fluid-introduction` (singular) | `supercritical-fluids-introduction` (plural) |
| format | HTML (index + chapter-1..5) | Markdown only |
| linked from | `knowledge/{en,jp}/MS/index.html`, sitemap | nothing |
| content | qualitative, zero code | ~24,600 words, EOS derivations, ~38 Python examples |

The orphan was unreachable from any page. Its unique quantitative content was merged
into the published course as two new chapters rather than being published as a second,
competing course.

Note that the orphan's EN and JP trees were **not** translations of each other — they had
diverged independently (e.g. JP chapter-3 had 14 Python blocks vs EN 6; JP chapter-5 14 vs
EN 10, with different examples). Harvesting was therefore done from both locale trees.

## What was harvested

Into `knowledge/{en,jp}/MS/supercritical-fluid-introduction/chapter-6.html`
("Thermodynamics and Equations of State", 8 code examples):

- Prose: EN `chapter-2.md` as the base (richer), with JP `chapter-2.md` additions:
  ideal-gas breakdown reasons, `a`/`b` from critical constants, isotherm regimes,
  acentric-factor table, EOS accuracy table, critical-exponent table, Antoine equation,
  `k_ij` table, near-critical operating guidance (Tr = 1.05-1.20, Pr = 1.1-2.0),
  the extended kappa correlation for omega > 0.49.
- Code: EN examples 1 (vdW isotherms), 3 (critical point), 2 (PR solver),
  4 (compressibility chart), 5 (Chrastil regression); JP examples 3 (critical exponent),
  5 (fugacity coefficient), 6 (mixing rules).

Into `knowledge/{en,jp}/MS/supercritical-fluid-introduction/chapter-7.html`
("Practical Python for Supercritical Fluids", 10 code examples):

- Prose: EN `chapter-5.md` as the base, with JP `chapter-5.md` additions:
  property-code table, fixed-point lookup, CoolProp mixture (HEOS) usage, MD tooling notes.
- Code: EN examples 1 (property table), 2 (P-T diagram), P-V isotherms, 3 (PR class);
  JP examples: fixed-point lookup, vdW class, 4 (extraction process model),
  5 (RESS particle size), 6 (Chrastil fit), the HEOS mixture snippet.

## What was deliberately NOT carried over

- **Orphan chapters 1, 3, 4** (transport properties, applications in materials science):
  their scope overlaps the published chapters 1-4, and section-level transplants into
  existing chapters were deferred by the lead. These files remain here as the source of
  record if that work is picked up later. Chapter 3 in particular (transport properties:
  viscosity, diffusivity, thermal conductivity, mass-transfer correlations) has no
  counterpart in the published course and is the strongest candidate for a future chapter 8.
- **EN chapter-5 "Interactive Property Dashboard"** (matplotlib `Slider` widget, ~200
  lines): duplicates the property-map plotting already covered, cannot run in a headless
  or static context, and would have dominated the chapter's length.
- **EN chapter-5 P-V/P-rho duplication**: JP's `P-rho` isotherm example was dropped in
  favour of EN's `P-V` example (which also handles the two-phase tie line); the `P-rho`
  variant is described in prose as a one-line edit instead.
- **EN chapter-2 example 6 (binary bubble-point `BinaryPR`)**: the implementation is
  self-described as "simplified" and returns `ln_phi2 = ln_phi1` and `y1 = x1`, so the
  P-x-y diagram it plots is not the diagram it claims. JP's mixing-rule example was
  harvested instead.

## Corrections applied to harvested material

The orphan's equations were correct (Peng-Robinson kappa, Z_c = 3/8, critical exponents,
scaling relations all verified). These defects in the surrounding code/text were fixed:

1. **Critical-point verification tolerance** (EN ch2 ex3): compared SI-unit derivatives
   against `1e-6`, which is meaningless when `dP/dVm ~ 1e11 Pa per m3/mol`; the check
   printed `False` for an exactly correct derivation. Now compares dimensionless residuals.
2. **Crude CO2 density correlation** (EN ch2 ex5): `467.6 * Pr / Tr` returned up to
   1539 kg/m3 for CO2, physically impossible. Replaced by the Peng-Robinson solver from
   the preceding example.
3. **Caffeine Chrastil correlation** (JP ch2): `ln C = 7.5 ln rho - 5200/T - 15.3` evaluates
   to ~1e8 kg/kg. Removed; replaced with a units-provenance warning.
4. **Caffeine Chrastil parameters** (JP ch5 ex4): `{k: 8, a: -5000, b: 15}` gave a
   solubility of 89 g/kg CO2 (roughly 100x too high). `b` recalibrated to 10.51 so the
   solubility at the reference state is ~1 g/kg, and labelled in-code as calibrated for
   illustration rather than cited.
5. **Molar/mass basis error** (JP ch5 vdW validation): `1/PropsSI('D',...) * 44.01`
   is 1000x too large. Fixed to `molar_mass / density`.
6. **Partition coefficient** (EN ch5 ex4): `K_partition = 2.5` for caffeine in scCO2 gives
   an extraction factor of 50, so the "multi-stage" cascade completes in one stage and
   reports exactly 100.0000% recovery. Changed to `K = 0.05` (physically correct for
   caffeine, mass-fraction basis), giving E = 1 and a visible stage profile.
7. **Wrong "expected output" comments** (EN ch5): `628.19 kg/m3` for CO2 at 10 MPa / 50 C
   (actual reference value: 384.33 kg/m3), and a fabricated 2.05% PR error. All output
   blocks in the new chapters were regenerated by actually running the code (CoolProp 8.0.0,
   NumPy 2.3.3, SciPy 1.17.0, Matplotlib 3.10.7).
8. **CoolProp `PropsSI(fluid, 'Tcrit')`** argument order normalized to the canonical
   `PropsSI('Tcrit', fluid)`.
9. **CO2 + ethanol HEOS mixture** (JP ch5 exercise hint): CoolProp has no binary
   interaction data for that pair and raises. The example now sweeps several partners and
   shows the failure explicitly, which is itself the useful lesson.

## Related changes in the same merge

- `knowledge/{en,jp}/MS/supercritical-fluid-introduction/chapter-5.html`: added a
  next-chapter link to chapter 6.
- `knowledge/{en,jp}/MS/supercritical-fluid-introduction/index.html`: chapters 6-7 added to
  the chapter list and learning path; time/chapter/code-example counts updated.
- `knowledge/{en,jp}/MS/index.html`: chapter badge 5 -> 7.

Sitemap, search index and `knowledge/en/MS/TRANSLATION_STATUS.md` are generated artifacts
and are regenerated separately.

---

## Addendum: chapter 8 harvest (2026-08-10)

Into `knowledge/{en,jp}/MS/supercritical-fluid-introduction/chapter-8.html`
("Transport Properties of Supercritical Fluids", 9 code examples).

### Correction to this README's own provenance note

The "What was deliberately NOT carried over" section above describes orphan
**chapter 3** as "transport properties: viscosity, diffusivity, thermal conductivity,
mass-transfer correlations". That is wrong. Orphan chapter 3 in both locales is
*"Common Supercritical Fluids and Their Properties"* — a CO2/water/ethanol/propane/
nitrogen/xenon solvent survey with a selection guide. It contains no transport-property
treatment at all.

The mis-attribution came from the orphan's own forward references: EN `chapter-2.md:1333`
previews "Viscosity, diffusivity, and thermal conductivity" as the next chapter's content,
and JP `chapter-2.md:964` states outright that chapter 3 covers 輸送物性（粘度、拡散係数、
熱伝導率）. Neither chapter 3 delivers it. The only transport material in the orphan is
the qualitative order-of-magnitude tables in section 1.3 of both locales' `chapter-1.md`.

### What was harvested

- **Prose, transport sections (8.1-8.6)**: essentially new. The orphan supplied only the
  gas/SCF/liquid order-of-magnitude ranges and comparison tables from EN `chapter-1.md`
  §1.3.2-1.3.5 and JP `chapter-1.md` §1.3.2-1.3.6, which are already present qualitatively
  in the published chapters 1-3 and are therefore cross-referenced rather than restated.
  The quantitative treatment (residual-viscosity decomposition, Lucas correlation, critical
  enhancement, Prandtl/Schmidt/Sherwood, Ergun, critical slowing down) has no counterpart
  in the archive.
- **Prose, solvent survey (8.7)**: EN `chapter-3.md` §3.3-3.4 as the base (ethanol,
  propane, nitrogen, xenon, fluorinated fluids), with JP `chapter-3.md` §3.3-3.4 additions
  (HFC-134a critical constants and uses, green-solvent framing, sc-EtOH biodiesel timings).
- **Prose, selection guide (8.8)**: JP `chapter-3.md` §3.5.1 Mermaid decision tree (carried
  over as-is in structure, translated for EN); the comparison table merges EN §3.5.2 (safety,
  cost, applications) with JP §3.5.2 (usable density ranges); the modifier table is JP §3.5.3;
  the safety/regulatory table is JP §3.5.4 generalised past Japanese statute names to
  hazard-and-control language, since the published course has an international audience.
- **Code**: none of the orphan's code was usable as-is (see defects below). Example 9 is a
  rewrite of EN `chapter-3.md` Code 4 / JP `chapter-3.md` コード例2 (solvent selectors);
  Examples 1-8 are new, built on CoolProp reference transport properties plus published
  correlations (Lucas, Wilke-Chang, Stokes-Einstein, Ergun, Wakao-Kaguei).

### Defects found in the source material (chapter-8 harvest)

Numbering continues from the list above.

10. **README mis-attribution of orphan chapter 3** and the orphan's own two forward
    references to a transport chapter that was never written. Corrected in this addendum.
11. **Lucas viscosity implemented upside-down** (JP `chapter-1.md`, `SupercriticalFluid.
    viscosity_estimate`): computes `eta = eta0 * xi * fp` where the Lucas group `xi` is an
    *inverse* viscosity scale, so the correlation is `eta0 = [bracket]/xi`. Combined with a
    mislabelled unit (micropoise treated as micropascal-second) the method returns ~2.5e-9
    Pa·s for scCO2 instead of ~1.6e-5 Pa·s — four orders of magnitude low, and inverted in
    its dependence on the fluid constants. Chapter 8 Example 1 implements Lucas correctly
    (including the F_P polarity factor, which the source omits) and agrees with the
    reference correlation to within 1-4%.
12. **Fabricated diffusivity correlation** (JP `chapter-1.md`, `diffusivity_estimate`):
    documented as "a variant of the Wilke-Chang equation" but is `D = 1e-8 * T /
    (eta*1e6 * rho**0.6)`, which is neither Wilke-Chang nor dimensionally consistent, and
    it is fed the broken viscosity above. Dropped. Chapter 8 uses Stokes-Einstein (exact)
    and the actual Wilke-Chang expression, and states the residual uncertainty.
13. **Selector sample output does not match the selector code** (EN `chapter-3.md` Code 4):
    the published sample output shows CO2 scoring 105 with reason "Polarity mismatch", but
    the code's `polarity_match['medium']` list contains `'low'`, so CO2 matches and scores
    +20 with "Polarity match". The sample output also lists only 3 of the 5 database
    entries, silently dropping Propane and Nitrogen. It was written by hand, not run.
14. **Arbitrary additive scoring** (EN `chapter-3.md` Code 4 and JP コード例2): both mix
    hard requirements and preferences into one integer score, so a fluid that violates a
    temperature limit can still outrank one that does not, and the resulting number cannot
    be audited. Example 9 replaces this with a pass/fail constraint filter followed by a
    ranking on computed transport properties only, and prints the rejection reason for every
    eliminated candidate.
15. **`PropsSI(fluid, 'Tcrit')` argument order** recurs in JP `chapter-3.md` コード例1
    (same defect as item 8 above).
16. **Unsourced illustrative data presented as results**: JP `chapter-3.md` contains a
    green-solvent radar chart, an environmental-burden bar chart, a safety risk matrix, a
    biodiesel three-panel comparison, a modifier extraction-yield chart and an economics
    calculator, all built from hard-coded scores and costs with no provenance. The
    qualitative conclusions were carried into prose and tables where they are defensible;
    the plots and the fitted "prediction model" over invented data were dropped rather
    than reproduced with a caveat, because their only content is the invented numbers.

### Deliberately excluded from chapter 8

- **EN `chapter-3.md` Code 1 (pandas SCF database)** and **Code 6 (six-panel comparison
  dashboard)**: hand-tabulated critical constants and 1-5 "polarity/cost/safety" scores.
  Example 8 gets the same critical constants from CoolProp instead, and computes transport
  properties rather than scoring them. Note that the source table's xenon critical density
  (1155 kg/m3) and cost ($8000/kg) are not reproduced here; the density is defensible but
  the price is a moving target and was left as a qualitative "very high".
- **EN `chapter-3.md` Code 3 / JP §3.2.1 water dielectric constant**: dielectric behaviour
  belongs to published chapter 3 (supercritical water), which already covers it, and both
  source implementations use crude self-described approximations. Not a transport property.
- **JP `chapter-3.md` §3.2.2 SCWO Arrhenius kinetics** and **§3.5.4 LCA scoring**: reaction
  kinetics and life-cycle assessment are out of scope for a transport chapter, and the
  Arrhenius parameters (A = 1e13 1/s, Ea = 150 kJ/mol) are given without a source.
- **JP `chapter-3.md` コード例6 process optimisation**: optimises a Gaussian "efficiency"
  function invented for the example, so the optimum is a property of the invented function
  and not of any supercritical process.
- **Xenon transport properties**: CoolProp has no viscosity or thermal-conductivity model
  for xenon. Rather than substitute an estimate, Example 8 catches the exception and prints
  "no model", and the chapter draws the lesson that reference-data availability is itself a
  solvent-selection constraint.

### Overlap check against the published course

Chapter 8 was written after auditing published chapters 1-7 for duplication. Qualitative
transport statements already exist in chapter 1 §1.3 (gas/SCF/liquid property table),
chapter 2 §2.2 (scCO2 viscosity/diffusivity/zero surface tension, and the Wilke-Chang
formula quoted but not implemented) and chapter 3 §3.2 (supercritical water viscosity and
diffusivity). Chapter 8 cross-references these rather than restating them, and in two
places sharpens them: the "10-100x liquid diffusivity" figure of chapters 1-2 is shown to
be 14x at 8 MPa but only 3.9x at 20 MPa, and thermal conductivity — absent from chapters
1-7 except as an aerogel figure and a CoolProp output code — is treated for the first time.
Chapter 7 exercise 1 asks the reader to compare density, viscosity and thermal conductivity
of CO2/ethanol/water at matched reduced conditions; Example 8 is a superset of that
exercise's answer, which the lead may wish to note or reword.

### Related changes in the chapter-8 merge

- `knowledge/{en,jp}/MS/supercritical-fluid-introduction/chapter-7.html`: next-chapter link
  to chapter 8.
- `knowledge/{en,jp}/MS/supercritical-fluid-introduction/index.html`: chapter 8 added to the
  chapter list and learning path; badges 7 -> 8 chapters and 18 -> 27 code examples; total
  time 185-225 -> 220-265 min; a "process design" reading path added.
- `knowledge/{en,jp}/MS/index.html`: series badge 7 Chapters -> 8 Chapters / 27 Examples,
  description extended with transport properties.
- `knowledge/{en,jp}/index.html` and `knowledge/{en,jp}/MS/index.html` header stats:
  regenerated with `scripts/update_index_stats.py --write` (MS 177 -> 178 chapters,
  site total 649 -> 650).
- `knowledge/en/MS/TRANSLATION_STATUS.md`: MS totals 167 -> 168 chapters; the supercritical
  series entry gains `chapter-8.html`.
- `chapter-8.md` generated for both locales with `tools/html_to_md.py`.

All 18 output boxes (9 per locale) were produced by executing the code: CoolProp 8.0.0,
NumPy 2.5.2, SciPy 1.18.0, Matplotlib 3.11.1, pandas 3.0.5, Python 3.13.
