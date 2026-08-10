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
