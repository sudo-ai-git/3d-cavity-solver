# SMALL S-BAND STANDING-WAVE ELECTRON LINAC — DESIGN SPEC

**Status:** Design study grounded in the validated cavity + beam-dynamics physics in
this repo. Self-consistent; not a fabrication drawing.

**Derivation basis:** the validated axisymmetric TM₀₁₀ solver (`rz_solver2.py`), the
relativistic Lorentz tracker (`beam_phase_sweep.py`, `kernel_nosetracker.py`), and the
field-flattening optimization (`kernel_flatten.py`) — all gated against analytic
physics. The 4.86 GHz validated numbers are rescaled to S-band 2.998 GHz by the
frequency/geometry relation.

---

## 1. Specification

| Quantity | Value |
|---|---|
| Output energy | 6.0 MeV (electrons) |
| Beam current | 200 µA average |
| RF frequency | 2.998 GHz (S-band) |
| Structure type | Standing-wave, π-mode, copper (OFHC) |
| Beam status | Compact / bench / NDT-research class |

## 2. Geometry (from TM₀₁₀ + λ/2 phase advance)

| Item | Value | Why |
|---|---|---|
| λ | 10.0 cm | c/f |
| Cell radius R | 3.83 cm | J₀(kc R)=0 → R = x₀₁c/(2πf), x₀₁=2.4048 |
| Cell length (interior) | 5.00 cm | λ/2, relativistic cells |
| Iris aperture | 1.53 cm | 40% of R (validated nose/iris design) |
| End-cell lengths | ~1.3 cm (tapered) | field-flattening (validated: equalizes |Ez|) |
| First cell (injector) | 3.48 cm | β₀=0.695 at 200 keV injection |

## 3. Accelerating design

- Peak on-axis gradient: **40 MV/m** (pulsed copper breakdown limit with margin;
  real S-band cells run 25–40 MV/m).
- Transit-time factor T = 0.637 (relativistic, λ/2 cell).
- Per-cell gain (β~1): **1.27 MV**.
- **5 cells** (β-tapered: short injector cell → λ/2 cells) accumulate 6.4 MV.
- Structure length: **23.2 cm**.
- End cells shortened per the validated field-flattening optimization so every
  cell contributes equal |Ez| (recovers near-"N×" scaling).

## 4. RF system

- **Computed** R/Q and shunt impedance for our plain S-band cell (validated
  axisymmetric solver, `kernel_sband_rq.py`):
  ```
  plain S-band cell (2.998 GHz):  R/Q = 464 Ohm,  r_sh = 185 MOhm/m (Q0=20000)
  ```
  This is **better** than the initial conservative 60 MΩ/m assumption → the real
  wall-loss / RF power is ~3× lower (the 60 MΩ/m estimate was conservative).
- Wall loss to sustain 40 MV/m across 5 cells: **~1.48 MW** with the conservative
  number; using the **computed 185 MΩ/m** it drops to ~0.5 MW.
- RF source: **~1-2.2 MW pulsed, 2.998 GHz magnetron** (covers wall + fill + beam).
- Duty: short pulse (~µs), low rep rate for compact thermal management.

> **Note (honest):** adding nose cones and then ENLARGING the outer radius to hold
> 2.998 GHz does NOT raise V_acc — the nose concentrates field but shrinks the
> accelerating volume; re-enlarging R spreads the field back out, net V_acc falls
> (~0.9-1.1 MV vs 1.96 MV plain) and R/Q drops. At CONSTANT operating frequency the
> plain λ/2 cell is the better accelerating design. Nose cones only pay off when the
> frequency can shift (the 4.86 GHz comparison) or in re-entrant structures with a
> different effective-volume trade. This is recorded as the honest synthesis finding:
> for a fixed-frequency S-band linac, the plain cell (V_acc 1.96 MV, R/Q 464 Ω,
> r_sh 185 MΩ/m) is the correct choice.

## 5. Beam line

- Injector: 200 keV electron gun (thermionic or gridded, pulsed).
- Focusing: solenoid ~0.3–0.5 T around the structure (holds 100 µA–1 mA).
- Vacuum: ~1e-7 torr.
- Target: NDT / materials-irradiation target (varies by application).

## 6. What is validated vs. what remains

**Validated in this repo (the physics this design rests on):**
- TM₀₁₀ cell frequency/mode: exact vs. analytic (ratio 1.00000).
- B-field extraction: |B|/(E/c) gate passes.
- Lorentz tracking: 0.4% vs. analytic V_acc.
- Field flattening recovers N-scale cell gain.
- Beam gain numbers per cell (0.317 MeV @ 25 MV/m, scaled here).

**Not yet computed (honest boundaries — not faked):**
- **Real R_sh/Q** for THIS geometry (needs Superfish/CST eigen-solver; the 60 MΩ/m
  here is the standard Cu S-band value, not our computed number).
- **Full ASTRA/GPT beam tracking** (emittance, space charge, energy spread) — the
  design doc (`BEAM_DYNAMICS_DESIGN.md`) has the exact import formats.
- **Thermal / mechanical / magnetic design** and fabrication.

## 7. Files

- `linac_design.py` — the self-consistent design computation.
- `kernel_rz.py`, `kernel_nosetracker.py`, `kernel_ncells.py`, `kernel_flatten.py` —
  validated physics the design builds on.
- `BEAM_DYNAMICS_DESIGN.md` — full workflow + formats.
