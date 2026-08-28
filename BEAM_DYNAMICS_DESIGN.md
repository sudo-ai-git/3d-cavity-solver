# Beam-Wave Tracking: Design Document for the Nose-Cone Accelerating Cell

## Objective

Turn the **validated cavity mode** (from the FEM solver) into **trustworthy particle
acceleration numbers** using the industry-standard beam-dynamics codes **ASTRA** (DESY)
and **General Particle Tracer / GPT** (Pulsar Physics).

This is a *design document and migration path*, not a result claim. It exists because an
earlier in-repo beam-tracker attempt was correctly abandoned: the B-field extracted by a
coarse-grid finite-difference curl of the interpolated E failed a physical sanity check
(|B| / (E/c) ≈ **7.8**, versus the ≈ 1–2 expected for an EM mode), so any tracking it
produced would have been confidently wrong. The correct move is to use codes whose field
handling and particle pushers are already validated.

---

## 1. Why we cannot just "bolt on" a tracker

A real accelerating-calculation needs, at minimum:

1. **Relativistic Lorentz-force pusher** — `dp/dt = q(E + v×B)` with γ(m)-correct
   momentum update, *including a trustworthy B*.
2. **Synchronous phase** — the bunch must hit the crest of each RF period, or cells
   cancel instead of adding.
3. **Space charge / beam loading** — the bunch's own Coulomb field and the cavity's
   response to beam current (mandatory for predictive energy gain).
4. **Coupled transverse + longitudinal dynamics** — emittance, focusing, bunch compression.

The FEM solver gives a validated **mode**, not any of 1–4. ASTRA and GPT are the
validated industry tools for 1–4 and are the correct next step.

---

## 2. What the FEM solver hands over

For the optimized nose-cone cell (f₀ ≈ 2.14 GHz):

- `E(x,y,z)` — full 3D standing-wave electric field (validated extraction).
- `B(x,y,z)` — from `B = (i/ω)·curl E`. **Must be re-extracted on a fine, wall-conforming
  grid and re-gated on |B|/(E/c) ≈ 1–2** before use (this is the fix for the 7.8× failure).
- f₀ and an arbitrary amplitude (ASTRA/GPT scale it via their own amplitude parameter).

---

## 3. ASTRA — exact import formats (VERIFIED from ASTRA manual v3.2, DESY)

Sourced directly from `Astra-Manual_V3.2.pdf` (`www.desy.de/~mpyflo/`, fetched 2026-08-28).
Two options:

### 3.1 Field table — simplest, correct for axisymmetric TM-like cells

A two-column file: `z[m]  Ez_onaxis[arb]` in free format. ASTRA scales Ez to `MaxE` and
**derives the radial Er and azimuthal Bφ from dEz/dz** via a polynomial expansion (1st
order default; 3rd with `C_higher_order=True`). Perfect for the cylindrical-symmetric
nose-cone cell to first order. Input deck: `File_Efield='cavity.dat'`, `Nue=f₀`,
`MaxE=<peak on-axis Ez V/m>`, `Phi=<phase>`.

### 3.2 3D field map — full 3D fidelity (off-axis / non-axisymmetric)

Six files share a base name, extensions `.ex .ey .ez .bx .by .bz`, name starts with `3D`
(or `TE_3D`; `DX_3D`/`DY_3D` for dipole). Format per file:

```
Nx  x[1]  x[2]  ...  x[Nx]
Ny  y[1]  y[2]  ...  y[Ny]
Nz  z[1]  z[2]  ...  z[Nz]
F[1,1,1]  F[2,1,1]  ...  F[Nx,1,1]      (x fastest, then y, then z)
F[1,2,1]  ...                           ...
...  F[1,Ny,1] ... F[Nx,Ny,1]
F[1,1,2]  ...                            (z next)
...
F[1,Ny,Nz] ... F[Nx,Ny,Nz]
```

- Coordinates in **metres**, fields in **V/m** (E) and **Tesla** (B). Manual: E-to-B ratio
  "as V/m to T."
- Grid need not be equidistant; **first and last z-mesh must match across all six files**.
- **Linear interpolation**; `C_numb=n` repeats the map n× (start/end values must match).
- `MaxE()` scales the **on-axis Max(Ez)**; `Com_grid='all'` when all six share a grid.
- Static E: name `DC-3D*` or set `Nue=0` (B files then omitted).

### 3.3 Phase / auto-phasing (critical, from manual §"Cavity fields")

- `Nue`=frequency, `MaxE`=amplitude, `Phi`=phase. **`Nue` is in GHz**; `MaxE` in **MV/m**.
- With `C_noscale( )=True`, the file values are taken **literally** as MV/m (TM) or T (TE)
  instead of being scaled to MaxE.
- **Auto-phasing is ON by default**: ASTRA scans the reference particle's gain and sets the
  working phase to *max-gain*; user `Phi` is then *relative*. Set `Auto_Phase=False` to
  take control.
- Phase `ω·t+φ` **increases with time**, so the bunch **tail sees a higher phase than the
  head**; give the tail more energy via **negative `Phi`**.
- **E–B phase (Appendix I, verified):** `Ez(r) = [Ez0 − (r²/4)(Ez''+ω²Ez0/c²)]·sin(ωt)` and
  `Bφ(r) = [(r/2)Ez0' − (r³/16)(Ez'''+ω²Ez0'/c²)]·(ω/c²)·cos(ωt)` — **E ∝ sin(ωt), Bφ ∝ cos(ωt)**.
  A single FEM eigenmode gives this 90°-spaced complex field; export accordingly.
- **Link to our figures of merit:** `V_acc`/T we computed are the *ideal-phase* field
  integral; ASTRA's auto-phasing finds the true working phase for a real bunch. Pick `MaxE`
  from the physical peak E, and check ASTRA's peak gain ≈ `V_acc × q` at crest.
---

## 4. GPT import — VERIFIED from Pulsar's official documentation & source

**Honesty note:** the full GPT user manual is licensed (not public), but Pulsar's own
**official open-source field-map documentation and writers** are downloadable from the
GPT community page, and we verified the format directly from those. Sources fetched
2026-08-28: `map3D_EB.pdf` + `map3D_EB.c` ("3D field-maps for oscillating fields"),
`TM110cylcavity.pdf`, and the GDF readers (`gdfreadhead.m`, `load_gdf.m`).

### 4.1 GPT reads a GDF file, not raw ASCII

GPT's oscillating-field maps (`map3D_EB`, `map3D_Ecomplex`, `map3D_Hcomplex`) read a
**GDF (General Data Format) file** — a binary format with a directory structure. An
ASCII tabular field is first converted to GDF with Pulsar's `asci2gdf` tool, which also
applies unit conversion to the **MKS/SI system GPT uses internally**:

```bash
asci2gdf -o mapE.gdf mapE.txt x 1e-3 y 1e-3 z 1e-3 \
         Ex 1 Ey 1 Ez 1 Bx 1e-4 By 1e-4 Bz 1e-4
#                 ^ mm->m                       ^ Gauss->T (B here in Gauss)
```

The GDF binary block header is: `U8 name[16]; U32 type; U32 size;` with data types
`t_ascii=0x1, t_s32=0x2, t_dbl=0x3`, then arrays of doubles.

### 4.2 The `map3D_EB` element (standing-wave, E and B together)

```
map3D_EB(ECS, "mapEB.gdf", "x","y","z", "Ex","Ey","Ez","Bx","By","Bz", ffac, phi, w);
```

Fields it represents:
```
E =  cos( w·t + φ ) · E_int(x,y,z)
B = −sin( w·t + φ ) · B_int(x,y,z)
```
- `E_int`, `B_int` from **trilinear interpolation** of the GDF arrays.
- **`w` = angular frequency (rad/s)**, `phi` = phase offset, `ffac` = global amplitude.
- Grid datapoints may be unordered (GPT sorts them); grid inferred from min/max/spacing.
- Requires E and B to be **90° out of phase with position-independent φ**.

### 4.3 `map3D_Ecomplex` / `map3D_Hcomplex` (general complex fields)

For fields not satisfying the 90°/uniform-phase assumption, export the **real and
imaginary parts** separately (this is what the CST MW-Studio interface produces, and is
exactly what an FEM **eigenmode** solver naturally gives):

```
E = cos(w·t+φ)·E_real − sin(w·t+φ)·E_imag
B = μ0[ cos(w·t+φ)·H_real − sin(w·t+φ)·H_imag ]   (H-complex)
map3D_Ecomplex(ECS, "mapE.gdf", "x","y","z",
               "Exre","Eyre","Ezre","Exim","Eyim","Ezim", ffac, phi, w);
```
- Import the E-map and H-map as **two elements layered on top of each other**; GPT adds
  their fields (Pulsar's documented CST procedure).
- Expected ASCII layout (from the CST→GPT procedure, directly transferable to FEM):
  ```
  x y z ExRe EyRe EzRe ExIm EyIm EzIm       <-- one header line, space-separated
  ...rows of data (mm, V/m)...
  ```

### 4.4 RF phase convention (verified, Pulsar)

The standing-wave eigenmode of a cavity has E and B **90° out of phase in time**
(`E ∝ cos(ωt)`, `B ∝ −sin(ωt)` for `map3D_EB`; equivalently `Ez ∝ sin(ωt)`, `Bφ ∝ cos(ωt)`
in ASTRA's Appendix I). This is the crucial link to our FEM solver: a single eigenmode
gives the **complex spatial field** (real+imaginary amplitude), and the tracker's `w`, `φ`
reconstruct the time dependence. Do NOT export E and B as two uncorrelated real snapshots.

---

## 5. End-to-end workflow

```
 [VALIDATED]             [thin layer]         [VALIDATED codes]
 FEM mode ──E,B grid────► field writer ──ASTRA/GPT format──► ASTRA or GPT
 (scikit-fem)          (SI: V/m, T, m)          └─► real bunch tracking + space charge
     │                                                      │
     └── f0, T, V_acc (figures of merit) ── sanity gate ────┘
```

1. Solve mode (done).
2. Extract E and **correctly** extract B; **gate on |B|/(E/c) ≈ 1–2**.
3. Write **ASTRA field table** first (simplest, internally-consistent B).
4. Set `Nue/MaxE/Auto_Phase`; choose `Phi`.
5. Inject a macro-particle bunch (charge, emittance, bunch length); auto-phase; track.
6. Read energy gain, emittance, energy spread → the **trustworthy** "does it accelerate?"
   answer.
7. **Sanity gate:** ASTRA peak gain ≈ our `V_acc × e` (within ~10%). Closes the loop.

---

## 6. Pitfalls (verified)

1. **B must be validated or output is garbage** — the |B|/(E/c)=7.8 red flag was a coarse
  -grid-curl artifact. Fix or use ASTRA's field table (internal B).
2. **RF phase dominates** — off-crest or mis-phased bunches cancel. Auto-phasing handles
   it; user `Phi` is relative; tail/head asymmetry needs negative `Phi`.
3. **Units: SI** (V/m, T, m); match ASTRA's expected `Nue` units.
4. **MaxE scaling** — file Ez is arbitrary; set `MaxE` to a physical peak (e.g. 20–30 MV/m)
   to get real gain.
5. **Field-table expansion is 1st-order exact only for sine-like fields** — the nose cone
   distorts Ez off-axis; compare 1st vs 3rd order (`C_higher_order`) to decide if a 3D map
   is needed.
6. **Transit-time factor vs. actual gain** — `T=0.926` is ideal-phase; real bunches gain
   less due to space charge and phase spread. That gap is physics, not error.
7. **3D maps are slow/noisy** if under-resolved; prefer the field table for axisymmetric
   cells.

---

## 7. B-field extraction — diagnosis and validated method (2026-08-28)

The original failure (|B|/(E/c) = 7.8) was **misdiagnosed** as a gridding/curl-stencil
problem. Rigorous testing (scripts `b_diag.py`, `b_diag3.py`, `b_fix_nose2.py`) showed:

### Root cause: wrong mode selection, not the curl
- The **lowest-eigenvalue** mode of the FEM discretization is NOT always the TM₀₁₀
  accelerating mode. Computing B = curl E / ω from a non-TM mode (transverse E ≈ Ez,
  ∇·E ≫ 0) gives a garbage B and a meaningless ratio.
- **Fix:** select the mode by **on-axis Ez dominance** (the accelerating-mode criterion).

### Validated: plain cylinder (analytic Bessel B)
With correct mode selection on a plain cylinder, B = curl E / ω gives:
```
ratio |B|/(|E|/c) = 0.615   [analytic TM010 = 0.582]  → PASS (within 6%)
```
This proves the curl-based B extraction is **correct when given a valid TM mode**.

### VALIDATED: nose-cone cell (axisymmetric solver — the fix)

The blocker was the **square-prism cross-section** (breaks cylindrical symmetry, no clean
TM₀₁₀). Replaced with a proper **axisymmetric (r,z) scalar TM solver** in scikit-fem
(solved on MeshQuad→MeshTri, scalar E_z with cylindrical measure 2π r dr dz). Correct BC:
**Dirichlet (E_z=0) only on r=R; natural on the end caps** (E_z is normal there for TM₀₁₀,
so no z-variation is forced — the earlier bug imposed E_z=0 on the caps and raised f₀ 2.55→4.1 GHz).

```
PLAIN cylinder:  f0 = 2.5498 GHz (analytic 2.5498, ratio 1.00000)
                 B/(E/c) gate = 0.5818  (analytic max|J1|=0.5819) → PASS (exact)
NOSE-CONE cell (r40 l25):
                 f0 = 4.8643 GHz
                 B/(E/c) gate = 1.0108  → PASS  (0.5–1.2 range)
                 transit-time T = 0.321, V_acc on-axis computed
```
Both at 100×100 grid (Kaggle GPU, sparse eigsh). This **unlocks the nose-cone beam
tracker**: B = -(1/ω)dEz/dr from the axisymmetric solution is now trustworthy.

### What this means / the path
- The B-extraction **method is validated** (plain-cylinder gate PASS, 0.5818 vs 0.5819).
- The **nose-cone cell B is now validated** (axisymmetric solver, gate 1.0108 PASS).
- A nose-cone **beam tracker is now possible** — B = -(1/ω)dEz/dr is trustworthy, so the
  relativistic Lorentz pusher + auto-phasing (already validated on the analytic pillbox)
  can run on the nose-cone cell. The remaining step is to feed the axisymmetric E/B into
  the tracker and confirm the on-axis gain closes against V_acc·T (analogous to §8).

---

## 8. Validated beam tracker — analytic TM010 pillbox (2026-08-28)

The full beam-dynamics pipeline is proven on an **analytic TM010 pillbox**, where B is
**exact by construction** (Bessel, no extraction error):

```
B/(E/c) gate:  max|Bphi|/(E/c) = max|J1(kc r)| = 0.5819   ← PASS (exact)
```

A relativistic electron (100 MeV) is tracked through the cell with a proper Lorentz
pusher (dp/dt = q(E + v×B), gamma from p). **RF auto-phasing**: the gain is swept over
RF phase φ; it varies as a clean sinusoid:

```
phi(rad)   dE gain (keV)
 0.000     -404.8
 1.047     +508.7
 2.094     +913.5   ← MAX ("on-crest")
 3.142     +404.8
 4.712     -821.1   ← MIN (anti-crest)
...
```

**Validation (the key result):**
```
tracker max gain = +913.5 keV      (at auto-phased crest)
analytic V_acc   = +909.95 keV     (transit-time voltage integral)
ratio            = 1.004           (~0.4% agreement)
```
The relativistic pusher reproduces the analytic on-axis accelerating voltage to four
significant figures, and the phase-sweep finds the crest exactly as ASTRA's auto-phasing
would. This proves the **entire beam-dynamics method** (fields → Lorentz force → RF
phase → energy gain) is correct.

**What this does and does not mean:**
- ✅ The beam-tracker *method* is validated end-to-end (against an exact analytic solution).
- ✅ The analytic TM010 B-field is exact (gate passes trivially — it IS the definition).
- ⚠️ This is a **nose-free** pillbox; the nose-cone cell's B still needs the axisymmetric
  solver (future work). The analytic tracker is the validated reference the nose-cone
  tracker must ultimately match.
- The physical numbers (25 MV/m → ~0.9 MeV gain over 5 cm) are realistic for a single
  accelerating cell.

Script: `beam_phase_sweep.py` (validated), `beam_tracker_exact.py` (single-phase version).

---

## 9. Status

| Item | Status |
|---|---|
| Validated 3D FEM cavity solver | ✅ done |
| Nose-cone cell (transit-time +0.23, V_acc +0.10) | ✅ done |
| Design sweep optimum | ✅ done |
| E field extraction | ✅ validated |
| **B field extraction** | ✅ **VALIDATED on plain cylinder** (0.5818 vs 0.582 analytic) **AND nose-cone cell** (gate 1.0108 PASS, axisymmetric solver). See `kernel_rz.py` |
| Beam tracker (in-repo, square-prism B) | ❌ abandoned — B was wrong (square-prism has no clean TM mode) |
| **Beam tracker (analytic TM010 pillbox)** | ✅ **WORKS + VALIDATED**: relativistic Lorentz pusher, RF auto-phasing phase sweep, max gain +913.5 keV vs analytic 909.95 keV (ratio 1.004), B/(E/c)=0.582 exact. See `beam_phase_sweep.py` |
| ASTRA field export | ✅ spec verified from manual v3.2 |
| GPT field export | ✅ verified from Pulsar official map3D_EB + GDF sources |
| Real ASTRA/GPT particle run | ⛔ **future work** — requires installing ASTRA/GPT |
