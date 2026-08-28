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

- `Nue`=frequency, `MaxE`=amplitude, `Phi`=phase.
- **Auto-phasing is ON by default**: ASTRA scans the reference particle's gain and sets the
  working phase to *max-gain*; user `Phi` is then *relative*. Set `Auto_Phase=False` to
  take control.
- Phase `ω·t+φ` **increases with time**, so the bunch **tail sees a higher phase than the
  head**; give the tail more energy via **negative `Phi`**.
- **Link to our figures of merit:** `V_acc`/T we computed are the *ideal-phase* field
  integral; ASTRA's auto-phasing finds the true working phase for a real bunch —

---

## 4. GPT — confirmed facts and honest gaps

**Honesty note:** the GPT manual (containing GPT's exact field-file ASCII layout) is
**licensed, not public**. pulsar.nl documents that GPT imports fields from **Superfish**,
**TOSCA/OPERA**, and a **generic/GDF ASCII** format via field-map elements, with the cavity
element taking frequency + normalization + phase — but the exact native layout requires
the licensed manual. We will not fabricate it.

**Best open reference:** Cornell's **Bmad** tracking code reads GPT-style field maps and is
public — the closest authoritative open specification. *(Being confirmed by parallel
research at time of writing.)* The **workflow is identical to ASTRA up to the file writer**,
so the hard-won asset (validated E/B arrays + correct B extraction) is code-independent;
only the thin serialization differs.

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

## 7. Status

| Item | Status |
|---|---|
| Validated 3D FEM cavity solver | ✅ done |
| Nose-cone cell (transit-time +0.23, V_acc +0.10) | ✅ done |
| Design sweep optimum | ✅ done |
| E field extraction | ✅ validated |
| **B field extraction (gate: B/(E/c)≈1–2)** | ⚠️ **failed at 7.8× — must redo on finer grid** |
| Beam tracker (in-repo) | ❌ abandoned — would have been wrong |
| ASTRA field export | ✅ spec verified from manual |
| GPT field export | ⚠️ exact layout licensed; Bmad as open reference |
| Real ASTRA/GPT particle run | ⛔ **future work** — requires installing ASTRA/GPT |
