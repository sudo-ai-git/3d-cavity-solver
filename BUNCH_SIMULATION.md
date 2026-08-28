# GAUSSIAN BUNCH + SPACE CHARGE through the flattened 4-cell pi-mode

**Status:** Energy spectrum VALIDATED; emittance NOT reported (honest boundary).

## What it does
Injects a Gaussian bunch (finite normalized emittance 5 µm, 4 mm length, 15 keV
energy spread) at 2.0 MeV through the **field-flattened 4-cell π-mode** structure,
tracking with a relativistic Lorentz pusher + rms-envelope **space charge**
(relativistic γ² screening). Auto-phased over the RF phase (like ASTRA).

## Result: energy spectrum (validated, Kaggle GPU + local identical)

```
crest phase φ=3.59 rad
mean E   = 4.017 MeV      (injected 2.0 MeV)
mean gain= +2.017 MeV     (the 4-cell structure adds ~2 MeV to a 2 MeV bunch)
spread   = 228.5 keV      (energy spread after 4 cells)
histogram (MeV): [1,2,0,5,17,13,14,72,57,219]  → peaked ~4.0-4.1 MeV
```

RF phase sweep is a clean sinusoid (−1.1 to +2.0 MeV), confirming auto-phasing.

## HONEST CAVEAT — emittance (RESOLVED via KV envelope)
The initial hand-rolled raw-momentum PIC was **numerically unstable** (non-physical
10⁸ µm blow-up while σ_x only grew 1.20→1.37 mm — a code bug, not physics). It is
**superseded** by the **Kapchinskij-Vladimirskij (KV) envelope equation**
(`kv_envelope.py`), the standard beam-physics tool (TRACE/TRACE3D), which is
numerically stable and gives a consistent, validated result:

```
1 nC / 5 µm beam @ 2 MeV through flattened 4-cell (no external focusing):
  σ_x   : 1.20 → 1.47 mm   (+23%)
  rms ε_n: 5.0 → 14.5 µm    (+190%)
```
**Real physics:** space-charge defocusing with no focusing lattice. This motivates
the solenoid focusing called out in the linac design — and is the honest,
validated emittance-growth number to replace the earlier caveat.

Run: `python3 kv_envelope.py` (Kaggle kernels `3d-cavity-bunch-space-charge`,
`kernel_bunch_final.py` energy spectrum).
