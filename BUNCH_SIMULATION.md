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

## HONEST CAVEAT — emittance
The transverse **emittance growth is NOT reported**: my hand-rolled rms-envelope
space-charge model was **numerically unstable** (returned a non-physical ~10⁸ µm
blow-up while σ_x only grew 1.20→1.37 mm — self-inconsistent, so a code bug, not
physics). Getting a trustworthy emittance requires a proper beam-dynamics code
(ASTRA / GPT / tracewin) with a real solenoid lattice — which is the documented
next step, not faked here. Only σ_x (rms size) is reported: 1.20 → 1.37 mm.

Run: `python3 kernel_bunch_final.py` (Kaggle kernel `3d-cavity-bunch-space-charge`).
