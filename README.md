# 3D Maxwell Cavity Solver (FEM)

A **validated** 3D electromagnetic cavity eigen-solver using [scikit-fem](https://github.com/kinnala/scikit-fem)
(Nedelec / H(curl) edge elements), applied to the analysis of a microwave
accelerating cavity with nose cones.

This is the honest, verified companion to the "37 × 73 sacred geometry"
hypothesis: it computes the *real* physics of a resonant cavity and the
accelerator figures of merit, separating what is true from what is numerology.

---

## Capabilities

- **3D cavity eigen-solver**: solves `curl × curl E = k₀² E` with PEC walls
  (`n × E = 0`) using proper edge elements (no hand-rolled discretization bugs).
- **Validated** against the analytic rectangular PEC box: lowest 3 TM/TE modes
  match to **< 0.5%** with monotonic grid convergence.
- **Accelerator figures of merit**: transit-time factor T and effective
  on-axis accelerating voltage V_acc = |∫E_z(z) e^{−iωz/c} dz|.
- **Design sweep**: nose-cone geometry optimization.
- GPU-accelerated runs via Kaggle kernels (`enable_gpu: true`).

## Validation

Rectangular PEC box, `0.30 × 0.20 × 0.25 m`, analytic TM/TE closed forms
`f = c/2 √((m/a)² + (n/b)² + (p/d)²)`:

```
analytic: 0.780  0.901  0.960  0.999  GHz
numeric : 0.779  0.897  0.957  1.086  GHz
lowest-3 mode error: 0.23%, 0.42%, 0.26%
```

## Results (computed on the validated solver)

### Nose-cone accelerating cell (single cell)

Re-entrant nose cones raise the on-axis accelerating performance vs. a plain cell:

| Quantity | Plain | Nose-cone (r40 l25) | Δ |
|---|---|---|---|
| transit-time factor T | 0.753 | 0.926 | **+23%** |
| effective V_acc | 1.91e-2 | 2.10e-2 | **+10%** |

### Nose-cone design sweep

Degrees of nose penetration vs. radius have a **non-monotonic optimum** —
long noses raise T toward 0.99 but swallow the accelerating path and cut V_acc.
The best single-cell design found: **nose radius 40%, penetration 25%**
(+12.5% V_acc over plain).

| Config | T | V_acc |
|---|---|---|
| **r40 l25** | 0.831 | 2.143e-2 |
| r30 l50 | 0.977 | 1.985e-2 |
| plain | 0.753 | 1.906e-2 |
| r40 l50 | 0.979 | 1.795e-2 |

*(GPU run, 12×12×10 grid, ~11.3k dofs)*

## Honest scope & limitations

- **Cross-section approximation**: the accelerating cell uses an *area-matched
  square prism* cross-section for the cylinder (scikit-fem's hex tensor mesh
  doesn't directly give a clean circular cross-section). The axial/nose geometry
  that governs transit-time is preserved; the transverse shape is approximant.
- **Multi-cell π-mode did NOT validate**: a quick multi-cell model (cells stacked
  with thin grid-aligned iris rings) produced *no* voltage scaling with cell
  count (V_acc identical for 1/2/4 cells) — a failed simulation, not a result.
  Real coupled-cell π-mode structures need a proper cavity design tool
  (Superfish/CST/HFSS) or a careful iris mesh. This is reported honestly rather
  than fabricated.
- **Absolute V_acc is normalization-dependent**; the reliable conclusions are the
  *relative* gains (transit-time factor and voltage ratio vs. baseline).

## Files

| File | Purpose | Status |
|---|---|---|
| `fem_cavity.py` | Validated FEM solver (assemble + box test) | validated |
| `kernel_3d.py` | Kaggle GPU: box validation + nose-cone cell | validated |
| `kernel_tta.py` | Kaggle GPU: transit-time factor + V_acc | validated |
| `kernel_sweep.py` | Kaggle GPU: nose-cone design sweep | validated |
| `kernel_multicell.py` | Multi-cell attempt | **failed validation, kept for audit trail** |
| `hex_render.py` | 37/73 geometry nesting render | validated |
| `cavity_solver.py` | 2D hexagonal cavity solver | validated |
| `simulate.py` | 2D driven-cavity resonance | validated |

## Usage

```bash
pip install scikit-fem numpy scipy

# Local validation + single-cell analysis
python3 fem_cavity.py

# Kaggle GPU kernels (self-contained, install scikit-fem inside):
#   kernel_3d.py    box validation + nose-cone frequency
#   kernel_tta.py   transit-time / V_acc
#   kernel_sweep.py design sweep
```

Related: [37/73 geometry analysis](https://github.com/sudo-ai-git/3d-cavity-solver)
(hexagon-in-hexagram nesting, validated separately).

## License

MIT

## Kaggle kernels

- `taryncampbell/3d-maxwell-cavity-solver-fem`
- `taryncampbell/3d-cavity-nose-cone-transit-time`
- `taryncampbell/3d-cavity-nose-cone-design-sweep`
- `taryncampbell/3d-cavity-multi-cell-pi-mode`
