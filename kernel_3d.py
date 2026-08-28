#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: 3D Maxwell cavity eigen-solver (scikit-fem Nedelec edge elements)
#   1) validated against analytic box
#   2) applied to an accelerating cell: plain vs re-entrant nose-cones
import numpy as np

c0 = 299792458.0

# ---------- scikit-fem ----------
import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-fem"])
    import skfem
print(f"scikit-fem {skfem.__version__}", flush=True)

from skfem import (MeshHex, MeshTet, ElementTetN1, Basis, BilinearForm, asm)
from skfem.helpers import dot, curl

@BilinearForm
def stiff(u, v, w): return dot(curl(u), curl(v))
@BilinearForm
def mass(u, v, w):  return dot(u, v)

def assemble_pair(mesh):
    """Return A (curl-curl), B (mass), Basis, exterior-boundary dofs."""
    basis = Basis(mesh, ElementTetN1())
    A = asm(stiff, basis); B = asm(mass, basis)
    bd = basis.get_dofs(facets=mesh.boundary_facets())
    b_ext = np.sort(np.unique(np.asarray(bd.all()).ravel()))
    return A, B, basis, b_ext

def solve_dense(A, B, free, fmin=1e8):
    from scipy.linalg import eigh
    Aff = A[free][:, free].toarray(); Bff = B[free][:, free].toarray()
    lam, _ = eigh(Aff, Bff)
    f = np.sort(np.sqrt(np.maximum(lam, 0))*c0/(2*np.pi))
    return f[f > fmin]

# ---------------------------------------------------------------------------
print("="*70); print(" [1] VALIDATE: rectangular PEC box vs analytic TM/TE"); print("="*70)
a, b, d = 0.30, 0.20, 0.25
mesh_box = MeshHex.init_tensor(
    np.linspace(0,a,7), np.linspace(0,b,5), np.linspace(0,d,6)).to_meshtet()
A, B, basis, b_ext = assemble_pair(mesh_box)
free = np.setdiff1d(np.arange(A.shape[0]), b_ext)
fs = solve_dense(A, B, free)
an = sorted(set(round(c0/2*np.sqrt((m/a)**2+(n/b)**2+(p/d)**2), 1)
                for m in range(5) for n in range(5) for p in range(5)
                if m+n+p >= 2))[:4]
print(f"dofs={A.shape[0]} free={len(free)}", flush=True)
print("analytic:", [f"{x/1e9:.4f}" for x in an], flush=True)
print("numeric :", [f"{x/1e9:.4f}" for x in fs[:4]], flush=True)
errs = [100*abs(fs[i]-an[i])/an[i] for i in range(3)]
print(f"lowest-3 mode err: {[round(e,2) for e in errs]}%", flush=True)

# ---------------------------------------------------------------------------
print(); print("="*70); print(" [2] ACCELERATING CELL: plain vs re-entrant nose-cones"); print("="*70)
Rc = 0.045; L = 0.05
side = np.sqrt(np.pi)*Rc    # area-matched square cross-section side  (~7.98 cm)
print(f"cylinder Rc={Rc*100:.2f}cm -> area-matched square side {side*100:.2f}cm, len {L*100:.0f}cm",
      flush=True)
nx = ny = 12; nz = 10
base_mesh = MeshHex.init_tensor(
    np.linspace(0,side,nx+1), np.linspace(0,side,ny+1),
    np.linspace(0,L,nz+1)).to_meshtet()
A, B, basis, b_ext = assemble_pair(base_mesh)
free0 = np.setdiff1d(np.arange(A.shape[0]), b_ext)
fs0 = solve_dense(A, B, free0)
f0 = fs0[0] if len(fs0) else float('nan')
print(f"PLAIN cell   : TM010-like f0 = {f0/1e9:.4f} GHz", flush=True)

# re-entrant nose cones: PEC cylinders protruding from each end wall toward
# the center, leaving a central aperture for the beam.
nose_radius = 0.40*side/2   # base radius
nose_len    = 0.35*(L/2)    # each nose reaches toward mid-plane
xyz = basis.doflocs
b_ext_set = set(b_ext.tolist())
nose_dofs = []
for i in range(A.shape[0]):
    if i in b_ext_set:
        continue
    x, y, z = xyz[0,i]-side/2, xyz[1,i]-side/2, xyz[2,i]
    rr = np.hypot(x, y)
    if (rr <= nose_radius and z <= nose_len) or (rr <= nose_radius and z >= L-nose_len):
        nose_dofs.append(i)
all_pec = np.unique(np.concatenate([b_ext, np.array(nose_dofs)]))
freeN = np.setdiff1d(np.arange(A.shape[0]), all_pec)
fsN = solve_dense(A, B, freeN)
fN = fsN[0] if len(fsN) else float('nan')
print(f"NOSE cell    : TM010-like f0 = {fN/1e9:.4f} GHz", flush=True)
if np.isfinite(f0) and np.isfinite(fN):
    print(f"fundamental shift: {100*(fN-f0)/f0:+.2f}%  "
          f"(nose removes volume; freq change reflects field rearrangement toward axis)",
          flush=True)
print(); print("DONE", flush=True)
