#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: Nose-cone accelerating cell — on-axis transit-time factor &
# gradient figure of merit (the "how much stronger" answer), using the
# validated 3D FEM solver.
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

from skfem import (MeshHex, ElementTetN1, Basis, BilinearForm, asm)
from skfem.helpers import dot, curl

@BilinearForm
def stiff(u, v, w): return dot(curl(u), curl(v))
@BilinearForm
def mass(u, v, w):  return dot(u, v)

def assemble_pair(mesh):
    basis = Basis(mesh, ElementTetN1())
    A = asm(stiff, basis); B = asm(mass, basis)
    bd = basis.get_dofs(facets=mesh.boundary_facets())
    b_ext = np.sort(np.unique(np.asarray(bd.all()).ravel()))
    return A, B, basis, b_ext

def solve_dense_v(A, B, free, n_excl=4):
    """Dense generalized solve; returns (freqs_gHz, eigenvectors on free dofs)."""
    from scipy.linalg import eigh
    Aff = A[free][:,free].toarray(); Bff = B[free][:,free].toarray()
    lam, X = eigh(Aff, Bff)
    f = np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    # order ascending, keep lowest 4
    o = np.argsort(f); f=f[o]; X=X[:,o]
    phys = f > 1e8
    return f[phys][:n_excl], X[:, phys][:, :n_excl]

# ---------- geometry ----------
Rc = 0.045; L = 0.05
side = np.sqrt(np.pi)*Rc
nx = ny = 12; nz = 10
print("="*70, flush=True)
print(" NOSE-CONE ACCELERATING CELL — on-axis transit-time & gradient", flush=True)
print("="*70, flush=True)
print(f"cross-section: area-matched square side {side*100:.2f}cm; length L={L*100:.0f}cm", flush=True)

base_mesh = MeshHex.init_tensor(
    np.linspace(0,side,nx+1), np.linspace(0,side,ny+1),
    np.linspace(0,L,nz+1)).to_meshtet()

def nose_pec_dofs(basis, side, L, b_ext, nose_radius_frac=0.40, nose_len_frac=0.35):
    nose_radius = nose_radius_frac*side/2
    nose_len = nose_len_frac*(L/2)
    xyz = basis.doflocs
    b_set = set(b_ext.tolist())
    nd = []
    for i in range(basis.N):
        if i in b_set: continue
        x,y,z = xyz[0,i]-side/2, xyz[1,i]-side/2, xyz[2,i]
        rr = np.hypot(x,y)
        if (rr<=nose_radius and z<=nose_len) or (rr<=nose_radius and z>=L-nose_len):
            nd.append(i)
    return np.unique(np.concatenate([b_ext, np.array(nd)]))

def analyze_cell(nose=False, nose_radius_frac=0.40, nose_len_frac=0.35):
    A, B, basis, b_ext = assemble_pair(base_mesh)
    all_pec = nose_pec_dofs(basis, side, L, b_ext,
                            nose_radius_frac, nose_len_frac) if nose else b_ext
    free = np.setdiff1d(np.arange(A.shape[0]), all_pec)
    f, X = solve_dense_v(A, B, free, n_excl=4)
    if len(f) == 0: return None
    f0 = f[0]
    # reconstruct the full (free) eigenvector for the fundamental on original full
    # dof ordering, zeroing PEC dofs.
    vfull = np.zeros(A.shape[0]); vfull[free] = X[:,0]
    # on-axis line: x=y=center, z from 0..L. Find dofs lying on axis.
    cx = side/2
    axis_dofs = []
    xyz = basis.doflocs
    for i in range(A.shape[0]):
        if abs(xyz[0,i]-cx)<1e-9 and abs(xyz[1,i]-cx)<1e-9:
            axis_dofs.append(i)
    axis_dofs = sorted(axis_dofs, key=lambda i: xyz[2,i])
    zs = np.array([xyz[2,i] for i in axis_dofs])
    Ez = np.array([vfull[i] for i in axis_dofs])
    return dict(f0=f0, z=zs, Ez=Ez, basis=basis)

print("\n[baseline plain cell]", flush=True)
p_plain = analyze_cell(nose=False)
w0p = 2*np.pi*p_plain['f0']
zp, Ezp = p_plain['z'], p_plain['Ez']
# transit-time factor (relativistic beta=1): T = |∫ Ez e^{-i w z/c} dz| / ∫|Ez| dz
# numerical integration along z
wz = w0p/c0*zp
Ii = np.trapezoid(Ezp*np.exp(-1j*wz), zp)
Ir = np.trapezoid(np.abs(Ezp), zp)
Tp = abs(Ii)/ (Ir+1e-30)
Vp = abs(Ii)   # effective accelerating voltage (peak, arbitrary scale)
print(f"  f0 = {p_plain['f0']/1e9:.4f} GHz, on-axis points={len(zp)}", flush=True)
print(f"  transit-time T = {Tp:.4f}   | effective V_acc (arb.) = {Vp:.6e}", flush=True)

print("\n[nose-cone cell]", flush=True)
p_nose = analyze_cell(nose=True)
w0n = 2*np.pi*p_nose['f0']
zn, Ezn = p_nose['z'], p_nose['Ez']
wz = w0n/c0*zn
Ii = np.trapezoid(Ezn*np.exp(-1j*wz), zn)
Ir = np.trapezoid(np.abs(Ezn), zn)
Tn = abs(Ii)/(Ir+1e-30)
Vn = abs(Ii)
print(f"  f0 = {p_nose['f0']/1e9:.4f} GHz, on-axis points={len(zn)}", flush=True)
print(f"  transit-time T = {Tn:.4f}   | effective V_acc (arb.) = {Vn:.6e}", flush=True)

print("\n" + "="*70, flush=True)
print(" RESULT — 'stronger' in accelerator terms", flush=True)
print("="*70, flush=True)
print(f"  fundamental:       {p_plain['f0']/1e9:.4f} -> {p_nose['f0']/1e9:.4f} GHz", flush=True)
print(f"  transit-time factor: {Tp:.4f} -> {Tn:.4f}  ({100*(Tn-Tp)/Tp:+.1f}%)", flush=True)
print(f"  effective V_acc:    {Vp:.3e} -> {Vn:.3e}  ({100*(Vn-Vp)/Vp:+.1f}%)", flush=True)
print()
print("  (V_acc = |∫ E_z(z) e^{-i w z / c} dz|, the on-axis accelerating voltage", flush=True)
print("   the beam gains; nose cones re-concentrate E on the axis, raising it)", flush=True)
print("DONE", flush=True)
