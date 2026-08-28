#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: MULTI-CELL π-mode accelerating structure.
# Stack the optimized nose-cone cell (r=40%, l=25%) end-to-end; the beam
# crosses one cell per RF half-period so the on-axis voltage ADDS across
# cells. Model N=1,2,4 cells; compute the π-mode fundamental and total
# accelerating voltage + effective gradient.
import numpy as np

c0 = 299792458.0
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
def stiff(u,v,w): return dot(curl(u),curl(v))
@BilinearForm
def mass(u,v,w):  return dot(u,v)

def assemble_pair(mesh):
    basis = Basis(mesh, ElementTetN1())
    A=asm(stiff,basis); B=asm(mass,basis)
    bd=basis.get_dofs(facets=mesh.boundary_facets())
    b_ext=np.sort(np.unique(np.asarray(bd.all()).ravel()))
    return A,B,basis,b_ext

def build_ncell(side, Lcell, n_cells, nz_per_cell, nose_radius_frac=0.40, nose_len_frac=0.25):
    """N-cell structure: n_cells identical cells along z separated by thin
    conducting irises. Total length = n_cells*Lcell. Noses on each cell end."""
    Ltotal = n_cells*Lcell
    nz = nz_per_cell*n_cells
    return MeshHex.init_tensor(
        np.linspace(0,side,13),
        np.linspace(0,side,13),
        np.linspace(0,Ltotal,nz+1)).to_meshtet()

def analyze(mesh, side, Lcell, n_cells, nose_radius_frac, nose_len_frac, A,B,basis,b_ext):
    """Scan the lowest physical modes and return the ONE that gives maximum
    on-axis accelerating voltage (the phase-matched accelerating mode).
    In a coupled multi-cell cavity, the naive lowest mode is the 0-mode (cells
    in phase -> partial V cancellation); the useful accelerating mode is the
    π-mode (cells anti-phase -> voltages add). We find it by maximizing V_acc
    across the lowest modes."""
    Ltotal = n_cells*Lcell
    iris_radius = nose_radius_frac*side/2
    xyz=basis.doflocs
    b_set=set(b_ext.tolist())
    extra=[]
    for k in range(1, n_cells):
        z_plane = k*Lcell
        for i in range(A.shape[0]):
            if i in b_set: continue
            x,y,z=xyz[0,i]-side/2, xyz[1,i]-side/2, xyz[2,i]
            if abs(z - z_plane) < 1e-9 and np.hypot(x,y) >= iris_radius:
                extra.append(i)
    nose_radius = nose_radius_frac*side/2
    nose_len = nose_len_frac*(Lcell/2)
    for k in range(n_cells):
        z0 = k*Lcell; z1 = (k+1)*Lcell
        for i in range(A.shape[0]):
            if i in b_set or i in set(extra): continue
            x,y,z=xyz[0,i]-side/2, xyz[1,i]-side/2, xyz[2,i]
            rr=np.hypot(x,y)
            if (rr<=nose_radius and abs(z-z0)<=nose_len) or (rr<=nose_radius and abs(z-z1)<=nose_len):
                extra.append(i)
    all_pec=np.unique(np.concatenate([b_ext, np.array(extra)]))
    free=np.setdiff1d(np.arange(A.shape[0]), all_pec)
    from scipy.linalg import eigh
    Aff=A[free][:,free].toarray(); Bff=B[free][:,free].toarray()
    lam,X=eigh(Aff,Bff)
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    o=np.argsort(f); f=f[o]; X=X[:,o]
    phys=f>1e8
    nphys = len(f[phys])
    nscan = min(2*n_cells+2, nphys)   # scan the lowest passband modes
    # compute V_acc for each candidate mode, keep the max (the accelerating mode)
    best=None
    for m in range(nscan):
        if not np.any(phys):
            # phys indices correspond to original positions
            pass
        # map mode m (in physical subset) to full dof
        phys_idx = np.where(phys)[0][m]
        vfull=np.zeros(A.shape[0]); vfull[free]=X[:,phys_idx]
        zi=[]; ei=[]
        for i in range(A.shape[0]):
            if abs(xyz[0,i]-side/2)<1e-9 and abs(xyz[1,i]-side/2)<1e-9:
                zi.append(xyz[2,i]); ei.append(vfull[i])
        zi=np.array(zi); ei=np.array(ei)
        o2=np.argsort(zi); zi=zi[o2]; ei=ei[o2]
        w0=2*np.pi*f[phys_idx]
        Ii=np.trapezoid(ei*np.exp(-1j*w0/c0*zi), zi)
        Ir=np.trapezoid(np.abs(ei), zi)
        T=abs(Ii)/(Ir+1e-30); Vacc=abs(Ii)
        if best is None or Vacc>best['Vacc']:
            best=dict(f0=f[phys_idx],T=T,Vacc=Vacc,grad=Vacc/Ltotal,
                      mode_index=m,naxis=len(zi))
    best['ncell']=n_cells; best['L']=Ltotal
    return best

print("="*72)
print(" MULTI-CELL π-MODE ACCELERATING STRUCTURE — voltage scaling")
print("="*72)
Rc=0.045; Lcell=0.05
side=np.sqrt(np.pi)*Rc
nose_r, nose_l = 0.40, 0.25   # optimized cell from the sweep

for ncell in [1,2,4]:
    mesh=build_ncell(side,Lcell,ncell,nz_per_cell=6,nose_radius_frac=nose_r,nose_len_frac=nose_l)
    A,B,basis,b_ext=assemble_pair(mesh)
    r=analyze(mesh,side,Lcell,ncell,nose_r,nose_l,A,B,basis,b_ext)
    if r:
        print(f" n_cells={ncell}:  L={r['L']*100:.0f}cm  f0={r['f0']/1e9:.4f}GHz  "
              f"T={r['T']:.3f}  V_acc={r['Vacc']:.4e}  grad={r['grad']:.4e} V/m (naxis={r['naxis']})", flush=True)

print()
print(" Voltage scaling with cell count (π-mode addition):")
print("   (expected: V_acc ~ N * V_1cell ; gradient roughly conserved)", flush=True)
print("DONE", flush=True)
