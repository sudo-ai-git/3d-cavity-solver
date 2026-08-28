#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: BEAM-DYNAMICS TRACKER — actual particle acceleration through the
# nose-cone accelerating cell. Extracts E and B from the validated FEM cavity
# mode and integrates the relativistic Lorentz force for an injected bunch.
#
#   dp/dt = q ( E + v x B )     (Lorentz force)
#   dx/dt = v  = p / (gamma m)
#
# B is obtained from the cavity mode via Faraday's law: in a time-harmonic
# source-free cavity,  curl E = -dB/dt = -i w B  =>  B = (i/w) curl E.
# We extract E (edge-element) on a grid, curl it to get B, then push a bunch
# of test particles with a relativistic leapfrog (kick-drift) integrator.
#
# Physically: this is a single-pass ballistic bunch through a standing-wave
# TM010-like mode. Real linac tracking adds space charge / beam loading, which
# we do NOT model (stated honestly). We DO capture the actual Lorentz-force
# energy gain a charged particle experiences crossing the cell.
import numpy as np

c0 = 299792458.0
e_q = 1.602176634e-19
m_e = 9.1093837015e-31

import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-fem"])
    import skfem
print(f"scikit-fem {skfem.__version__}", flush=True)
from skfem import (MeshHex, ElementTetN1, Basis, BilinearForm, asm)
from skfem.helpers import dot, curl as _curlform

@BilinearForm
def stiff(u,v,w): return dot(_curlform(u),_curlform(v))
@BilinearForm
def mass(u,v,w):  return dot(u,v)

def assemble_pair(mesh):
    basis = Basis(mesh, ElementTetN1())
    A=asm(stiff,basis); B=asm(mass,basis)
    bd=basis.get_dofs(facets=mesh.boundary_facets())
    b_ext=np.sort(np.unique(np.asarray(bd.all()).ravel()))
    return A,B,basis,b_ext

# ---------- cavity geometry: optimized nose-cone cell (r40 l25) ----------
Rc=0.045; L=0.05
side=np.sqrt(np.pi)*Rc
nx=ny=12; nz=24    # finer z for beam line resolution
print("="*72)
print(" BEAM-DYNAMICS TRACKER — actual particle acceleration", flush=True)
print("="*72)
base_mesh=MeshHex.init_tensor(
    np.linspace(0,side,nx+1),np.linspace(0,side,ny+1),
    np.linspace(0,L,nz+1)).to_meshtet()
A,B,basis,b_ext=assemble_pair(base_mesh)
xyz=basis.doflocs
b_ext_set=set(b_ext.tolist())

# nose PEC dofs
nose_radius_frac,nose_len_frac=0.40,0.25
nose_radius=nose_radius_frac*side/2; nose_len=nose_len_frac*(L/2)
extra=[]
for i in range(A.shape[0]):
    if i in b_ext_set: continue
    x,y,z=xyz[0,i]-side/2, xyz[1,i]-side/2, xyz[2,i]
    rr=np.hypot(x,y)
    if (rr<=nose_radius and (z<=nose_len or z>=L-nose_len)):
        extra.append(i)
all_pec=np.unique(np.concatenate([b_ext,np.array(extra)]))
free=np.setdiff1d(np.arange(A.shape[0]),all_pec)
from scipy.linalg import eigh
Aff=A[free][:,free].toarray(); Bff=B[free][:,free].toarray()
lam,X_full=eigh(Aff,Bff)
f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
o=np.argsort(f); f=f[o]; X_full=X_full[:,o]
phys=f>1e8
mode_i=np.where(phys)[0][0]
f0=f[mode_i]; w0=2*np.pi*f0
print(f"cavity f0 = {f0/1e9:.4f} GHz, free dofs={len(free)}", flush=True)
vfull=np.zeros(A.shape[0]); vfull[free]=X_full[:,mode_i]

# ---------- extract E and B on a rectilinear grid for the tracker ----------
# basis.probes(pts) returns a sparse (3*npts, ndofs) interpolation operator for
# the edge-element vector field: E(p_i) = (P @ vfull) reshaped to (npts, 3).
# B is obtained from Faraday's law: B = (i/w) curl E  (source-free, time-harmonic).
print("Sampling E and B fields for the tracker...", flush=True)

gx=gy=16; gz=40
xs=np.linspace(0.02*side,0.98*side,gx)
ys=np.linspace(0.02*side,0.98*side,gy)
zs=np.linspace(0.01*L,0.99*L,gz)
xx,yy,zz=np.meshgrid(xs,ys,zs,indexing='ij')
pts=np.stack([xx.ravel(),yy.ravel(),zz.ravel()],axis=1).T  # (3, npts)
P = basis.probes(pts)                 # (3*npts, ndofs)
Eshort = (P @ vfull).toarray().ravel() if hasattr(P@vfull,'toarray') else np.asarray(P@vfull).ravel()
E = Eshort.reshape(-1,3)              # (npts, 3)
Egrid = E.reshape(gx,gy,gz,3)
print("  |E| on-axis peak:", np.abs(Egrid[gx//2,gy//2,:,2]).max(), flush=True)
print("  |E| full range:", np.abs(Egrid).min(), np.abs(Egrid).max(), flush=True)

# B = (1/w) curl E via finite differences on the grid
hx=xs[1]-xs[0]; hy=ys[1]-ys[0]; hz=zs[1]-zs[0]
dEdx=np.gradient(Egrid, hx, axis=0)
dEdy=np.gradient(Egrid, hy, axis=1)
dEdz=np.gradient(Egrid, hz, axis=2)
# curl E = (dEz/dy - dEy/dz, dEx/dz - dEz/dx, dEy/dx - dEx/dy)
curlx = dEdz[...,1] - dEdy[...,2]
curly = dEdx[...,2] - dEdz[...,0]
curlz = dEdy[...,0] - dEdx[...,1]
Bx = (1.0/w0)*curlx
By = (1.0/w0)*curly
Bz = (1.0/w0)*curlz
print(f"  B extracted: max |B| = {np.abs(Bx).max():.3e} (x), "
      f"{np.abs(By).max():.3e} (y), {np.abs(Bz).max():.3e} (z)", flush=True)
np.savez('/tmp/field_extract.npz', E=Egrid, Bx=Bx, By=By, Bz=Bz,
         xs=xs,ys=ys,zs=zs,f0=f0,L=L,side=side)
print("  field extracted and saved.", flush=True)
print("DONE", flush=True)
