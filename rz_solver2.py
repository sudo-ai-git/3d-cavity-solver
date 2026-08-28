#!/usr/bin/env python
# -*- coding: utf-8 -*-
# AXISYMMETRIC (r,z) TM010 scalar solver — CORRECTED boundary conditions.
#
# Physics: TM0n0 modes have only E_z, E_r, B_phi nonzero (m=0, axial symmetry).
# Scalar equation for E_z(r,z):
#   -(1/r) d/dr(r dEz/dr) - d^2Ez/dz^2 = (w/c)^2 Ez
# weak form (measure 2*pi*r dr dz):
#   int r (dEz/dr dV/dr + dEz/dz dV/dz) = k0^2 int r Ez V
#
# BOUNDARY CONDITIONS (the fix):
#   * r=R (outer wall): E_z is TANGENTIAL -> must be 0 (Dirichlet).   [J0(kc R)=0]
#   * z=0, z=L (end caps): E_z is NORMAL -> NOT required to vanish. The actual
#     cap condition is E_r=0 (tangential E at cap), which for TM010 (E_r=0 as
#     m=0) is auto-satisfied. So NO Dirichlet on the caps -> natural condition.
#   * r=0 (axis): regularity -> natural (handled by r measure).
#   => Dirichlet ONLY on r=R.
# This yields the pure TM010: Ez=J0(kc r), f = c x01/(2 pi R) = 2.55 GHz for R=4.5cm.
#
# (My earlier bug: imposed E_z=0 on the end caps too, which forced a z-variation
#  and raised the lowest mode to 4.1GHz. The caps are a magnetic-wall-like plane
#  for TM010, not an E_z=0 plane.)
import numpy as np
import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-fem"])
    import skfem
from skfem import MeshQuad, ElementTriP2, BilinearForm, Basis, asm, condense
from skfem.helpers import grad, dot
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.special import jn_zeros, jv
c0=299792458.0

@BilinearForm
def stiff(u,v,w):
    gu=grad(u); gv=grad(v)
    return w.x[0]*(gu[0]*gv[0]+gu[1]*gv[1])
@BilinearForm
def mass(u,v,w):
    return w.x[0]*u*v

def solve_rz(R, L, nr, nz, nose_r=None, nose_len=None):
    """Solve TM010-like mode. nose_r/nose_len: if given, place PEC nose cones on
    the z-caps (a central cylinder of radius nose_r protruding nose_len) and
    force E_z=0 inside them (Dirichlet)."""
    rs=np.linspace(0,R,nr+1); zs=np.linspace(0,L,nz+1)
    m=MeshQuad.init_tensor(rs,zs).to_meshtri()
    basis=Basis(m,ElementTriP2())
    A=asm(stiff,basis); Bm=asm(mass,basis)
    # Dirichlet on r=R (outer wall): facets whose centroid r ~ R
    facets = basis.mesh.boundary_facets()
    # facet centroid = mean of its 2 endpoint nodes
    fn = basis.mesh.facets[:, facets]                    # (2, nbf) node indices
    fc = 0.5*(basis.mesh.p[:, fn[0]] + basis.mesh.p[:, fn[1]])   # (2,nbf): r,z
    rmax = R*(1-1e-6)
    wall_facets = facets[fc[0] >= rmax]
    # optional nose PEC: nodes inside the nose volume
    pec_extra=[]
    if nose_r is not None:
        doflocs=basis.doflocs  # (2, ndof): r,z
        rr=doflocs[0]; zz=doflocs[1]
        in_n1=(rr<=nose_r)&(zz<=nose_len)
        in_n2=(rr<=nose_r)&(zz>=L-nose_len)
        pec_extra=np.where(in_n1|in_n2)[0]
    D_inds = np.sort(np.unique(np.concatenate([np.asarray(basis.get_dofs(facets=wall_facets).all()).ravel(), pec_extra])))
    # condense: solve interior eigenproblem with Dirichlet DOFs -> 0
    free = np.setdiff1d(np.arange(basis.N), D_inds)
    Aff = A[free][:,free].toarray(); Bff = Bm[free][:,free].toarray()
    lam,X = eigh(Aff,Bff)
    o=np.argsort(lam.real); lam=lam[o]; X=X[:,o]
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    # full vector (zero on Dirichlet)
    def full_vector(col):
        v=np.zeros(basis.N); v[free]=col; return v
    return dict(f=f, X=X, free=free, basis=basis, D_inds=D_inds, full=full_vector)

# ===== validate: plain cylinder vs analytic TM010 =====
R=0.045; L=0.05
print("="*64)
print(" AXISYMMETRIC TM-SOLVER (fixed BC: Dirichlet only on r=R)")
print("="*64)
x01=jn_zeros(0,1)[0]; kc=x01/R
f_an=c0*x01/(2*np.pi*R)
print(f"analytic TM010: f = c*x01/(2*pi*R) = {f_an/1e9:.4f} GHz")
print()
for (nr,nz) in [(15,10),(25,15),(35,20),(50,25)]:
    sol=solve_rz(R,L,nr,nz)
    phys=sol['f']>1e8
    f0=sol['f'][phys][0]
    ratio=f0/f_an
    print(f"  grid {nr}x{nz}: f0 = {f0/1e9:.4f} GHz   (ratio vs analytic {ratio:.4f})")
print()
# field check: on the lowest mode, Ez ~ J0(kc r) (z-independent)
sol=solve_rz(R,L,35,20)
phys=sol['f']>1e8
v0=sol['full'](sol['X'][:,np.where(phys)[0][0]])
basis=sol['basis']
# sample Ez on r (at z=L/2)
zs=basis.doflocs[1]; rs=basis.doflocs[0]
zmid=np.argmin(np.abs(zs-L/2)); zmid=np.where(np.abs(zs-np.max(zs)/2)<0.001)[0]
print(f"field check: Ez(r) at z=L/2 vs J0(kc r)")
rr=np.linspace(0,0.9*R,20)
for r in rr:
    i=np.argmin(np.abs(rs-r))
    if zs[i]>L/2-0.001 and zs[i]<L/2+0.001:
        print(f"  r={r*100:.2f}cm: |Ez|={abs(v0[i]):.3f}  J0(kc r)={abs(jv(0,kc*r)):.3f}")
print()
# B/(E/c) gate: Bphi = -(1/w) dEz/dr on-axis->0; max at mid-radius ~ J1
rg=np.linspace(1e-9,R,300)
Bmax_an=np.max(np.abs(jv(1,kc*rg)))
print(f"analytic B/(E/c) = max|J1| = {Bmax_an:.4f}  (gate target)")
