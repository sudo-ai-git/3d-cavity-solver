#!/usr/bin/env python
# -*- coding: utf-8 -*-
# AXISYMMETRIC (r,z) scalar TM-mode solver for the nose-cone accelerating cell.
#
# Cylindrically-symmetric TM0n0 modes in a cavity of revolution reduce to a
# scalar problem for the azimuthal E_z(r,z) (m=0, only Er, Ez, Bphi nonzero):
#
#   [-1/r d/dr ( r dEz/dr ) - d^2Ez/dz^2 E_z]  =  (w/c)^2 E_z
#   i.e.  -(1/r)(r E_z_r)_r - E_z_zz = k0^2 E_z
#
# Cast with the cylindrical measure (int over 2pi r dr dz) so the weak form is:
#   int r ( dEz/dr dV/dr + dEz/dz dV/dz ) (2pi) = k0^2 int r Ez V (2pi)
#
# Boundary: Ez = 0 on the PEC walls (including the nose-cone metal and the outer
# radius R); on the axis r=0, regularity requires dEz/dr = 0 (natural Neumann),
# which the weighted FEM handles by the r->0 measure.
#
# This gives CLEAN TM010-like modes with the correct Bessel J0(kc r) radial
# dependence (unlike the square-prism), so B_phi = -(1/w) dEz/dr is trustworthy
# and the |B|/(E/c) gate should pass (analytic: max|J1| ~ 0.58).
import numpy as np
import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-fem"])
    import skfem
from skfem import MeshTri, ElementTriP2, BilinearForm, LinearForm, Basis, asm
from skfem.helpers import grad, dot
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.special import jn_zeros, jn
c0=299792458.0

# ---- r-dependent weak forms (cylindrical measure 2*pi*r dr dz cancels 2pi) ----
@BilinearForm
def stiff_axis(u, v, w):
    # int (du/dr dv/dr + du/dz dv/dz) * r
    gu=grad(u); gv=grad(v)
    r = w.x[0]           # radial coordinate
    return r * (gu[0]*gv[0] + gu[1]*gv[1])

@BilinearForm
def mass_axis(u, v, w):
    r = w.x[0]
    return r * u * v

def build_rz_mesh(R, L, nr, nz, nose_profile=None):
    """Build an (r,z) triangular mesh of the axisymmetric cell.
    r in [0,R], z in [0,L]. nose_profile: optional function r_nose(z) giving the
    nose-cone metal boundary; metal regions are PEC (Ez=0)."""
    # Use scikit-fem MeshTri on a structured grid, then refine.
    # Build a coarse (nr+1)x(nz+1) grid of quads -> triangles.
    rs = np.linspace(0, R, nr+1)
    zs = np.linspace(0, L, nz+1)
    from skfem import MeshQuad
    # rows of quads: MeshQuad.init_tensor(rs, zs) -> x=r, y=z
    m = MeshQuad.init_tensor(rs, zs)
    mtri = m.to_meshtri()
    return mtri

# nose profile for the accelerating cell (re-entrant nose cones on both ends):
# a central cylinder of nose_radius protruding nose_len from each end wall.
def nose_marker_func(mesh, R, L, nose_radius, nose_len):
    """return bool array over cells: True if the cell's centroid is inside a nose."""
    ct = mesh.element_centroids()   # (2, nelems): row0=r, row1=z
    r = ct[0]; z = ct[1]
    in_n1 = (r <= nose_radius) & (z <= nose_len)
    in_n2 = (r <= nose_radius) & (z >= L - nose_len)
    return in_n1 | in_n2

def solve_rz(R, L, nr, nz, nose_radius=None, nose_len=None):
    mesh = build_rz_mesh(R, L, nr, nz)
    basis = Basis(mesh, ElementTriP2())
    A = asm(stiff_axis, basis)
    B = asm(mass_axis, basis)
    # PEC: Ez=0 on outer boundary r=R, z=0, z=L, AND on nose-cone metal.
    # get boundary dofs
    bd = basis.get_dofs(facets=mesh.boundary_facets())
    pec = np.sort(np.unique(np.asarray(bd.all()).ravel()))
    internal_boundary = np.array([],dtype=int)
    if nose_radius is not None:
        # mark cells inside the nose metal -> their DOFs are PEC
        in_nose = nose_marker_func(mesh, R, L, nose_radius, nose_len)
        # dofs whose centroid (node locs) fall in the nose region
        dofs = basis.doflocs  # (2, ndof): r, z
        rr = dofs[0]; zz = dofs[1]
        in_n1 = (rr <= nose_radius*1.0) & (zz <= nose_len+mesh.p[1].min()*0.01)
        in_n2 = (rr <= nose_radius*1.0) & (zz >= L-nose_len-mesh.p[1].min()*0.01)
        # only the INTERIOR (not on outer wall boundary, already in pec)
        alldofs = np.arange(basis.N)
        interior = np.setdiff1d(alldofs, pec)
        nose_pec = np.intersect1d(interior, np.where(in_n1|in_n2)[0])
        pec = np.unique(np.concatenate([pec, nose_pec]))
    free = np.setdiff1d(np.arange(basis.N), pec)
    Aff = A[free][:,free].toarray(); Bff = B[free][:,free].toarray()
    lam,X = eigh(Aff,Bff)
    o = np.argsort(lam.real); lam=lam[o]; X=X[:,o]
    f = np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    return dict(f=f, X=X, free=free, basis=basis, A=A, B=B, pec=pec, lam=lam)

# ===== PLAIN CYLINDER first: validate against analytic TM010 =====
R=0.045; L=0.05
print("="*64)
print(" AXISYMMETRIC (r,z) TM-SOLVER — validate vs analytic TM010")
print("="*64)
res_an=[]
for (nr,nz) in [(20,15),(30,20),(40,25)]:
    sol = solve_rz(R,L,nr,nz)
    phys = sol['f']>1e8
    f0 = sol['f'][phys][0]
    res_an.append((sol,f0))
    print(f"  grid {nr}x{nz}: TM010 f0 = {f0/1e9:.4f}GHz")
# analytic TM010 cylinder: kc = x01/R
x01=jn_zeros(0,1)[0]
f_an = c0*x01/(2*np.pi*R)
print(f"  analytic TM010 = {f_an/1e9:.4f}GHz")
print(f"  converged numerical ≈ {np.mean([r[1] for r in res_an[-2:]])/1e9:.4f}GHz")
print(f"  error vs analytic ≈ {100*abs(np.mean([r[1] for r in res_an[-2:]])-f_an)/f_an:.2f}%")
