#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: AXISYMMETRIC (r,z) TM010 solver for the nose-cone accelerating cell.
#  - Correct BC (Dirichlet only on r=R) validated vs analytic TM010 (2.55 GHz)
#  - B = -(1/w) dEz/dr validated vs Bessel (|B|/(E/c) ~ 0.58 gate)
#  - Nose-cone cell: add re-entrant PEC noses on the end caps, recompute mode+B
import numpy as np
import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-fem"])
    import skfem
print(f"scikit-fem {skfem.__version__}", flush=True)
from skfem import MeshQuad, ElementTriP2, BilinearForm, Basis, asm
from skfem.helpers import grad
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, ArpackError
from scipy.sparse import eye as spyeye
from scipy.special import jn_zeros, jv
c0=299792458.0

@BilinearForm
def stiff(u,v,w):
    gu=grad(u); gv=grad(v); return w.x[0]*(gu[0]*gv[0]+gu[1]*gv[1])
@BilinearForm
def mass(u,v,w):
    return w.x[0]*u*v

def solve_rz(R,L,nr,nz,nose_r=None,nose_len=None):
    rs=np.linspace(0,R,nr+1); zs=np.linspace(0,L,nz+1)
    m=MeshQuad.init_tensor(rs,zs).to_meshtri()
    basis=Basis(m,ElementTriP2())
    A=asm(stiff,basis); Bm=asm(mass,basis)
    facets=basis.mesh.boundary_facets()
    fn=basis.mesh.facets[:,facets]
    fc=0.5*(basis.mesh.p[:,fn[0]]+basis.mesh.p[:,fn[1]])
    rmax=R*(1-1e-6)
    wall_facets=facets[fc[0]>=rmax]
    pec_extra=[]
    if nose_r is not None:
        doflocs=basis.doflocs; rr=doflocs[0]; zz=doflocs[1]
        in1=rr<=nose_r; in2=zz<=nose_len; in3=zz>=L-nose_len
        nose=np.where((in1 & in2) | (in1 & in3))[0]
        pec_extra=nose
    D_inds=np.sort(np.unique(np.concatenate([np.asarray(basis.get_dofs(facets=wall_facets).all()).ravel(),pec_extra])))
    free=np.setdiff1d(np.arange(basis.N),D_inds)
    Aff=A[free][:,free].tocsr(); Bff=Bm[free][:,free].tocsr()
    # Sparse lowest-mode solve (shift-invert; A is symmetric pos-def on free DOFs)
    kk=min(6, len(free)-2)
    sigma=-1e-6   # near 0 -> find smallest (k0^2) eigenvalues
    try:
        lam, Xsp = eigsh(Aff, k=kk, M=Bff, sigma=sigma, which='LM', maxiter=20000, tol=1e-10)
    except ArpackError:
        lam, Xsp = eigsh(Aff, k=kk, M=Bff, which='SM', maxiter=60000, tol=1e-9)
    o=np.argsort(lam.real); lam=lam[o]; Xsp=Xsp[:,o]
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    def full(col):
        v=np.zeros(basis.N); v[free]=col; return v
    return dict(f=f,X=Xsp,free=free,basis=basis,D=D_inds,full=full,R=R,L=L)

def B_extract(v0,basis,R,L,w0):
    """Bphi = -(1/w) dEz/dr from the solved Ez field on a radial grid.
    Uses a thick mid-z band, dedups r by averaging |Ez|, subsamples for even spacing."""
    rs=basis.doflocs[0]; zs=basis.doflocs[1]
    bandmask=np.array([abs(z-L/2)<0.35*L for z in zs])
    idx=np.where(bandmask)[0]
    idx=idx[np.argsort(rs[idx])]
    r_=rs[idx]; Ez=abs(v0[idx])
    # dedup r: average |Ez| over the (r,multi-z) bins
    r_key=np.round(r_,10)
    uniq_r=np.unique(r_key)
    rinv=np.unique(r_key, return_inverse=True)[1]
    sums=np.zeros(len(uniq_r)); cnts=np.zeros(len(uniq_r))
    np.add.at(sums, rinv, Ez); np.add.at(cnts, rinv, 1.0)
    Ez_bin=sums/np.maximum(cnts,1)
    # subsample for even spacing (np.gradient needs strictly monotonic r)
    r_bin=uniq_r
    if len(r_bin)>3:
        step=max(1,len(r_bin)//60)
        r_bin=r_bin[::step]; Ez_bin=Ez_bin[::step]
    dEzdr=np.gradient(Ez_bin, r_bin)   # d|Ez|/dr (sign handled by abs; magnitude ok)
    Bphi=-dEzdr/w0
    return r_bin, Bphi

print("="*66)
print(" AXISYMMETRIC TM-SOLVER on Kaggle GPU")
print("="*66)
R=0.045; L=0.05
x01=jn_zeros(0,1)[0]; kc=x01/R; f_an=c0*x01/(2*np.pi*R)
print(f"analytic TM010: f={f_an/1e9:.4f} GHz  (R={R*100:.1f}cm L={L*100:.1f}cm)")
print()
# B gate: robust on-axis Ez = global max over all r=0 nodes
def gate_check(sol, v0, R, L, w0, label):
    rs0=sol['basis'].doflocs[0]
    onax=np.where(rs0<1e-9)[0]
    E_axis=abs(v0[onax]).max() if len(onax)>0 else abs(v0).max()
    r_,Bphi=B_extract(v0,sol['basis'],R,L,w0)
    ratio=np.abs(Bphi).max()/(E_axis/c0) if E_axis>0 else float('nan')
    print(f"  {label}: B/(E/c) gate = {ratio:.4f}   {'PASS' if 0.4<ratio<1.2 else 'FAIL'}")
    return ratio

print("[1] PLAIN CYLINDER validation:")
nr=nz=100   # fine on GPU
sol=solve_rz(R,L,nr,nz)
phys=sol['f']>1e8
f0=sol['f'][phys][0]
print(f"  grid {nr}x{nz}: f0={f0/1e9:.4f} GHz  ratio={f0/f_an:.5f}  {'PASS' if abs(f0/f_an-1)<0.005 else 'FAIL'}")
v0=sol['full'](sol['X'][:,np.where(phys)[0][0]])
w0=2*np.pi*f0
print(f"  analytic max|J1| = {np.max(np.abs(jv(1,kc*np.linspace(1e-9,R,300)))):.4f}")
gate_check(sol, v0, R, L, w0, "plain cylinder")
print()
print("[2] NOSE-CONE cell (nose r=40% R, len=25% L):")
nose_r=0.40*R; nose_len=0.25*L
sol_n=solve_rz(R,L,nr,nz,nose_r,nose_len)
phys_n=sol_n['f']>1e8
f0n=sol_n['f'][phys_n][0]
print(f"  nose-cone f0 = {f0n/1e9:.4f} GHz")
v0n=sol_n['full'](sol_n['X'][:,np.where(phys_n)[0][0]])
w0n=2*np.pi*f0n
gate_check(sol_n, v0n, R, L, w0n, "nose-cone cell")
print()
print("[3] Transit-time / V_acc for nose-cone cell (on-axis Ez):")
# on-axis Ez(z) profile -> V_acc = |int Ez dz|, T = V_acc/(E_axis*L)
rsn=sol_n['basis'].doflocs[0]; zsn=sol_n['basis'].doflocs[1]
onaxmask=np.array([abs(r)<1e-3 for r in rsn])
iz=np.where(onaxmask)[0]; iz=iz[np.argsort(zsn[iz])]
Ezn=v0n[iz]; zzn=zsn[iz]
Vacc=abs(np.trapezoid(Ezn, zzn))
Emax_n=np.abs(Ezn).max()
T=Vacc/(Emax_n*L) if Emax_n>0 else 0
print(f"  on-axis V_acc = {Vacc:.4e} (arb), peak Ez={Emax_n:.2e}, T = {T:.3f}")
print("DONE", flush=True)
