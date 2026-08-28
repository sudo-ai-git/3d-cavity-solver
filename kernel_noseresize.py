#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: NOSE-CONE RESIZE — find outer radius R that holds f0=2.998 GHz
# with nose cones, then compare V_acc / R/Q vs the plain cell at the SAME freq.
# This answers: does adding nose cones (with R re-sized to hold frequency) give a
# higher accelerating voltage for the same peak E? (the 'stronger' design trade)
import numpy as np
import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-fem"])
    import skfem
from skfem import MeshQuad, ElementTriP2, BilinearForm, Basis, asm
from skfem.helpers import grad
from scipy.sparse.linalg import eigsh, ArpackError
from scipy.special import jn_zeros
c0=299792458.0; m_e=9.1093837015e-31; eV=1.602176634e-19; eps0=8.8541878128e-12

@BilinearForm
def stiff(u,v,w):
    gu=grad(u); gv=grad(v); return w.x[0]*(gu[0]*gv[0]+gu[1]*gv[1])
@BilinearForm
def mass(u,v,w):
    return w.x[0]*u*v

def solve(R,L,nr,nz,nose_r=None,nose_len=None):
    rs=np.linspace(0,R,nr+1); zs=np.linspace(0,L,nz+1)
    m=MeshQuad.init_tensor(rs,zs).to_meshtri()
    basis=Basis(m,ElementTriP2()); A=asm(stiff,basis); Bm=asm(mass,basis)
    facets=basis.mesh.boundary_facets(); fn=basis.mesh.facets[:,facets]
    fc=0.5*(basis.mesh.p[:,fn[0]]+basis.mesh.p[:,fn[1]])
    wall=facets[fc[0]>=R*(1-1e-6)]
    extra=[]
    if nose_r is not None:
        dl=basis.doflocs; rr=dl[0]; zz=dl[1]
        extra=np.where(((rr<=nose_r)&(zz<=nose_len))|((rr<=nose_r)&(zz>=L-nose_len)))[0]
    D=np.sort(np.unique(np.concatenate([np.asarray(basis.get_dofs(facets=wall).all()).ravel(),extra])))
    free=np.setdiff1d(np.arange(basis.N),D)
    Aff=A[free][:,free].tocsr(); Bff=Bm[free][:,free].tocsr()
    kk=min(8,len(free)-2)
    try:
        lam,X=eigsh(Aff,k=kk,M=Bff,sigma=-1e-6,which='LM',maxiter=40000,tol=1e-10)
    except ArpackError:
        lam,X=eigsh(Aff,k=kk,M=Bff,which='SM',maxiter=60000,tol=1e-9)
    o=np.argsort(lam.real); lam=lam[o]; X=X[:,o]
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    def full(col):
        v=np.zeros(basis.N); v[free]=col; return v
    return dict(f=f,X=X,full=full,basis=basis)

def cell_metrics(sol,Epeak=40e6,Q0=20000.0):
    phys=sol['f']>1e8; f0=sol['f'][phys][0]; w0=2*np.pi*f0
    v0=sol['full'](sol['X'][:,0]); basis=sol['basis']
    U_eigen=0.5*eps0*2*np.pi
    zg=np.linspace(0.01*basis.doflocs[1].max(),0.99*basis.doflocs[1].max(),300)
    pts=np.stack([np.zeros(len(zg)),zg],axis=0)
    P=basis.probes(pts); Ez=np.asarray(P@v0).ravel()
    V0=abs(np.trapezoid(Ez,zg)); Epk=np.abs(Ez).max()
    scale=Epeak/Epk if Epk>0 else 1.0
    V=V0*scale; U=U_eigen*scale**2; Rq=V**2/(w0*U)
    L=basis.doflocs[1].max()
    return dict(f0=f0,V_acc=V,R_over_Q=Rq,r_sh=Rq*Q0/L)

def f0_at(R,L,nr,nz,nfr,nfl):
    nrq=nfr*R; nlen=nfl*L
    sol=solve(R,L,nr,nz,nrq,nlen)
    return sol['f'][sol['f']>1e8][0]

F=2.998e9; lam=c0/F; L=lam/2
R0=jn_zeros(0,1)[0]*c0/(2*np.pi*F)
print("="*66)
def f0_at_R(R,L,nr,nz,nose_r_abs,nose_len_abs):
    sol=solve(R,L,nr,nz,nose_r_abs,nose_len_abs)
    return sol['f'][sol['f']>1e8][0]

def R_for_freq_abs(L,nr,nz,nose_r_abs,nose_len_abs,target=F):
    """Bisect outer radius R so f0->target, HOLDING nose geometry FIXED (absolute
    dims). This is real cavity synthesis: pick nose, then size outer radius."""
    lo,hi=R0*0.8, R0*4.0
    for _ in range(16):
        Rm=0.5*(lo+hi)
        f0=f0_at_R(Rm,L,nr,nz,nose_r_abs,nose_len_abs)
        if f0>target: hi=Rm
        else: lo=Rm
    Rf=0.5*(lo+hi)
    return Rf, f0_at_R(Rf,L,nr,nz,nose_r_abs,nose_len_abs)

# baseline plain cell at target
solp=solve(R0,L,50,50)
mp=cell_metrics(solp)
print(" NOSE-CONE RESIZE (hold 2.998 GHz, nose geometry fixed, enlarge R)")
print("="*66)
print(f"R0(plain)={R0*100:.2f}cm, L={L*100:.2f}cm (lambda/2)")
print(f"PLAIN cell:   f0={mp['f0']/1e9:.4f} V_acc={mp['V_acc']/1e6:.3f}MV "
      f"R/Q={mp['R_over_Q']:.0f} r_sh={mp['r_sh']/1e6:.1f}MOhm/m")
print()
# nose designs: choose nose absolute dims from a fraction of the PLAIN cell R
# so the nose is a meaningful, fixed aperture; then enlarge outer R to hold f0.
for nfr,nfl in [(0.30,0.15),(0.40,0.10),(0.40,0.15),(0.50,0.08)]:
    nose_r_abs=nfr*R0; nose_len_abs=nfl*L
    R_res,f0r=R_for_freq_abs(L,50,50,nose_r_abs,nose_len_abs)
    sol=solve(R_res,L,50,50,nose_r_abs,nose_len_abs)
    m=cell_metrics(sol)
    dv=m['V_acc']/mp['V_acc']; drq=m['R_over_Q']/mp['R_over_Q']
    print(f"nose r={nfr:.2f}R0 l={nfl:.2f}L: R={R_res*100:.2f}cm f0={m['f0']/1e9:.4f} "
          f"V_acc={m['V_acc']/1e6:.3f}MV({dv:.2f}x) R/Q={m['R_over_Q']:.0f}({drq:.2f}x)")
print("\nDONE",flush=True)
