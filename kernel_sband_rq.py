#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: S-band nose-cone cell — R_over_Q and gain with/without nose cones.
# Uses the validated axisymmetric (r,z) TM010 solver on the S-band (2.998 GHz)
# geometry from the linac design, ADDS nose cones, and computes:
#   - on-axis V_acc (beam gain per cell)
#   - stored energy U -> R_over_Q = V_acc^2 / (w U)   [formula (defines R/Q)]
#   - r_sh = R_over_Q * Q0  (removes the 'assumed 60 MOhm/m')
#   - gain comparison plain vs nose-cone cell at same peak E.
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
    return dict(f=f,X=X,full=full,basis=basis,A=A,B=Bm,D=D)

def cell_metrics(sol):
    """f0, on-axis V_acc, R_over_Q, r_sh for the TM010 mode (scaled to 40 MV/m).
    R/Q = V_acc^2/(w U); U = (1/2) eps0 * <v0,B v0> (2pi from cylindrical measure).
    eigenvector is B-normalized: <v0,B v0> = 1 -> U_eigen = 0.5*eps0*2pi*1."""
    phys=sol['f']>1e8
    f0=sol['f'][phys][0]; w0=2*np.pi*f0
    v0=sol['full'](sol['X'][:,0])
    basis=sol['basis']
    # stored energy in eigen units (B-normalized, cylindrical 2pi measure):
    U_eigen = 0.5*eps0*2*np.pi              # Joules in eigen normalization
    zg=np.linspace(0.01*basis.doflocs[1].max(),0.99*basis.doflocs[1].max(),300)
    pts=np.stack([np.zeros(len(zg)),zg],axis=0)
    P=basis.probes(pts); Ez_ax=np.asarray(P@v0).ravel()
    V_acc_unnorm=abs(np.trapezoid(Ez_ax,zg))
    E_axis_peak=np.abs(Ez_ax).max()
    scale=(40e6/E_axis_peak) if E_axis_peak>0 else 1.0
    V_acc=V_acc_unnorm*scale                # physical volts @ 40 MV/m peak
    U=U_eigen*scale**2                       # physical J (field^2 scales)
    R_over_Q=V_acc**2/(w0*U)                 # Ohm
    Q0=20000.0; L=basis.doflocs[1].max()
    r_sh_per_m=R_over_Q*Q0/L                 # Ohm/m
    return dict(f0=f0,V_acc=V_acc,R_over_Q=R_over_Q,r_sh_per_m=r_sh_per_m)

F=2.998e9; lam=c0/F
R=x01=jn_zeros(0,1)[0]*c0/(2*np.pi*F)   # 3.83 cm
L=lam/2                                    # 5.00 cm
print("="*66)
print(" S-BAND CELL: R/Q and nose-cone gain (validated solver)")
print("="*66)
print(f"geometry (2.998GHz): R={R*100:.2f}cm, L={L*100:.2f}cm (lambda/2), iris-none(cavity)")
for label,nr,nz,nose in [("plain",60,60,None),("nose r40 l25",60,60,(0.40*R,0.10*L))]:
    sol=solve(R,L,nr,nz,nose[0] if nose else None, nose[1] if nose else None)
    m=cell_metrics(sol)
    print(f"\n{label} cell: f0={m['f0']/1e9:.4f}GHz, V_acc={m['V_acc']/1e6:.3f}MV "
          f"(@40MV/m), R_over_Q={m['R_over_Q']:.1f} Ohm, "
          f"r_sh={m['r_sh_per_m']/1e6:.2f} MOhm/m (Q0=20000 assumed)")
print("\nnote: R_over_Q is geometry-only; r_sh here uses Q0=20000 (typical Cu S-band).")
print("DONE",flush=True)
