#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: MULTI-CELL π-MODE accelerating structure (axisymmetric).
# Builds on the VALIDATED axisymmetric (r,z) TM solver. The geometry is a
# 2-cell structure: total length 2L, with a central IRIS (thin PEC disk at
# z=L with a beam aperture of radius r_iris) that couples the two cells.
#
# Physics: the two cells support coupled modes. The π-mode has the electric
# field ANTI-PHASED across the cells (oscillating 180° out of phase in time),
# so a relativistic beam crossing cell 1 in its accelerating half-cycle enters
# cell 2 exactly one RF half-period later, where the anti-phased field gives
# the SAME acceleration sign -> voltages ADD. Total gain ~ N x single-cell.
#
# We solve the axisymmetric TM modes, identify the π-mode field pattern,
# and track a beam through both cells with the validated Lorentz pusher.
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
from scipy.interpolate import RegularGridInterpolator
c0=299792458.0; m_e=9.1093837015e-31; eV=1.602176634e-19

@BilinearForm
def stiff(u,v,w):
    gu=grad(u); gv=grad(v); return w.x[0]*(gu[0]*gv[0]+gu[1]*gv[1])
@BilinearForm
def mass(u,v,w):
    return w.x[0]*u*v

def solve_multicell(R,Ltot,nr,nz,iris_z,iris_r,nose_r_frac=0.0,nose_len=0.0):
    """Solve axisymmetric TM modes of a multi-cell cavity.
    PEC: outer wall r=R; iris at z=iris_z (disk with aperture radius iris_r);
    optional nose cones on the end caps.
    Returns modes with the field anti-phasing structure exposed."""
    zs=np.linspace(0,Ltot,nz+1)
    m=MeshQuad.init_tensor(np.linspace(0,R,nr+1),zs).to_meshtri()
    basis=Basis(m,ElementTriP2())
    A=asm(stiff,basis); Bm=asm(mass,basis)
    facets=basis.mesh.boundary_facets()
    fn=basis.mesh.facets[:,facets]
    fc=0.5*(basis.mesh.p[:,fn[0]]+basis.mesh.p[:,fn[1]])
    # PEC: outer wall (r=R), and iris disk (r>=iris_r at z=iris_z)
    rmax=R*(1-1e-6)
    wall=facets[fc[0]>=rmax]
    iris=facets[np.abs(fc[1]-iris_z)< (Ltot/nz)*1.5]  # facets on the iris plane
    # iris facets exclude the aperture (r<iris_r)
    iris_pec=facets[np.abs(fc[1]-iris_z)< (Ltot/nz)*1.5]
    iris_pec_r=iris_pec[fc[0][np.abs(fc[1]-iris_z)< (Ltot/nz)*1.5] >= iris_r]
    # combine outer wall + iris (with aperture hole)
    D_facets=np.unique(np.concatenate([wall, iris_pec_r]))
    dl=basis.doflocs; rr=dl[0]; zz=dl[1]
    # nose PEC if requested
    extra=[]
    if nose_len>0:
        nrq=nose_r_frac*R
        extra=np.where(((rr<=nrq)&(zz<=nose_len))|((rr<=nrq)&(zz>=Ltot-nose_len)))[0]
    D_inds=np.sort(np.unique(np.concatenate([np.asarray(basis.get_dofs(facets=D_facets).all()).ravel(),extra])))
    free=np.setdiff1d(np.arange(basis.N),D_inds)
    Aff=A[free][:,free].tocsr(); Bff=Bm[free][:,free].tocsr()
    kk=min(12,len(free)-2)
    lam,X=eigsh(Aff,k=kk,M=Bff,sigma=-1e-6,which='LM',maxiter=30000,tol=1e-10)
    o=np.argsort(lam.real); lam=lam[o]; X=X[:,o]
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    def full(col):
        v=np.zeros(basis.N); v[free]=col; return v
    return dict(f=f,X=X,full=full,basis=basis,free=free)

def onaxis_field(sol, zg, mode_idx):
    """Return Ez(z) on-axis for a given mode."""
    basis=sol['basis']
    v0=sol['full'](sol['X'][:,mode_idx])
    pts=np.stack([np.zeros(len(zg)),zg],axis=0)
    P=basis.probes(pts)
    return np.asarray(P@v0).ravel()

R=0.045; Lcell=0.05; Ltot=2*Lcell   # 2 cells
iris_r=0.40*R        # beam aperture radius (iris hole)
iris_z=Lcell         # central iris plane
print("="*66)
print(" MULTI-CELL 2-MODE structure (axisymmetric, iris coupling)")
print("="*66)
print(f"R={R*100:.2f}cm, 2 cells each {Lcell*100:.1f}cm, Ltot={Ltot*100:.1f}cm")
print(f"central iris at z={iris_z*100:.1f}cm, aperture r={iris_r*100:.2f}cm")

sol=solve_multicell(R,Ltot,80,100,iris_z,iris_r)
phys=sol['f']>1e8
fmodes=sol['f'][phys]
print(f"\nlowest {len(fmodes)} physical modes:")
for i in range(min(6,len(fmodes))):
    print(f"  mode {i}: f={fmodes[i]/1e9:.4f} GHz")
print()
# Identify modes by on-axis phase structure (0-mode in-phase vs pi-mode anti-phase)
zg=np.linspace(0.01*Ltot,0.99*Ltot,200)
zc1=Lcell/2; zc2=3*Lcell/2
print("on-axis Ez phase structure per mode:")
pi_mode=None
for i in range(min(4,len(fmodes))):
    Ez=onaxis_field(sol,zg,i)
    E1=Ez[np.argmin(np.abs(zg-zc1))]; E2=Ez[np.argmin(np.abs(zg-zc2))]
    phase="in-phase (0-mode)" if (E1*E2)>0 else "ANTI-PHASE (pi-mode)"
    if phase.startswith("ANTI") and pi_mode is None:
        pi_mode=i
    print(f"  mode {i}: f={fmodes[i]/1e9:.4f}GHz  E(c1)={E1:+.2f} E(c2)={E2:+.2f} -> {phase}")

# ---- Beam tracking through the PI-MODE ----
# We need the full E(r,z) and B(r,z) for the pi-mode on a grid, then the Lorentz
# tracker (as in kernel_nosetracker.py). A relativistic beam crosses cell1 in the
# accelerating phase and cell2 (anti-phased) in the next half-period - same sign.
print("\nBuilding pi-mode E/B field grid + tracking beam...")
pi_idx=pi_mode if pi_mode is not None else 1
f0=sol['f'][pi_idx]; w0=2*np.pi*f0
v0=sol['full'](sol['X'][:,pi_idx])
basis=sol['basis']
# field grid over full length
nr_grid=80; nz_grid=160
rgrid=np.linspace(0,R,nr_grid+1); zgrid=np.linspace(0,Ltot,nz_grid+1)
pts=np.stack(np.meshgrid(rgrid,zgrid,indexing='ij'),axis=0).reshape(2,-1)
P=basis.probes(pts)
Ez=np.asarray(P@v0).ravel().reshape(nr_grid+1,nz_grid+1)
# physical normalization: on-axis peak Ez
E_onaxis_peak=np.abs(Ez[0,:]).max()
scale=25e6/E_onaxis_peak if E_onaxis_peak>0 else 1.0   # 25 MV/m
Ez=Ez*scale
# Bphi = -(1/w)dEz/dr probe-based
delta=rgrid[1]-rgrid[0]
rp=rgrid[1:-1]
pp=np.stack(np.meshgrid(np.minimum(rp+delta,R),zgrid,indexing='ij'),axis=0).reshape(2,-1)
Pp=basis.probes(pp)
Ezp=np.asarray(Pp@v0).ravel().reshape(len(rp),len(zgrid))*scale
Bp=np.zeros_like(Ez)
Bp[1:-1,:]=-(Ezp-Ez[1:-1,:])/(delta)/w0
Bp[0,:]=Bp[1,:]; Bp[-1,:]=Bp[-2,:]
ie=RegularGridInterpolator((rgrid,zgrid),Ez,bounds_error=False,fill_value=0.0)
ib=RegularGridInterpolator((rgrid,zgrid),Bp,bounds_error=False,fill_value=0.0)
print(f"pi-mode f0={f0/1e9:.4f}GHz, on-axis peak Ez scaled to 25 MV/m")

def track_pi(phi,nsteps,dt):
    gamma0=100e6*eV/(m_e*c0*c0); p0=np.sqrt(gamma0**2-1)*m_e*c0
    x=y=0.0; z=-0.5*Lcell; px=py=0.0; pz=p0; gamma=gamma0; m=m_e; q=-1.602e-19
    for it in range(nsteps):
        t=it*dt; r=np.hypot(x,y)
        E_z=ie((r,z)) if 0<=z<=Ltot else 0.0
        if r>1e-9:
            B_phi=ib((r,z)); Bx=B_phi*(-y/r); By=B_phi*(x/r)
        else: Bx=By=0.0
        Bz=0.0
        wtn=w0*t+phi
        Ezt=E_z*np.sin(wtn); Bxt=Bx*np.cos(wtn); Byt=By*np.cos(wtn)
        vx,vy,vz=px/(gamma*m),py/(gamma*m),pz/(gamma*m)
        cvx=vy*Bz-vz*Byt; cvy=vz*Bxt-vx*Bz; cvz=vx*Byt-vy*Bxt
        px+=q*dt*(cvx); py+=q*dt*(cvy); pz+=q*dt*(Ezt+cvz)
        p2=px*px+py*py+pz*pz; gamma=np.sqrt(1+p2/(m*c0)**2)
        x+=vx*dt; y+=vy*dt; z+=vz*dt
    return (gamma-gamma0)*m*c0*c0

# auto-phase sweep
dt=5e-12; nsteps=int(4*Ltot/c0/dt)
print("\n  phi(rad) | electron gain (MeV)  [pi-mode, 2 cells]")
gains=[]
for phi in np.linspace(0,2*np.pi,9):
    dE=track_pi(phi,nsteps,dt); g=dE/(1e6*eV); gains.append((phi,g))
    print(f"  {phi:7.3f}  | {g:+8.4f}")
gmax=max(gains,key=lambda x:x[1])
print(f"\npi-mode auto-phased max gain = {gmax[1]:+.4f} MeV (2 cells)")
print(f"single-cell reference = 0.317 MeV")
print(f"voltage multiplier = {gmax[1]/0.317:.3f}x")
print("DONE",flush=True)
