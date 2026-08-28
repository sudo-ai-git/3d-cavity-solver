#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: N-CELL pi-mode structure — validate the cell-count scaling law.
# Generalizes the validated 2-cell pi-mode to arbitrary N cells (N-1 irises).
#
# Question answered: does the auto-phased electron gain really scale with cell
# count (N x single-cell) in the coupled multi-cell structure, or does
# iris coupling / transit-time cause diminishing returns? Real physics check.
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

def solve_ncell(R,Ltot,nr,nz,ncells,iris_r):
    """Axisymmetric TM modes of an N-cell structure. Irises at z=k*Ltot/ncells.
    PEC: outer wall r=R + iris disks (aperture radius iris_r at each iris)."""
    Lc=Ltot/ncells
    m=MeshQuad.init_tensor(np.linspace(0,R,nr+1),np.linspace(0,Ltot,nz+1)).to_meshtri()
    basis=Basis(m,ElementTriP2()); A=asm(stiff,basis); Bm=asm(mass,basis)
    facets=basis.mesh.boundary_facets(); fn=basis.mesh.facets[:,facets]
    fc=0.5*(basis.mesh.p[:,fn[0]]+basis.mesh.p[:,fn[1]])
    rmax=R*(1-1e-6)
    wall=facets[fc[0]>=rmax]
    # iris PEC facets: at z=iris planes, r>=iris_r
    tol=0.5*Ltot/nz
    iris_facets=[]
    for k in range(1,ncells):
        zk=k*Lc
        onplane=np.abs(fc[1]-zk)<tol
        ir=facets[onplane]
        irr=ir[fc[0][onplane]>=iris_r]
        iris_facets.append(irr)
    D_facets=np.unique(np.concatenate([wall]+iris_facets))
    D_inds=np.sort(np.unique(np.asarray(basis.get_dofs(facets=D_facets).all()).ravel()))
    free=np.setdiff1d(np.arange(basis.N),D_inds)
    Aff=A[free][:,free].tocsr(); Bff=Bm[free][:,free].tocsr()
    kk=min(16,len(free)-2)
    try:
        lam,X=eigsh(Aff,k=kk,M=Bff,sigma=-1e-6,which='LM',maxiter=40000,tol=1e-10)
    except ArpackError:
        lam,X=eigsh(Aff,k=kk,M=Bff,which='SM',maxiter=60000,tol=1e-9)
    o=np.argsort(lam.real); lam=lam[o]; X=X[:,o]
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    def full(col):
        v=np.zeros(basis.N); v[free]=col; return v
    return dict(f=f,X=X,full=full,basis=basis)

def onaxis_field(sol,zg,mode_idx):
    basis=sol['basis']; v0=sol['full'](sol['X'][:,mode_idx])
    pts=np.stack([np.zeros(len(zg)),zg],axis=0)
    P=basis.probes(pts)
    return np.asarray(P@v0).ravel()

def find_pi_mode(sol,Ltot,ncells):
    """Find the pi-mode: field alternates sign across cell centers."""
    Lc=Ltot/ncells
    zg=np.linspace(0.01*Ltot,0.99*Ltot,min(300,150*ncells))
    zcs=[(i+0.5)*Lc for i in range(ncells)]
    phys=sol['f']>1e8
    for i in range(len(phys)):
        if not phys[i]: continue
        Ez=onaxis_field(sol,zg,i)
        Ei=[Ez[np.argmin(np.abs(zg-zc))] for zc in zcs]
        # pi-mode: consecutive cell fields alternate sign
        alternating=all(Ei[j]*Ei[j-1]<0 for j in range(1,len(Ei)))
        if alternating:
            return i, Ei, zg, Ez
    return None

def track_ncell(sol,pi_idx,Ltot,ncells,phi,nsteps,dt,nr_grid=100,nz_grid=200):
    """Track beam through the pi-mode field, return energy gain."""
    f0=sol['f'][pi_idx]; w0=2*np.pi*f0; v0=sol['full'](sol['X'][:,pi_idx])
    basis=sol['basis']; R=Ltot # will fix
    # R not passed; re-derive from basis doflocs
    R=basis.doflocs[0].max()
    rgrid=np.linspace(0,R,nr_grid+1); zgrid=np.linspace(0,Ltot,nz_grid+1)
    pts=np.stack(np.meshgrid(rgrid,zgrid,indexing='ij'),axis=0).reshape(2,-1)
    P=basis.probes(pts)
    Ez=np.asarray(P@v0).ravel().reshape(nr_grid+1,nz_grid+1)
    E_axis=np.abs(Ez[0,:]).max(); scale=25e6/E_axis if E_axis>0 else 1.0
    Ez=Ez*scale
    delta=rgrid[1]-rgrid[0]; rp=rgrid[1:-1]
    pp=np.stack(np.meshgrid(np.minimum(rp+delta,R),zgrid,indexing='ij'),axis=0).reshape(2,-1)
    Pp=basis.probes(pp)
    Ezp=np.asarray(Pp@v0).ravel().reshape(len(rp),len(zgrid))*scale
    Bp=np.zeros_like(Ez); Bp[1:-1,:]=-(Ezp-Ez[1:-1,:])/(delta)/w0
    Bp[0,:]=Bp[1,:]; Bp[-1,:]=Bp[-2,:]
    ie=RegularGridInterpolator((rgrid,zgrid),Ez,bounds_error=False,fill_value=0.0)
    ib=RegularGridInterpolator((rgrid,zgrid),Bp,bounds_error=False,fill_value=0.0)
    gamma0=100e6*eV/(m_e*c0*c0); p0=np.sqrt(gamma0**2-1)*m_e*c0
    Lc=Ltot/ncells
    x=y=0.0; z=-0.5*Lc; px=py=0.0; pz=p0; gamma=gamma0
    m=m_e; q=-1.602e-19
    for it in range(nsteps):
        t=it*dt; r=np.hypot(x,y)
        E_z=ie((r,z)) if 0<=z<=Ltot else 0.0
        if r>1e-9:
            B_phi=ib((r,z)); Bx=B_phi*(-y/r); By=B_phi*(x/r)
        else: Bx=By=0.0
        Bz=0.0; wtn=w0*t+phi
        Ezt=E_z*np.sin(wtn); Bxt=Bx*np.cos(wtn); Byt=By*np.cos(wtn)
        vx,vy,vz=px/(gamma*m),py/(gamma*m),pz/(gamma*m)
        cvx=vy*Bz-vz*Byt; cvy=vz*Bxt-vx*Bz; cvz=vx*Byt-vy*Bxt
        px+=q*dt*(cvx); py+=q*dt*(cvy); pz+=q*dt*(Ezt+cvz)
        p2=px*px+py*py+pz*pz; gamma=np.sqrt(1+p2/(m*c0)**2)
        x+=vx*dt; y+=vy*dt; z+=vz*dt
    return (gamma-gamma0)*m*c0*c0

R=0.045; Lcell=0.05; iris_r=0.40*R; SingleGain=0.317
print("="*66)
print(" N-CELL pi-MODE accelerating structure - scaling law")
print("="*66)
print(f"R={R*100:.2f}cm, cell len {Lcell*100:.1f}cm, iris aperture r={iris_r*100:.2f}cm")
results=[]
for ncells in [2,4]:
    Ltot=ncells*Lcell
    sol=solve_ncell(R,Ltot,80,120,ncells,iris_r)
    phys=sol['f']>1e8
    print(f"\n[{ncells} cells, Ltot={Ltot*100:.1f}cm] lowest modes:")
    for i in range(min(6,len(phys))):
        print(f"  mode {i}: f={sol['f'][i]/1e9:.4f} GHz")
    found=find_pi_mode(sol,Ltot,ncells)
    if found is None:
        print("  !! pi-mode not found in first 10"); results.append((ncells,None)); continue
    pi_idx,Ei,zg,Ezpi=found
    print(f"  pi-mode: mode {pi_idx}, f={sol['f'][pi_idx]/1e9:.4f}GHz, cell fields={[f'{e:+.1f}' for e in Ei]}")
    # auto-phase sweep
    dt=5e-12; nsteps=int(4*Ltot/c0/dt)
    gmax=0
    phis=np.linspace(0,2*np.pi,9)
    for phi in phis:
        dE=track_ncell(sol,pi_idx,Ltot,ncells,phi,nsteps,dt)
        g=dE/(1e6*eV)
        gmax=max(gmax,abs(g))
    mult=gmax/SingleGain
    results.append((ncells,gmax,mult))
    print(f"  auto-phased max |gain| = {gmax:.4f} MeV ; multiplier={mult:.3f}x")

print("\n=== SCALING LAW SUMMARY ===")
for r in results:
    if r[1] is not None:
        print(f"  {r[0]} cells: {r[1]:.4f} MeV  ({r[2]:.2f}x single-cell)")
print("DONE",flush=True)
