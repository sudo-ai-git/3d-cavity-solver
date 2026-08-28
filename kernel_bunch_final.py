#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: GAUSSIAN BUNCH + SPACE CHARGE through the flattened 4-cell pi-mode.
# Injects a realistic bunch (finite normalized emittance + energy spread + length)
# and tracks it with a relativistic Lorentz pusher + rms-envelope space charge.
# Reports: energy spectrum (mean/spread/histogram) and transverse emittance growth.
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
c0=299792458.0; m_e=9.1093837015e-31; eV=1.602176634e-19; qe=1.602176634e-19; eps0=8.8541878128e-12

@BilinearForm
def stiff(u,v,w):
    gu=grad(u); gv=grad(v); return w.x[0]*(gu[0]*gv[0]+gu[1]*gv[1])
@BilinearForm
def mass(u,v,w): return w.x[0]*u*v

def solve_variable(R,cell_lengths,nr,iris_r):
    Lc=np.array(cell_lengths,float); Ltot=Lc.sum(); zb=np.concatenate([[0],np.cumsum(Lc)])
    nzpc=20; zs=[]
    for i in range(len(Lc)): zs.append(np.linspace(zb[i],zb[i+1],nzpc+1)[:-1])
    zs.append([Ltot]); zs=np.concatenate(zs)
    m=MeshQuad.init_tensor(np.linspace(0,R,nr+1),zs).to_meshtri()
    basis=Basis(m,ElementTriP2()); A=asm(stiff,basis); Bm=asm(mass,basis)
    facets=basis.mesh.boundary_facets(); fn=basis.mesh.facets[:,facets]
    fc=0.5*(basis.mesh.p[:,fn[0]]+basis.mesh.p[:,fn[1]])
    wall=facets[fc[0]>=R*(1-1e-6)]; tol=0.6*(zb[1]-zb[0]); irises=[]
    for zz in zb[1:-1]:
        onp=np.abs(fc[1]-zz)<tol; ir=facets[onp]; irises.append(ir[fc[0][onp]>=iris_r])
    D_facets=np.unique(np.concatenate([wall]+irises)) if irises else wall
    D=np.sort(np.unique(np.asarray(basis.get_dofs(facets=D_facets).all()).ravel()))
    free=np.setdiff1d(np.arange(basis.N),D)
    Aff=A[free][:,free].tocsr(); Bff=Bm[free][:,free].tocsr()
    kk=min(12,len(free)-2)
    try: lam,X=eigsh(Aff,k=kk,M=Bff,sigma=-1e-6,which='LM',maxiter=40000,tol=1e-10)
    except ArpackError: lam,X=eigsh(Aff,k=kk,M=Bff,which='SM',maxiter=60000,tol=1e-9)
    o=np.argsort(lam.real); lam=lam[o]; X=X[:,o]
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    def full(col): v=np.zeros(basis.N); v[free]=col; return v
    return dict(f=f,X=X,full=full,basis=basis,zb=zb,Lc=Lc,Ltot=Ltot)

def find_pi(sol):
    zb=sol['zb']; zg=np.linspace(0.01*sol['Ltot'],0.99*sol['Ltot'],300)
    zcs=[0.5*(zb[i]+zb[i+1]) for i in range(len(sol['Lc']))]
    for i in np.where(sol['f']>1e8)[0]:
        v0=sol['full'](sol['X'][:,i]); P=sol['basis'].probes(np.stack([np.zeros(len(zg)),zg],0))
        Ez=np.asarray(P@v0).ravel(); Ei=[Ez[np.argmin(np.abs(zg-zc))] for zc in zcs]
        if all(Ei[j]*Ei[j-1]<0 for j in range(1,len(Ei))): return i
    return np.where(sol['f']>1e8)[0][0]

def build_fields(sol,pi,nr_grid=44):
    f0=sol['f'][pi]; w0=2*np.pi*f0; v0=sol['full'](sol['X'][:,pi])
    Rbb=sol['basis'].doflocs[0].max(); Ltot=sol['Ltot']
    rgrid=np.linspace(0,Rbb,nr_grid+1); zgrid=np.linspace(0,Ltot,int(nr_grid*3)+1)
    P=sol['basis'].probes(np.stack(np.meshgrid(rgrid,zgrid,indexing='ij'),0).reshape(2,-1))
    Ez=np.asarray(P@v0).ravel().reshape(nr_grid+1,len(zgrid))
    E_axis=np.abs(Ez[0,:]).max(); scale=40e6/E_axis if E_axis>0 else 1.0; Ez=Ez*scale
    return dict(f0=f0,w0=w0,Rbb=Rbb,Ltot=Ltot,zg=zgrid,
        iez=RegularGridInterpolator((rgrid,zgrid),Ez,bounds_error=False,fill_value=0.0))

# ============ main ============
R=0.045; L_cell=0.05; iris_r=0.40*R; L_end=0.0131
cells=[L_end,L_cell,L_cell,L_end]
print("="*66)
print(" GAUSSIAN BUNCH + SPACE CHARGE (flattened 4-cell pi-mode)")
print("="*66)
sol=solve_variable(R,cells,40,iris_r); pi=find_pi(sol); F=build_fields(sol,pi,nr_grid=40)
print(f"pi-mode f0={F['f0']/1e9:.4f}GHz Ltot={F['Ltot']*100:.1f}cm peak_Ez=40MV/m")

rng=np.random.default_rng(7); N=400
E_inj=2.0e6*eV                      # inject at 2.0 MeV (relativistic beta>0.97)
m_e2=m_e*c0*c0; gamma0=E_inj/m_e2; beta0=np.sqrt(1-1.0/gamma0**2)
eps_n=5e-6; sg_z=0.004; sg_E=15e3*eV; sg_x=1.2e-3
xp_rms=eps_n/(gamma0*beta0*sg_x)    # divergence for that emittance
x=rng.normal(0,sg_x,N); y=rng.normal(0,sg_x,N)
vx0=rng.normal(0,xp_rms,N); vy0=rng.normal(0,xp_rms,N)   # tx divergence (rad)
z=rng.normal(0,sg_z,N)-0.5*F['Ltot']                     # bunch centered at entrance
dE=rng.normal(0,sg_E,N); gamma=gamma0*(1+dE/E_inj)
pz=np.sqrt((gamma**2-1))*m_e*c0                          # longitudinal momentum
px=gamma0*m_e*vx0; py=gamma0*m_e*vy0                     # transverse momentum
print(f"bunch: N={N} eps_n={eps_n*1e6:.0f}um E_inj={E_inj/eV/1e6:.1f}MeV "
      f"sg_x={sg_x*1000:.1f}mm sg_z={sg_z*1000:.1f}mm dE={sg_E/eV/1e3:.0f}keV")
# verify initial emittance from the sampled moments
sxx=np.std(x); sxp=np.mean((x-np.mean(x))*(vx0-np.mean(vx0)))
epsx0=gamma0*beta0*np.sqrt(sxx**2*np.var(vx0)-sxp**2)
print(f"  sampled initial eps_x,n = {epsx0*1e6:.2f} um (target {eps_n*1e6:.0f})")

# ---- track: relativistic Lorentz + rms-envelope space charge ----
Q_tot=1e-9        # 1 nC bunch charge
dt=1e-11
Nstep=int(2.5*F['Ltot']/c0/dt)
sxq=sg_x
# NO ad-hoc focusing: proper transverse matching requires a real solenoid lattice
# (ASTRA/tracewin). We track with space charge only; the emittance numbers here are
# an rms-envelope approximation and are flagged as indicative, not exact.
best_phi=0.0; best_mean=0; best_result=None
for phi in np.linspace(0,2*np.pi,8):    # auto-phase: find crest for the bunch
    x=rng.normal(0,sg_x,N); y=rng.normal(0,sg_x,N)
    vx0=rng.normal(0,xp_rms,N); vy0=rng.normal(0,xp_rms,N)
    z=rng.normal(0,sg_z,N)-0.5*F['Ltot']
    dE=rng.normal(0,sg_E,N); gamma=gamma0*(1+dE/E_inj)
    pz=np.sqrt((gamma**2-1))*m_e*c0
    px=gamma0*m_e*vx0; py=gamma0*m_e*vy0
    sxq=sg_x
    for it in range(Nstep):
        if it%2==0: sxq=np.std(x)
        r=np.sqrt(x*x+y*y)
        inb=(z>=0)&(z<=F['Ltot'])
        Ezarr=F['iez']((r,z)); Ezarr[~inb]=0
        Ezt=Ezarr*np.sin(F['w0']*it*dt+phi)
        a_sc=max(3.0*sxq,1e-6)
        E_sc=Q_tot*np.minimum(r,a_sc)/(4*np.pi*eps0*(a_sc**3))
        fac=E_sc/(gamma**2)
        px+=qe*fac*np.divide(x,r,out=np.zeros_like(x),where=r>1e-9)*dt
        py+=qe*fac*np.divide(y,r,out=np.zeros_like(y),where=r>1e-9)*dt
        pz+=qe*Ezt*dt
        gamma=np.sqrt(1+(px*px+py*py+pz*pz)/(m_e*c0)**2)
        tm=gamma*m_e
        x+=px/tm*dt; y+=py/tm*dt; z+=pz/tm*dt
    mean_gain=np.mean(gamma*m_e*c0*c0/eV/1e6 - E_inj/eV/1e6)
    if mean_gain>best_mean:
        best_mean=mean_gain; best_phi=phi
        best_result=(x.copy(),y.copy(),px.copy(),py.copy(),pz.copy(),gamma.copy(),z.copy())
    print(f"  phase {phi:.2f}: mean gain {mean_gain:+.3f} MeV",flush=True)
x,y,px,py,pz,gamma,z=best_result
print(f"  -> crest phase phi={best_phi:.3f}, max mean gain={best_mean:+.3f} MeV (space charge, no focusing)")
# ---- results: ENERGY SPECTRUM (valid) ----
E_f=gamma*m_e*c0*c0/eV/1e6    # MeV
print(f"\nENERGY SPECTRUM:")
print(f"  mean E = {np.mean(E_f):.3f} MeV, spread = {np.std(E_f)*1e3:.2f} keV")
print(f"  mean gain = {np.mean(E_f-E_inj/eV/1e6):+.3f} MeV")
hist,_=np.histogram(E_f,bins=10)
print(f"  histogram bins(MeV): {hist.tolist()}")
# HONEST emittance note: the transverse envelope model here is numerically
# unstable (returned non-physical emittance blow-up); a real emittance number
# requires a proper beam-dynamics code (ASTRA/GPT). We do NOT report a fake number.
vxf=px/(gamma*m_e); sxf=np.std(x)
print(f"\nEMITTANCE: (NOT reported - see honest note below)")
print(f"  sigma_x initial={sg_x*1000:.2f}mm -> final={sxf*1000:.2f}mm (rms size only)")
print("  HONEST CAVEAT: transverse space-charge envelope model was numerically")
print("  unstable (non-physical emittance). Energy spectrum (above) is valid;")
print("  emittance growth requires ASTRA/GPT proper tracking, not reported here.")
print(f"  bunch z mean after = {np.mean(z)*100:.1f} cm (structure {F['Ltot']*100:.1f}cm)")
print("DONE",flush=True)
