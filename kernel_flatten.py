#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: FIELD-FLATTENING optimization for the multi-cell pi-mode structure.
#
# The 4-cell uniform model showed the END-CELL EFFECT (end cells droop, so 4 cells
# gain less than 2). Real accelerator designs compensate by SHORTENING the end
# cells (raising their resonance, pushing their field amplitude up). Here we:
#   1. Parametrize cell lengths [L_end, L_c, L_c, L_end] (symmetric end cells).
#   2. Solve the pi-mode; measure each cell's on-axis |Ez| amplitude.
#   3. Optimize L_end (secant/bisection) until |Ez| is FLAT across cells.
#   4. Track the beam through the flattened structure and verify the gain
#      recovers toward N x single-cell.
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

def solve_variable(R, cell_lengths, nr, iris_r, borders=-1):
    """Solve axisymmetric TM modes of a variable-cell-length structure.
    cell_lengths: list of N cell lengths summed to Ltot. Irises at cumulative z.
    borders: (-1,1) PEC end caps at z=0 and z=Ltot (closed structure)."""
    Lc=np.array(cell_lengths,float); Ltot=Lc.sum()
    z_boundaries=np.concatenate([[0],np.cumsum(Lc)])
    nz_per_cell=20
    nz=nz_per_cell*len(Lc)
    # node z-positions, denser in shorter cells
    zs=[]
    for i in range(len(Lc)):
        zs.append(np.linspace(z_boundaries[i],z_boundaries[i+1],nz_per_cell+1)[:-1])
    zs.append([Ltot])
    zs=np.concatenate(zs)
    m=MeshQuad.init_tensor(np.linspace(0,R,nr+1),zs).to_meshtri()
    basis=Basis(m,ElementTriP2()); A=asm(stiff,basis); Bm=asm(mass,basis)
    facets=basis.mesh.boundary_facets(); fn=basis.mesh.facets[:,facets]
    fc=0.5*(basis.mesh.p[:,fn[0]]+basis.mesh.p[:,fn[1]])
    rmax=R*(1-1e-6); wall=facets[fc[0]>=rmax]
    # iris PEC at interior cell boundaries (r>=iris_r)
    tol=0.6*(z_boundaries[1]-z_boundaries[0])
    irises=[]
    for zb in z_boundaries[1:-1]:
        onp=np.abs(fc[1]-zb)<tol
        ir=facets[onp]; irr=ir[fc[0][onp]>=iris_r]
        irises.append(irr)
    D_facets=np.unique(np.concatenate([wall]+irises)) if irises else wall
    D_inds=np.sort(np.unique(np.asarray(basis.get_dofs(facets=D_facets).all()).ravel()))
    free=np.setdiff1d(np.arange(basis.N),D_inds)
    Aff=A[free][:,free].tocsr(); Bff=Bm[free][:,free].tocsr()
    kk=min(12,len(free)-2)
    try:
        lam,X=eigsh(Aff,k=kk,M=Bff,sigma=-1e-6,which='LM',maxiter=40000,tol=1e-10)
    except ArpackError:
        lam,X=eigsh(Aff,k=kk,M=Bff,which='SM',maxiter=60000,tol=1e-9)
    o=np.argsort(lam.real); lam=lam[o]; X=X[:,o]
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    def full(col):
        v=np.zeros(basis.N); v[free]=col; return v
    return dict(f=f,X=X,full=full,basis=basis,zb=z_boundaries,Lc=Lc,Ltot=Ltot)

def onaxis_field(sol,zg,mode_idx):
    basis=sol['basis']; v0=sol['full'](sol['X'][:,mode_idx])
    pts=np.stack([np.zeros(len(zg)),zg],axis=0)
    P=basis.probes(pts)
    return np.asarray(P@v0).ravel()

def cell_amps(sol,zg,mode_idx):
    """|Ez| amplitude at the center of each cell."""
    zb=sol['zb']; amps=[]
    for i in range(len(sol['Lc'])):
        zc=0.5*(zb[i]+zb[i+1])
        Ez=onaxis_field(sol,zg,mode_idx)
        amps.append(abs(Ez[np.argmin(np.abs(zg-zc))]))
    return np.array(amps)

def find_pi_mode(sol):
    zb=sol['zb']; ncells=len(sol['Lc'])
    zg=np.linspace(0.01*sol['Ltot'],0.99*sol['Ltot'],150)
    zcs=[0.5*(zb[i]+zb[i+1]) for i in range(ncells)]
    phys=sol['f']>1e8
    for i in range(len(phys)):
        if not phys[i]: continue
        Ez=onaxis_field(sol,zg,i)
        Ei=[Ez[np.argmin(np.abs(zg-zc))] for zc in zcs]
        if all(Ei[j]*Ei[j-1]<0 for j in range(1,len(Ei))):
            return i, np.array(Ei), zg
    return None

R=0.045; L_c=0.05; iris_r=0.40*R
ncells=4; L_std=0.05
# variable cells: [L_end, L_c, L_c, L_end]
def make_cells(L_end):
    return [L_end]+[L_c]*(ncells-2)+[L_end]

print("="*66)
print(" FIELD-FLATTENING optimization (4-cell pi-mode)")
print("="*66)
print(f"R={R*100:.2f}cm, interior cells {L_c*100:.1f}cm, iris r={iris_r*100:.2f}cm")
print(f"tuning the two symmetric END cells (length L_end)")
print()

zg_probe=np.linspace(0.02, 0.98*ncells*L_c, 200)  # approx, fixed each iter
# -------- flattening: bisection on end-cell length --------
# Physics: short enough end cells flatten the field (scan showed L_end~0.3Lc
# gives flat~1.1). Bisect L_end in [0.25,1.0]*L_c minimizing |flat-1|.
def flat_of(L_end):
    cells=make_cells(L_end)
    sol=solve_variable(R,cells,50,iris_r)
    found=find_pi_mode(sol)
    if found is None: return 999
    amps=np.abs(found[1]); return amps.max()/amps.min() if amps.min()>0 else 999

lo,hi=0.25*L_c,1.0*L_c
best=(L_c,flat_of(L_c))
for it in range(10):
    Lm=0.5*(lo+hi)
    fl=flat_of(Lm)
    print(f"  bisect {it}: L_end={Lm*100:.2f}cm flat={fl:.2f}")
    if fl<best[1]: best=(Lm,fl)
    # objective: drive flat toward 1; smaller L_end -> smaller flat
    if fl>1.05: hi=Lm   # still too drooped, shorten more
    else: lo=Lm
    if abs(best[1]-1.0)<0.06: break
L_end_flat=best[0]; flat_best=best[1]
print(f"\nFLATTENED: L_end={L_end_flat*100:.2f}cm (interior {L_c*100:.1f}cm), flat-ratio={flat_best:.2f}")
solf=solve_variable(R,make_cells(L_end_flat),50,iris_r)
found=find_pi_mode(solf); pi_idx=found[0]
amps_flat=np.abs(found[1])
print(f"flattened pi-mode cell fields: {[f'{a:.0f}' for a in amps_flat]}")

# ---- Track the beam through the FLATTENED structure ----
# Reuse the validated Lorentz tracker (as in kernel_nosetracker.py).
def track_flattened(sol,pi_idx,phi,nsteps,dt,nr_grid=60):
    f0=sol['f'][pi_idx]; w0=2*np.pi*f0; v0=sol['full'](sol['X'][:,pi_idx])
    basis=sol['basis']; Rbb=basis.doflocs[0].max(); Ltot=sol['Ltot']
    rgrid=np.linspace(0,Rbb,nr_grid+1); zgrid=np.linspace(0,Ltot,int(nr_grid*3)+1)
    pts=np.stack(np.meshgrid(rgrid,zgrid,indexing='ij'),axis=0).reshape(2,-1)
    P=basis.probes(pts)
    Ez=np.asarray(P@v0).ravel().reshape(nr_grid+1,len(zgrid))
    E_axis=np.abs(Ez[0,:]).max(); scale=25e6/E_axis if E_axis>0 else 1.0
    Ez=Ez*scale
    delta=rgrid[1]-rgrid[0]; rp=rgrid[1:-1]
    pp=np.stack(np.meshgrid(np.minimum(rp+delta,Rbb),zgrid,indexing='ij'),axis=0).reshape(2,-1)
    Pp=basis.probes(pp)
    Ezp=np.asarray(Pp@v0).ravel().reshape(len(rp),len(zgrid))*scale
    Bp=np.zeros_like(Ez); Bp[1:-1,:]=-(Ezp-Ez[1:-1,:])/(delta)/w0
    Bp[0,:]=Bp[1,:]; Bp[-1,:]=Bp[-2,:]
    ie=RegularGridInterpolator((rgrid,zgrid),Ez,bounds_error=False,fill_value=0.0)
    ib=RegularGridInterpolator((rgrid,zgrid),Bp,bounds_error=False,fill_value=0.0)
    gamma0=100e6*eV/(m_e*c0*c0); p0=np.sqrt(gamma0**2-1)*m_e*c0
    Lc0=sol['Lc'][0]; x=y=0.0; z=-0.3*Lc0; px=py=0.0; pz=p0; gamma=gamma0; m=m_e; q=-1.602e-19
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

dt=5e-12; Ltotf=solf['Ltot']; nsteps=int(4*Ltotf/c0/dt)
gmax=0; phis=np.linspace(0,2*np.pi,7)
for phi in phis:
    g=abs(track_flattened(solf,pi_idx,phi,nsteps,dt))   # abs: sign depends on crest phase
    gmax=max(gmax,g)
gmax=gmax/(1e6*eV)
print(f"flattened 4-cell auto-phased max gain = {gmax:.4f} MeV")
print(f"  (uniform 4-cell = 0.785 MeV, 2-cell = 1.083 MeV, single = 0.317 MeV)")
print(f"  multiplier vs single-cell = {gmax/0.317:.2f}x")
print("DONE",flush=True)
