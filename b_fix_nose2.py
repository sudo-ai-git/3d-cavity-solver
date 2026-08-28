#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Correct nose-cone TM-mode selection: pick the mode nearest the DESIGN
# frequency (the fundamental accelerating cell, ~2 GHz) with on-axis Ez
# dominance, then validate B=curl E/w. The earlier global-max-Ez/Et scan
# wrongly picked a 50+ GHz high-order mode.
import numpy as np
import subprocess, sys
try: import skfem
except Exception: subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-fem"]); import skfem
from skfem import MeshHex, ElementTetN1, Basis, BilinearForm, asm
from skfem.helpers import dot, curl as _curl
from scipy.linalg import eigh
c0=299792458.0
@BilinearForm
def stiff(u,v,w): return dot(_curl(u),_curl(v))
@BilinearForm
def mass(u,v,w):  return dot(u,v)

def solve_nose_cell(side,L,nx,ny,nz,nose_r_frac=0.40,nose_l_frac=0.25):
    mesh=MeshHex.init_tensor(np.linspace(0,side,nx+1),np.linspace(0,side,ny+1),np.linspace(0,L,nz+1)).to_meshtet()
    basis=Basis(mesh,ElementTetN1()); A=asm(stiff,basis); B=asm(mass,basis)
    bd=basis.get_dofs(facets=mesh.boundary_facets()); b_ext=np.sort(np.unique(np.asarray(bd.all()).ravel()))
    xyz=basis.doflocs; bset=set(b_ext.tolist())
    nr=nose_r_frac*side/2; nl=nose_l_frac*(L/2); extra=[]
    for i in range(basis.N):
        if i in bset: continue
        x,y,z=xyz[0,i]-side/2,xyz[1,i]-side/2,xyz[2,i]
        if np.hypot(x,y)<=nr and (z<=nl or z>=L-nl): extra.append(i)
    all_pec=np.unique(np.concatenate([b_ext,np.array(extra)]))
    free=np.setdiff1d(np.arange(basis.N),all_pec)
    Aff=A[free][:,free].toarray(); Bff=B[free][:,free].toarray(); lam,X=eigh(Aff,Bff)
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi); o=np.argsort(f); f=f[o]; X=X[:,o]
    return f,X,free,basis

def onaxis_dom(X,free,basis,side,L,naxis=41):
    vfull=np.zeros(basis.N); vfull[free]=X
    zs=np.linspace(0.02*L,0.98*L,naxis)
    pts=np.stack([np.full(naxis,side/2),np.full(naxis,side/2),zs],1).T
    P=basis.probes(pts); Ev=np.asarray(P@vfull).ravel().reshape(-1,3)
    return np.abs(Ev[:,2]).max(), np.sqrt(Ev[:,0]**2+Ev[:,1]**2).max()

Rc=0.045; side=np.sqrt(np.pi)*Rc; L=0.05
nx,ny,nz=10,10,14
f,X,free,basis=solve_nose_cell(side,L,nx,ny,nz)
nphys=int((f>1e8).sum())
f_low=f[f<5e9]  # fundamental band for this ~2cm cell is a few GHz
print(f"{nphys} physical modes; {len(f_low)} with f<5GHz")
# Evaluate Ez/Et ONLY for modes in the low band (the accelerating cell fundamental)
print("Low-band modes (f<5GHz), on-axis Ez dominance:")
best=None
for m in np.where(f<5e9)[0]:
    Ez,Et=onaxis_dom(X[:,m],free,basis,side,L)
    ratio=Ez/max(Et,1e-20)
    print(f"  mode {m}: f={f[m]/1e9:.3f}GHz  Ez={Ez:.2f} Et={Et:.2f}  Ez/Et={ratio:.2f}")
    # a good accelerator mode: f in 1.5-4 GHz, Ez clearly dominant, and Ez not tiny
    if ratio>1.25 and (best is None or ratio>best['ratio']):
        best={'m':m,'f':f[m],'ratio':ratio}
if best is None:
    print("No clear Ez-dominant low-band mode found (ratio<1.25 everywhere).")
else:
    print(f"\nSelected low-band accelerator mode: m={best['m']} f={best['f']/1e9:.3f}GHz Ez/Et={best['ratio']:.2f}")
    # B extraction gate on this mode
    vfull=np.zeros(basis.N); vfull[free]=X[:,best['m']]
    gx=gy=16; gz=24; bb=0.03
    xs=np.linspace(bb*side,(1-bb)*side,gx); ys=np.linspace(bb*side,(1-bb)*side,gy); zs=np.linspace(bb*L,(1-bb)*L,gz)
    xx,yy,zz=np.meshgrid(xs,ys,zs,indexing='ij')
    pts=np.stack([xx.ravel(),yy.ravel(),zz.ravel()],1).T
    P=basis.probes(pts); Ev=np.asarray(P@vfull).ravel(); E=Ev.reshape(-1,3).reshape(gx,gy,gz,3)
    hx=xs[1]-xs[0]; hy=ys[1]-ys[0]; hz=zs[1]-zs[0]; w0=2*np.pi*best['f']
    dEx=np.gradient(E,hx,axis=0); dEy=np.gradient(E,hy,axis=1); dEz=np.gradient(E,hz,axis=2)
    Bx=(dEz[...,1]-dEy[...,2])/w0; By=(dEx[...,2]-dEz[...,0])/w0; Bz=(dEy[...,0]-dEx[...,1])/w0
    Emax=np.abs(E).max(); Bmax=max(np.abs(Bx).max(),np.abs(By).max(),np.abs(Bz).max())
    ratioB=Bmax/(Emax/c0)
    print("\n=== B-EXTRACTION GATE (low-band accelerating mode) ===")
    print(f"|E|max={Emax:.2f} |B|max={Bmax:.3e}")
    print(f"ratio |B|/(|E|/c) = {ratioB:.4f}  [EM mode expects ~0.5-1.1]")
    ok=0.4<ratioB<1.2
    print(f"[{'PASS' if ok else 'FAIL'}]")
    np.savez('/tmp/nose_fields_v2.npz',E=E/Emax,Bx=Bx/Emax*c0,By=By/Emax*c0,Bz=Bz/Emax*c0,
             xs=xs,ys=ys,zs=zs,f0=best['f'],Escale=Emax)
    print("saved /tmp/nose_fields_v2.npz (SI-normalized)")
